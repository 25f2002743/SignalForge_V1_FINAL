from __future__ import annotations
import json
from signalforge.ingestion.loader import load_signal
from signalforge.preprocessing.pipeline import prepare
from signalforge.observatory.engine import observe
from signalforge.parameters.estimate import extract_parameters
from signalforge.modulation.advisor import advise
from signalforge.hypotheses.generator import generate
from signalforge.hypotheses.scorer import rank_hypotheses
from signalforge.hypotheses.executor import execute_hypothesis
from signalforge.demod.engine import demodulate
from signalforge.bitstream.metrics import bit_statistics
from signalforge.deinterleave.engine import try_deinterleavers
from signalforge.fec.engine import decode_candidates
from signalforge.frame.engine import analyze_frames
from signalforge.validation.validator import validate_chain


def analyze(path: str, **kwargs) -> dict:
    raw = load_signal(path, **kwargs)
    prepared = prepare(raw)
    observations = observe(prepared)
    parameters = extract_parameters(prepared)
    modulation = advise(prepared)
    hypotheses = rank_hypotheses(generate(prepared))

    executions = []
    for h in hypotheses[:3]:
        dsp = execute_hypothesis(prepared, h)
        if dsp.get("status") in ("FAILED", "BLOCKED"):
            executions.append({"hypothesis": h.as_dict(), "dsp": {k: v for k, v in dsp.items() if k != "symbols"}})
            continue

        symbols = dsp["symbols"]
        demod = demodulate(prepared, h.modulation, symbols, samples_per_symbol=dsp["integer_samples_per_symbol"])
        deints = try_deinterleavers(demod.hard_bits, top_k=3)
        fec_results = decode_candidates(demod.hard_bits, demod.soft_bits)

        best_deint = deints[0] if deints else None
        best_fec = fec_results[0] if fec_results else None
        corrected_bits = best_fec.corrected_bits if best_fec and best_fec.success else (
            best_deint.bits if best_deint else demod.hard_bits
        )

        frames = analyze_frames(corrected_bits)
        frame_score = max((c["score"] for c in frames["candidates"]), default=0.0)
        deint_score = best_deint.score if best_deint else 0.0
        fec_score = best_fec.confidence if best_fec else 0.0

        validation = validate_chain(
            demod.confidence,
            deint_score,
            fec_score,
            frame_score,
        )

        executions.append({
            "hypothesis": h.as_dict(),
            "dsp": {k: v for k, v in dsp.items() if k != "symbols"},
            "demodulation": demod.as_dict(),
            "bit_statistics": bit_statistics(demod.hard_bits),
            "deinterleaving_candidates": [x.as_dict() for x in deints],
            "fec_candidates": [x.as_dict() for x in fec_results],
            "frame_analysis": frames,
            "validation": validation,
        })

    return {
        "record": prepared.summary(),
        "provenance": prepared.provenance,
        "observation_keys": list(observations.keys()),
        "parameters": {k: v.as_dict() for k, v in parameters.items()},
        "modulation_candidates": modulation["candidates"],
        "hypotheses": [h.as_dict() for h in hypotheses],
        "executions": executions,
    }


def main(path: str, **kwargs) -> None:
    print(json.dumps(analyze(path, **kwargs), indent=2, default=str))
