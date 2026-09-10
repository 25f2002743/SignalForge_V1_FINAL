from __future__ import annotations
from signalforge.core.signal_record import SignalRecord
from signalforge.hypotheses.model import Hypothesis
from signalforge.dsp.chain import run_dsp_chain


def execute_hypothesis(
    record: SignalRecord,
    hypothesis: Hypothesis,
    samples_per_symbol: int = 4,
) -> dict:
    if hypothesis.symbol_rate_hz is None:
        hypothesis.status = "BLOCKED"
        hypothesis.contradictions.append("No symbol-rate candidate available.")
        return {"status": "BLOCKED", "reason": "missing_symbol_rate"}

    try:
        result = run_dsp_chain(
            record,
            symbol_rate_hz=hypothesis.symbol_rate_hz,
            modulation=hypothesis.modulation,
            samples_per_symbol=samples_per_symbol,
        )
        hypothesis.status = "EXECUTED"
        hypothesis.evidence.append({
            "type": "DSP_EXECUTION",
            "estimated_cfo_hz": result["estimated_cfo_hz"],
            "timing_metric": result["timing_metric"],
            "symbol_count": int(result["symbols"].size),
        })
        return result
    except Exception as exc:
        hypothesis.status = "FAILED"
        hypothesis.contradictions.append(str(exc))
        return {"status": "FAILED", "error": str(exc)}
