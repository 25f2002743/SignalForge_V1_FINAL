from __future__ import annotations


def validate_chain(
    demod_confidence: float,
    deinterleave_score: float,
    fec_confidence: float,
    frame_score: float,
) -> dict:
    from signalforge.validation.score import candidate_quality

    score = candidate_quality(
        demod_confidence,
        deinterleave_score,
        fec_confidence,
        frame_score,
    )
    if score >= 0.80:
        status = "STRONG"
    elif score >= 0.55:
        status = "PLAUSIBLE"
    else:
        status = "WEAK"

    return {
        "score": score,
        "status": status,
        "components": {
            "demodulation": demod_confidence,
            "deinterleaving": deinterleave_score,
            "fec": fec_confidence,
            "framing": frame_score,
        },
    }
