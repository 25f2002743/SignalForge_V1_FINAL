from __future__ import annotations
import numpy as np


SUPPORTED = ("BPSK", "QPSK", "8PSK", "FSK", "16QAM")


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    keys = list(scores)
    vals = np.asarray([scores[k] for k in keys], dtype=float)
    vals -= np.max(vals)
    ex = np.exp(vals)
    probs = ex / max(np.sum(ex), 1e-12)
    return {k: float(p) for k, p in zip(keys, probs)}


def rule_based_advisor(features: dict) -> dict:
    """
    Lightweight V1 modulation advisor.

    It deliberately produces ranked candidates rather than declaring ground truth.
    The hypothesis/DSP validation layers remain authoritative.
    """
    if not features["complex"]:
        # Real WAV has limited direct phase/constellation evidence.
        return {
            k: float(v)
            for k, v in zip(SUPPORTED, [0.10, 0.10, 0.05, 0.55, 0.20])
        }

    cv = features["amplitude_cv"]
    phase_std = features["phase_diff_std_rad"]
    fi_std = features["instantaneous_frequency_std_hz"]

    scores = {k: 0.0 for k in SUPPORTED}

    # Constant-envelope families: PSK/FSK are favored over QAM.
    scores["BPSK"] += 1.0 * max(0.0, 1.0 - cv * 8)
    scores["QPSK"] += 1.0 * max(0.0, 1.0 - cv * 8)
    scores["8PSK"] += 0.8 * max(0.0, 1.0 - cv * 8)

    # Variable envelope is evidence toward QAM.
    scores["16QAM"] += 1.8 * min(cv * 8, 1.5)

    # Strong instantaneous-frequency variation is evidence toward FSK.
    scores["FSK"] += 1.4 * min(fi_std / max(fi_std + 1.0, 1.0), 1.0)

    # Very stable phase transitions slightly favor lower-order PSK.
    scores["BPSK"] += max(0.0, 0.8 - phase_std)
    scores["QPSK"] += max(0.0, 0.6 - phase_std) * 0.8

    return _softmax(scores)


def rank_candidates(probabilities: dict[str, float]) -> list[dict]:
    return [
        {"modulation": k, "score": float(v)}
        for k, v in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    ]
