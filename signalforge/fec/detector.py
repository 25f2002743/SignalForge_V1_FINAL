from __future__ import annotations
import numpy as np


def rank_fec_candidates(bits: np.ndarray, soft_bits: np.ndarray | None = None) -> list[dict]:
    """
    V1 FEC detector is an evidence/availability layer.
    It does not infer an FEC scheme from insufficient evidence.
    """
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    candidates = [
        {"code": "CONVOLUTIONAL", "available": True, "score": 0.25},
        {"code": "RS", "available": False, "score": 0.0},
        {"code": "LDPC", "available": False, "score": 0.0},
        {"code": "CONCATENATED", "available": False, "score": 0.0},
    ]
    if soft_bits is not None and len(soft_bits) >= 64:
        candidates[0]["score"] = 0.40
    return candidates
