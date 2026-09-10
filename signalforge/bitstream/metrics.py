from __future__ import annotations
import numpy as np


def bit_statistics(bits: np.ndarray) -> dict:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size == 0:
        return {"length": 0, "ones": 0, "zeros": 0, "one_ratio": None}
    ones = int(np.sum(b))
    return {
        "length": int(b.size),
        "ones": ones,
        "zeros": int(b.size - ones),
        "one_ratio": float(ones / b.size),
        "transition_ratio": float(np.mean(b[1:] != b[:-1])) if b.size > 1 else 0.0,
    }


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.uint8).reshape(-1)
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    n = min(a.size, b.size)
    return int(np.sum(a[:n] != b[:n]) + abs(a.size - b.size))


def periodicity_score(bits: np.ndarray, period: int) -> float:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if period <= 0 or b.size <= period:
        return 0.0
    return float(np.mean(b[:-period] == b[period:]))
