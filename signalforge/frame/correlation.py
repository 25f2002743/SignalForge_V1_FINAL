from __future__ import annotations
import numpy as np


def normalized_bit_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = 2.0 * np.asarray(a, dtype=np.uint8).reshape(-1) - 1.0
    y = 2.0 * np.asarray(b, dtype=np.uint8).reshape(-1) - 1.0
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    x, y = x[:n], y[:n]
    den = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return float(np.sum(x*y) / max(den, 1e-12))


def sliding_correlation(bits: np.ndarray, reference: np.ndarray, step: int = 1) -> list[dict]:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    r = np.asarray(reference, dtype=np.uint8).reshape(-1)
    if r.size == 0 or b.size < r.size:
        return []
    return [
        {
            "offset": i,
            "score": normalized_bit_correlation(b[i:i+r.size], r),
        }
        for i in range(0, b.size - r.size + 1, max(1, step))
    ]
