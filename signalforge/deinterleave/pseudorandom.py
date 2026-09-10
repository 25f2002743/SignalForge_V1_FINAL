from __future__ import annotations
import numpy as np


def pseudorandom_deinterleave(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    p = np.asarray(permutation, dtype=int).reshape(-1)
    if p.size != b.size or set(p.tolist()) != set(range(b.size)):
        raise ValueError("permutation must contain every input index exactly once.")
    return b[p]
