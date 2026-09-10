from __future__ import annotations
import numpy as np


def convolutional_deinterleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Inverse of a simple periodic branch-delay interleaver.
    This is a structural V1 candidate, not a claim about an unknown protocol.
    """
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    branches = [b[i::depth] for i in range(depth)]
    out = []
    for i in range(max(map(len, branches))):
        for branch in branches:
            if i < len(branch):
                out.append(branch[i])
    return np.asarray(out, dtype=np.uint8)
