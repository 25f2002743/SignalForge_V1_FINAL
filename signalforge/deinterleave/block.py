from __future__ import annotations
import numpy as np


def block_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if rows <= 0 or cols <= 0 or rows * cols > b.size:
        raise ValueError("Invalid block dimensions.")
    n = rows * cols
    matrix = b[:n].reshape(rows, cols)
    return matrix.T.reshape(-1)
