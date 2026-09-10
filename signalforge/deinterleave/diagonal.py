from __future__ import annotations
import numpy as np


def diagonal_deinterleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if rows <= 0 or cols <= 0 or rows * cols > b.size:
        raise ValueError("Invalid diagonal dimensions.")

    mat = b[:rows * cols].reshape(rows, cols)
    out = []
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                out.append(mat[r, c])
    return np.asarray(out, dtype=np.uint8)
