from __future__ import annotations
import numpy as np

from signalforge.bitstream.metrics import periodicity_score


def _score(bits: np.ndarray) -> float:
    b = np.asarray(bits, dtype=np.uint8)
    if b.size < 32:
        return 0.0
    periods = [1, 2, 4, 8, 16, 32]
    scores = [periodicity_score(b, p) for p in periods if p < b.size // 2]
    return float(max(scores, default=0.0))


def rank_candidates(bits: np.ndarray, max_block: int = 64) -> list[dict]:
    """
    Generate structural candidates. The score is only a heuristic ordering signal.
    """
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    candidates = []

    for rows in range(2, min(max_block, b.size) + 1):
        cols = b.size // rows
        if cols >= 2 and rows * cols == b.size:
            from signalforge.deinterleave.block import block_deinterleave
            x = block_deinterleave(b, rows, cols)
            candidates.append({"method": "BLOCK", "rows": rows, "cols": cols, "score": _score(x), "bits": x})

    for depth in range(2, min(32, b.size) + 1):
        from signalforge.deinterleave.convolutional import convolutional_deinterleave
        x = convolutional_deinterleave(b, depth)
        candidates.append({"method": "CONVOLUTIONAL", "depth": depth, "score": _score(x), "bits": x})

    return sorted(candidates, key=lambda x: x["score"], reverse=True)
