from __future__ import annotations
import numpy as np

from signalforge.deinterleave.result import DeinterleaveResult
from signalforge.deinterleave.detector import rank_candidates


def try_deinterleavers(bits: np.ndarray, top_k: int = 5) -> list[DeinterleaveResult]:
    candidates = rank_candidates(bits)
    results = []
    for c in candidates[:top_k]:
        results.append(
            DeinterleaveResult(
                method=c["method"],
                bits=c["bits"],
                score=float(c["score"]),
                metrics={k: v for k, v in c.items() if k not in ("bits", "score", "method")},
                evidence=["Candidate selected by structural bitstream heuristic."],
            )
        )
    return results
