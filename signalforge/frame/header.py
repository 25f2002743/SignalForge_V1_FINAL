from __future__ import annotations
import numpy as np

from signalforge.frame.models import FrameCandidate
from signalforge.frame.patterns import find_pattern


def detect_header_candidates(
    bits: np.ndarray,
    known_preambles: list[str] | None = None,
    max_errors: int = 1,
) -> list[FrameCandidate]:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    candidates = []

    for preamble in known_preambles or []:
        matches = find_pattern(b, preamble, max_errors=max_errors)
        for m in matches:
            # Conservative default: header is the first 32 bits after preamble
            # only if enough bits exist; otherwise leave boundaries unknown.
            header_start = m["end"]
            header_end = min(b.size, header_start + 32)
            payload_start = header_end if header_end < b.size else None
            payload_end = b.size if payload_start is not None else None

            candidates.append(FrameCandidate(
                start_bit=m["start"],
                end_bit=b.size,
                header_start=header_start if header_end > header_start else None,
                header_end=header_end if header_end > header_start else None,
                payload_start=payload_start,
                payload_end=payload_end,
                score=m["match_score"],
                evidence=[
                    "Known preamble matched within configured Hamming tolerance.",
                    "Header/payload boundaries are provisional unless protocol grammar confirms them.",
                ],
                metadata={"preamble": preamble, "preamble_hamming_distance": m["hamming_distance"]},
            ))

    return sorted(candidates, key=lambda x: x.score, reverse=True)
