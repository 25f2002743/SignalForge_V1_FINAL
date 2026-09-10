from __future__ import annotations
import numpy as np

from signalforge.frame.header import detect_header_candidates
from signalforge.frame.parser import parse_candidate_payload


def analyze_frames(
    bits: np.ndarray,
    known_preambles: list[str] | None = None,
    max_errors: int = 1,
) -> dict:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    candidates = detect_header_candidates(b, known_preambles, max_errors=max_errors)

    payloads = []
    for c in candidates:
        if c.payload_start is not None:
            payloads.append(parse_candidate_payload(b, c.payload_start, c.payload_end))

    return {
        "bit_count": int(b.size),
        "candidate_count": len(candidates),
        "candidates": [c.as_dict() for c in candidates],
        "payload_views": payloads,
        "status": "SUCCESS" if candidates else "UNOBSERVABLE",
        "note": "No framing is asserted when protocol-specific evidence is absent.",
    }
