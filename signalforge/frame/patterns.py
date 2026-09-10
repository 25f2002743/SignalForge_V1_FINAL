from __future__ import annotations
import numpy as np


def parse_pattern(value: str | list[int] | np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        s = value.replace(" ", "").replace("_", "")
        if not s or any(c not in "01" for c in s):
            raise ValueError("Pattern must be a binary string.")
        return np.fromiter((int(c) for c in s), dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8).reshape(-1)


def find_pattern(bits: np.ndarray, pattern: np.ndarray, max_errors: int = 0) -> list[dict]:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    p = parse_pattern(pattern)
    if p.size == 0 or b.size < p.size:
        return []

    out = []
    for i in range(b.size - p.size + 1):
        dist = int(np.sum(b[i:i+p.size] != p))
        if dist <= max_errors:
            out.append({
                "start": i,
                "end": i + p.size,
                "hamming_distance": dist,
                "match_score": float(1.0 - dist / p.size),
            })
    return out


def find_repeated_patterns(bits: np.ndarray, min_length: int = 8, max_length: int = 64) -> list[dict]:
    """
    Finds exact repeated substrings. Useful as an exploratory framing clue.
    """
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    results = []
    max_length = min(max_length, b.size // 2)
    for length in range(min_length, max_length + 1):
        seen = {}
        for i in range(b.size - length + 1):
            key = bytes(b[i:i+length])
            if key in seen and i - seen[key] >= length:
                results.append({
                    "length": length,
                    "first": seen[key],
                    "second": i,
                    "score": 1.0,
                })
            else:
                seen[key] = i
    return results
