from __future__ import annotations
import numpy as np


def bits_to_bytes(bits: np.ndarray, bit_order: str = "msb") -> bytes:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    usable = (b.size // 8) * 8
    b = b[:usable].reshape(-1, 8)
    if bit_order == "lsb":
        b = b[:, ::-1]
    elif bit_order != "msb":
        raise ValueError("bit_order must be 'msb' or 'lsb'.")
    weights = 1 << np.arange(7, -1, -1)
    return bytes((b * weights).sum(axis=1).astype(np.uint8).tolist())


def byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values, minlength=256).astype(float)
    p = counts[counts > 0] / values.size
    return float(-np.sum(p * np.log2(p)))


def parse_candidate_payload(bits: np.ndarray, start: int, end: int | None = None) -> dict:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    end = b.size if end is None else min(end, b.size)
    payload = b[max(0, start):end]
    raw = bits_to_bytes(payload)
    return {
        "start_bit": int(start),
        "end_bit": int(end),
        "bit_length": int(payload.size),
        "byte_length": len(raw),
        "byte_entropy": byte_entropy(raw),
        "hex_preview": raw[:32].hex(),
    }
