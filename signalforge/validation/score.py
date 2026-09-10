from __future__ import annotations
import numpy as np


def crc16_ccitt(bits: np.ndarray, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    usable = (b.size // 8) * 8
    data = np.packbits(b[:usable], bitorder="big").tobytes()
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def basic_bit_quality(bits: np.ndarray) -> dict:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size == 0:
        return {"length": 0, "one_ratio": None, "transition_ratio": None}
    return {
        "length": int(b.size),
        "one_ratio": float(np.mean(b)),
        "transition_ratio": float(np.mean(b[1:] != b[:-1])) if b.size > 1 else 0.0,
        "constant": bool(np.all(b == b[0])),
    }


def candidate_quality(
    demod_confidence: float,
    deinterleave_score: float,
    fec_confidence: float,
    frame_score: float,
) -> float:
    values = np.clip(
        np.array([demod_confidence, deinterleave_score, fec_confidence, frame_score], dtype=float),
        0, 1,
    )
    return float(np.average(values, weights=[0.30, 0.15, 0.30, 0.25]))
