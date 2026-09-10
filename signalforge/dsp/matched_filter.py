from __future__ import annotations
import numpy as np


def rrc_taps(
    samples_per_symbol: int,
    rolloff: float = 0.35,
    span_symbols: int = 8,
) -> np.ndarray:
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be >= 2")
    if not 0 < rolloff <= 1:
        raise ValueError("rolloff must be in (0, 1].")

    n = span_symbols * samples_per_symbol
    t = np.arange(-n / 2, n / 2 + 1, dtype=float) / samples_per_symbol
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - rolloff + 4 * rolloff / np.pi
        elif abs(abs(4 * rolloff * ti) - 1.0) < 1e-12:
            taps[i] = (
                rolloff / np.sqrt(2)
                * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
            )
        else:
            num = (
                np.sin(np.pi * ti * (1 - rolloff))
                + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            )
            den = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            taps[i] = num / den

    norm = np.sqrt(np.sum(taps ** 2))
    return taps / max(norm, 1e-12)


def matched_filter(samples: np.ndarray, samples_per_symbol: int, rolloff: float = 0.35) -> np.ndarray:
    taps = rrc_taps(samples_per_symbol, rolloff=rolloff)
    return np.convolve(np.asarray(samples), taps, mode="same")
