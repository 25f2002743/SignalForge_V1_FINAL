from __future__ import annotations
import numpy as np


def fsk_demodulate(
    samples: np.ndarray,
    sample_rate: float,
    samples_per_symbol: int,
    mark_hz: float | None = None,
    space_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.asarray(samples, dtype=np.complex128)
    if x.size < 2 or samples_per_symbol < 1:
        raise ValueError("Insufficient samples or invalid samples_per_symbol.")

    inst_freq = np.angle(x[1:] * np.conj(x[:-1])) * sample_rate / (2*np.pi)
    usable = (inst_freq.size // samples_per_symbol) * samples_per_symbol
    blocks = inst_freq[:usable].reshape(-1, samples_per_symbol)
    symbol_freq = np.median(blocks, axis=1)

    if mark_hz is None or space_hz is None:
        # Data-driven two-cluster initialization.
        lo, hi = np.percentile(symbol_freq, [25, 75])
        mark_hz, space_hz = float(hi), float(lo)

    d_mark = np.abs(symbol_freq - mark_hz)
    d_space = np.abs(symbol_freq - space_hz)
    hard = (d_mark < d_space).astype(np.uint8)
    # Positive soft value => mark/1 more likely.
    soft = d_space - d_mark

    separation = abs(mark_hz - space_hz)
    residual = np.minimum(d_mark, d_space)
    confidence = float(np.clip(1.0 - np.mean(residual) / max(separation, 1e-12), 0, 1))

    return hard, soft.astype(float), {
        "estimated_mark_hz": float(mark_hz),
        "estimated_space_hz": float(space_hz),
        "tone_separation_hz": float(separation),
        "mean_frequency_residual_hz": float(np.mean(residual)),
        "confidence": confidence,
    }
