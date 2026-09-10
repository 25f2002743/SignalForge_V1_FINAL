from __future__ import annotations
import numpy as np


def gardner_error(samples: np.ndarray, sps: float) -> float:
    """
    One-shot Gardner timing error estimate for an already roughly synchronized
    complex sequence. Used as evidence for a timing hypothesis.
    """
    if sps < 2:
        raise ValueError("Gardner detector requires at least 2 samples/symbol.")
    x = np.asarray(samples)
    step = max(1, int(round(sps / 2)))
    if x.size < 3 * step:
        return 0.0

    idx = np.arange(step, x.size - step, step)
    mid = x[idx]
    early = x[idx - step]
    late = x[idx + step]
    err = np.real((early - late) * np.conj(mid))
    return float(np.mean(err))


def resample_to_sps(samples: np.ndarray, sample_rate: float, symbol_rate: float, sps: int = 4) -> np.ndarray:
    """Simple rational resampling to a practical integer samples/symbol target."""
    if symbol_rate <= 0:
        raise ValueError("symbol_rate must be > 0.")
    target_rate = symbol_rate * sps
    ratio = target_rate / sample_rate
    n_out = max(1, int(round(len(samples) * ratio)))
    old_t = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    new_t = np.linspace(0.0, 1.0, n_out, endpoint=False)
    x = np.asarray(samples)
    if np.iscomplexobj(x):
        re = np.interp(new_t, old_t, x.real)
        im = np.interp(new_t, old_t, x.imag)
        return re + 1j * im
    return np.interp(new_t, old_t, x)
