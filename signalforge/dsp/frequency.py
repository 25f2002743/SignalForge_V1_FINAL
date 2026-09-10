from __future__ import annotations
import numpy as np


def mix_frequency(samples: np.ndarray, sample_rate: float, offset_hz: float) -> np.ndarray:
    """Complex frequency translation. Positive offset shifts spectrum downward."""
    x = np.asarray(samples)
    n = np.arange(x.size)
    return x * np.exp(-1j * 2 * np.pi * offset_hz * n / sample_rate)


def estimate_coarse_cfo(samples: np.ndarray, sample_rate: float) -> float:
    """Coarse frequency estimate from mean phase increment for complex input."""
    x = np.asarray(samples)
    if not np.iscomplexobj(x) or x.size < 2:
        return 0.0
    phase_inc = np.angle(x[1:] * np.conj(x[:-1]))
    return float(np.median(phase_inc) * sample_rate / (2 * np.pi))
