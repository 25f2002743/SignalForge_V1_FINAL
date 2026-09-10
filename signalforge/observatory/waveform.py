from __future__ import annotations
import numpy as np


def time_axis(num_samples: int, sample_rate: float) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    return np.arange(num_samples, dtype=float) / sample_rate


def waveform_components(samples: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(samples)
    return {
        "real": np.real(x),
        "imag": np.imag(x),
        "magnitude": np.abs(x),
        "phase": np.angle(x),
    }
