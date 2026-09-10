from __future__ import annotations
import numpy as np


def validate_samples(samples: np.ndarray) -> None:
    if samples.ndim != 1:
        raise ValueError("Samples must be 1-D.")
    if samples.size == 0:
        raise ValueError("Capture contains no samples.")
    if not np.all(np.isfinite(samples.real)):
        raise ValueError("Capture contains NaN/Inf values in real component.")
    if np.iscomplexobj(samples) and not np.all(np.isfinite(samples.imag)):
        raise ValueError("Capture contains NaN/Inf values in imaginary component.")


def amplitude_normalize(samples: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Return a derived normalized copy; input is not modified."""
    validate_samples(samples)
    x = np.asarray(samples).copy()
    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0:
        return x
    return x * (peak / max_abs)
