from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks


def fft_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    window: str = "hann",
) -> dict[str, np.ndarray]:
    x = np.asarray(samples)
    if x.size == 0 or sample_rate <= 0:
        raise ValueError("Invalid signal or sample rate.")

    n = x.size
    if window == "hann":
        w = np.hanning(n)
    elif window == "none":
        w = np.ones(n)
    else:
        raise ValueError("window must be 'hann' or 'none'")

    X = np.fft.fftshift(np.fft.fft(x * w))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    magnitude = np.abs(X) / max(np.sum(w), 1.0)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-15))

    return {
        "frequency_hz": f,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
    }


def spectral_peaks(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    prominence_db: float = 3.0,
) -> dict[str, np.ndarray]:
    idx, props = find_peaks(magnitude_db, prominence=prominence_db)
    order = np.argsort(magnitude_db[idx])[::-1]
    idx = idx[order]
    return {
        "indices": idx,
        "frequency_hz": frequency_hz[idx],
        "power_db": magnitude_db[idx],
        "prominence_db": props.get("prominences", np.array([]))[order],
    }
