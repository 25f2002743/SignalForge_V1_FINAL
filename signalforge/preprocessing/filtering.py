from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt


def butter_filter(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_hz: float | tuple[float, float],
    kind: str = "lowpass",
    order: int = 5,
) -> np.ndarray:
    nyq = sample_rate / 2.0
    if kind in {"lowpass", "highpass"}:
        if not 0 < float(cutoff_hz) < nyq:
            raise ValueError("Cutoff must be between 0 and Nyquist.")
        wn = float(cutoff_hz) / nyq
    elif kind == "bandpass":
        low, high = cutoff_hz
        if not 0 < low < high < nyq:
            raise ValueError("Bandpass cutoffs must satisfy 0 < low < high < Nyquist.")
        wn = [low / nyq, high / nyq]
    else:
        raise ValueError("kind must be lowpass, highpass, or bandpass")

    sos = butter(order, wn, btype=kind, output="sos")
    return sosfiltfilt(sos, np.asarray(samples))
