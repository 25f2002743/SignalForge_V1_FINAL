from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt


def lowpass(samples: np.ndarray, sample_rate: float, cutoff_hz: float, order: int = 6) -> np.ndarray:
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist.")
    sos = butter(order, cutoff_hz / (sample_rate / 2), btype="lowpass", output="sos")
    return sosfilt(sos, np.asarray(samples))


def bandpass(
    samples: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 6,
) -> np.ndarray:
    if not 0 < low_hz < high_hz < sample_rate / 2:
        raise ValueError("Require 0 < low_hz < high_hz < Nyquist.")
    sos = butter(
        order,
        [low_hz / (sample_rate / 2), high_hz / (sample_rate / 2)],
        btype="bandpass",
        output="sos",
    )
    return sosfilt(sos, np.asarray(samples))
