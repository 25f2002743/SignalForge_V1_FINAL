from __future__ import annotations
import numpy as np


def waterfall(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 1024,
    hop: int | None = None,
) -> dict[str, np.ndarray]:
    x = np.asarray(samples)
    if x.size < fft_size:
        raise ValueError("Signal is shorter than fft_size.")
    if hop is None:
        hop = fft_size // 2
    if hop <= 0:
        raise ValueError("hop must be > 0")

    window = np.hanning(fft_size)
    starts = np.arange(0, x.size - fft_size + 1, hop)
    frames = []

    for start in starts:
        frame = x[start:start + fft_size] * window
        spec = np.fft.fftshift(np.fft.fft(frame))
        frames.append(20.0 * np.log10(np.maximum(np.abs(spec) / np.sum(window), 1e-15)))

    matrix = np.asarray(frames)
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate)
    )
    times = (starts + fft_size / 2) / sample_rate

    return {
        "time_s": times,
        "frequency_hz": frequencies,
        "power_db": matrix,
    }
