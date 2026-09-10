from __future__ import annotations
import numpy as np


def time_domain(x: np.ndarray, sample_rate: float) -> dict:
    x = np.asarray(x)
    n = min(x.size, 200_000)
    t = np.arange(n) / float(sample_rate)
    if np.iscomplexobj(x):
        return {"time_s": t, "i": x[:n].real, "q": x[:n].imag, "magnitude": np.abs(x[:n])}
    return {"time_s": t, "amplitude": x[:n].real}


def spectrum(x: np.ndarray, sample_rate: float, max_points: int = 8192) -> dict:
    x = np.asarray(x)
    n = min(x.size, max_points)
    if n < 2:
        return {"frequency_hz": np.array([]), "power_db": np.array([])}
    y = x[:n] * np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(y))
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))
    power = 20 * np.log10(np.maximum(np.abs(spec) / n, 1e-12))
    return {"frequency_hz": freq, "power_db": power}


def waterfall(x: np.ndarray, sample_rate: float, fft_size: int = 1024, rows: int = 256) -> dict:
    x = np.asarray(x)
    if x.size < fft_size:
        return {"frequency_hz": np.array([]), "time_s": np.array([]), "power_db": np.empty((0, 0))}
    hop = fft_size // 2
    count = min(rows, 1 + (x.size - fft_size) // hop)
    window = np.hanning(fft_size)
    matrix = []
    for k in range(count):
        start = k * hop
        seg = x[start:start + fft_size] * window
        s = np.fft.fftshift(np.fft.fft(seg))
        matrix.append(20 * np.log10(np.maximum(np.abs(s) / fft_size, 1e-12)))
    freq = np.fft.fftshift(np.fft.fftfreq(fft_size, 1.0 / sample_rate))
    times = (np.arange(count) * hop + fft_size / 2) / sample_rate
    return {"frequency_hz": freq, "time_s": times, "power_db": np.asarray(matrix)}


def constellation(symbols: np.ndarray, max_points: int = 10_000) -> dict:
    s = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if s.size > max_points:
        idx = np.linspace(0, s.size - 1, max_points).astype(int)
        s = s[idx]
    return {"i": s.real, "q": s.imag, "count": int(s.size)}
