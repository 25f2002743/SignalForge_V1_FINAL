from __future__ import annotations
import numpy as np


def symbol_sample(
    samples: np.ndarray,
    samples_per_symbol: float,
    timing_offset: float = 0.0,
) -> np.ndarray:
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be > 0.")
    x = np.asarray(samples)
    positions = timing_offset + np.arange(0, x.size, samples_per_symbol)
    positions = positions[(positions >= 0) & (positions < x.size - 1)]

    i = np.floor(positions).astype(int)
    frac = positions - i

    if np.iscomplexobj(x):
        return x[i] * (1 - frac) + x[i + 1] * frac
    return x[i] * (1 - frac) + x[i + 1] * frac


def estimate_best_timing_offset(samples: np.ndarray, samples_per_symbol: float, search_points: int = 16) -> tuple[float, float]:
    """
    Searches fractional offsets using squared-envelope variance.
    It is a coarse timing metric, not a replacement for a tracking loop.
    """
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be >= 1.")
    best_offset = 0.0
    best_metric = -np.inf
    for offset in np.linspace(0, samples_per_symbol, max(2, search_points), endpoint=False):
        symbols = symbol_sample(samples, samples_per_symbol, offset)
        if symbols.size < 4:
            continue
        metric = float(np.var(np.abs(symbols) ** 2))
        if metric > best_metric:
            best_metric = metric
            best_offset = float(offset)
    return best_offset, best_metric
