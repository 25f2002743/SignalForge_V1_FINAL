from __future__ import annotations
import numpy as np


def constellation(
    samples: np.ndarray,
    max_points: int = 10000,
    normalize: bool = True,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(samples)
    if not np.iscomplexobj(x):
        raise ValueError("Constellation requires complex I/Q samples.")
    if x.size == 0:
        raise ValueError("Signal contains no samples.")

    step = max(1, int(np.ceil(x.size / max_points)))
    points = x[::step].astype(np.complex128, copy=True)

    if normalize:
        rms = np.sqrt(np.mean(np.abs(points) ** 2))
        if rms > 0:
            points /= rms

    return {
        "i": points.real,
        "q": points.imag,
        "magnitude": np.abs(points),
        "phase": np.angle(points),
        "num_points": float(points.size),
    }
