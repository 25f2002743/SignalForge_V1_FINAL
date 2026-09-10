from __future__ import annotations
import numpy as np
from scipy.signal import welch


def welch_psd(
    samples: np.ndarray,
    sample_rate: float,
    nperseg: int | None = None,
) -> dict[str, np.ndarray]:
    x = np.asarray(samples)
    if x.size == 0 or sample_rate <= 0:
        raise ValueError("Invalid signal or sample rate.")

    if nperseg is None:
        nperseg = min(4096, x.size)
    nperseg = max(8, min(nperseg, x.size))

    f, pxx = welch(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=not np.iscomplexobj(x),
        scaling="density",
    )
    if np.iscomplexobj(x):
        order = np.argsort(f)
        f, pxx = f[order], pxx[order]

    pxx_db = 10.0 * np.log10(np.maximum(np.abs(pxx), 1e-20))
    return {"frequency_hz": f, "psd": pxx, "psd_db": pxx_db}
