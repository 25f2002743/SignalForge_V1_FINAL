from __future__ import annotations
import numpy as np


def correlate_bits(
    bits: np.ndarray,
    reference: np.ndarray,
    max_lag: int | None = None,
) -> dict:
    a = np.asarray(bits, dtype=np.uint8).reshape(-1)
    b = np.asarray(reference, dtype=np.uint8).reshape(-1)
    if a.size == 0 or b.size == 0:
        return {"best_lag": None, "score": 0.0, "status": "UNOBSERVABLE"}

    if max_lag is None:
        max_lag = min(10000, max(a.size, b.size))

    # Map bits to +/-1 and use normalized cross-correlation.
    aa = 2.0 * a.astype(float) - 1.0
    bb = 2.0 * b.astype(float) - 1.0
    corr = np.correlate(aa, bb, mode="full")
    lags = np.arange(-bb.size + 1, aa.size)
    mask = np.abs(lags) <= max_lag
    corr = corr[mask]
    lags = lags[mask]

    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    scores = corr / max(denom, 1e-12)
    i = int(np.argmax(np.abs(scores)))
    return {
        "best_lag": int(lags[i]),
        "score": float(scores[i]),
        "absolute_score": float(abs(scores[i])),
        "status": "SUCCESS",
    }
