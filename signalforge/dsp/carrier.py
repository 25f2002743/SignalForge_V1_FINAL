from __future__ import annotations
import numpy as np


def costas_bpsk(
    samples: np.ndarray,
    loop_gain: float = 0.02,
    damping: float = 0.707,
) -> tuple[np.ndarray, dict]:
    """
    Lightweight decision-directed Costas-style phase loop for BPSK.
    Returns corrected samples and loop diagnostics.
    """
    x = np.asarray(samples, dtype=np.complex128)
    phase = 0.0
    freq = 0.0
    out = np.empty_like(x)
    errors = np.empty(x.size, dtype=float)

    alpha = loop_gain
    beta = (loop_gain ** 2) / max(4 * damping * damping, 1e-12)

    for k, sample in enumerate(x):
        corrected = sample * np.exp(-1j * phase)
        out[k] = corrected
        decision = 1.0 if corrected.real >= 0 else -1.0
        err = np.imag(corrected) * decision
        errors[k] = err
        freq += beta * err
        phase += freq + alpha * err
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

    return out, {
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "final_phase_rad": float(phase),
        "final_frequency_state": float(freq),
    }
