from __future__ import annotations
import numpy as np


def modulation_features(samples: np.ndarray, sample_rate: float) -> dict:
    x = np.asarray(samples)
    a = np.abs(x)
    phase = np.unwrap(np.angle(x)) if np.iscomplexobj(x) else None

    # Robust, normalized features. These are advisors, not final decisions.
    amp_mean = float(np.mean(a))
    amp_std = float(np.std(a))
    amp_cv = amp_std / max(amp_mean, 1e-12)

    if np.iscomplexobj(x):
        p = np.angle(x)
        phase_diff = np.angle(np.exp(1j * np.diff(p)))
        phase_diff_std = float(np.std(phase_diff)) if phase_diff.size else 0.0

        # Higher-order moments provide a compact modulation-family clue.
        z = x / max(np.sqrt(np.mean(np.abs(x) ** 2)), 1e-12)
        m2 = np.mean(z ** 2)
        m4 = np.mean(z ** 4)
        m20 = np.mean(np.abs(z) ** 2)
        m40 = np.mean(np.abs(z) ** 4)

        inst_freq = np.diff(phase) * sample_rate / (2 * np.pi) if phase is not None and len(phase) > 1 else np.array([])
        fi_std = float(np.std(inst_freq)) if inst_freq.size else 0.0
        fi_p90 = float(np.percentile(np.abs(inst_freq), 90)) if inst_freq.size else 0.0
    else:
        phase_diff_std = 0.0
        m2 = m4 = m20 = m40 = 0.0
        fi_std = fi_p90 = 0.0

    return {
        "amplitude_mean": amp_mean,
        "amplitude_std": amp_std,
        "amplitude_cv": float(amp_cv),
        "phase_diff_std_rad": phase_diff_std,
        "instantaneous_frequency_std_hz": fi_std,
        "instantaneous_frequency_abs_p90_hz": fi_p90,
        "complex": bool(np.iscomplexobj(x)),
        "m2_real": float(np.real(m2)),
        "m2_imag": float(np.imag(m2)),
        "m4_real": float(np.real(m4)),
        "m4_imag": float(np.imag(m4)),
        "m20": float(np.real(m20)),
        "m40": float(np.real(m40)),
    }
