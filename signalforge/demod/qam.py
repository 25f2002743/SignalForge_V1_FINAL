from __future__ import annotations
import numpy as np


def _gray2(v: int) -> int:
    return v ^ (v >> 1)


def qam16_demodulate(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.asarray(samples, dtype=np.complex128)
    if x.size == 0:
        raise ValueError("No symbols.")

    # 16-QAM square constellation, Gray labels on each I/Q axis.
    levels = np.array([-3, -1, 1, 3], dtype=float)
    scale = np.sqrt(10.0)
    y = x * scale

    i_idx = np.argmin(np.abs(y.real[:, None] - levels[None, :]), axis=1)
    q_idx = np.argmin(np.abs(y.imag[:, None] - levels[None, :]), axis=1)

    i_gray = np.array([_gray2(int(v)) for v in i_idx], dtype=int)
    q_gray = np.array([_gray2(int(v)) for v in q_idx], dtype=int)

    i_bits = ((i_gray[:, None] >> np.array([1, 0])) & 1).astype(np.uint8)
    q_bits = ((q_gray[:, None] >> np.array([1, 0])) & 1).astype(np.uint8)

    bits = np.concatenate([i_bits, q_bits], axis=1)
    hard = bits.reshape(-1)

    decided = (levels[i_idx] + 1j * levels[q_idx]) / scale
    evm = float(np.sqrt(np.mean(np.abs(x - decided) ** 2) / max(np.mean(np.abs(decided) ** 2), 1e-15)))

    # Approximate bit reliability from distance to decision boundaries.
    i_rel = np.abs(y.real - levels[i_idx])
    q_rel = np.abs(y.imag - levels[q_idx])
    rel = np.column_stack([i_rel, i_rel, q_rel, q_rel]).reshape(-1)
    soft = (1.0 - np.clip(rel / 4.0, 0, 1)) * (1.0 - 2.0 * hard)

    return hard, soft.astype(float), {
        "evm_rms": evm,
        "mean_decision_distance": float(np.mean(np.abs(x - decided))),
        "normalization": "sqrt(10)",
    }
