from __future__ import annotations
import numpy as np


def _nearest_phase_indices(samples: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    phases = np.angle(samples)
    constellation = np.exp(1j * 2 * np.pi * np.arange(order) / order)
    distances = np.abs(samples[:, None] - constellation[None, :])
    idx = np.argmin(distances, axis=1)
    return idx, constellation[idx]


def _gray_encode(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)


def psk_demodulate(samples: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if order not in (2, 4, 8):
        raise ValueError("PSK order must be 2, 4, or 8.")
    x = np.asarray(samples, dtype=np.complex128)
    if x.size == 0:
        raise ValueError("No symbols.")

    idx, decided = _nearest_phase_indices(x, order)
    gray = _gray_encode(idx)
    bits_per_symbol = int(np.log2(order))
    bits = ((gray[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1).astype(np.uint8)
    hard = bits.reshape(-1)

    error = np.angle(x * np.conj(decided))
    evm = float(np.sqrt(np.mean(np.abs(x - decided) ** 2) / max(np.mean(np.abs(decided) ** 2), 1e-15)))
    phase_std = float(np.std(error))
    confidence = float(np.clip(1.0 - evm, 0.0, 1.0))

    # A simple signed phase-distance soft metric per bit.
    soft = np.empty((x.size, bits_per_symbol), dtype=float)
    phase = np.angle(x)
    boundaries = 2 * np.pi / order
    for b in range(bits_per_symbol):
        # Approximate LLR using distance to nearest symbols carrying each bit.
        vals = np.arange(order)
        bitvals = (_gray_encode(vals) >> (bits_per_symbol - 1 - b)) & 1
        d0 = np.min(np.abs(np.angle(np.exp(1j * (phase[:, None] - 2*np.pi*vals[None, :]/order))) + 0.0) + 10*(bitvals[None, :] != 0), axis=1)
        d1 = np.min(np.abs(np.angle(np.exp(1j * (phase[:, None] - 2*np.pi*vals[None, :]/order))) + 0.0) + 10*(bitvals[None, :] != 1), axis=1)
        soft[:, b] = d0 - d1

    return hard, soft.reshape(-1), {
        "evm_rms": evm,
        "phase_error_std_rad": phase_std,
        "mean_symbol_decision_distance": float(np.mean(np.abs(x - decided))),
        "samples_per_symbol": 1,
    }


def bpsk(samples: np.ndarray):
    return psk_demodulate(samples, 2)


def qpsk(samples: np.ndarray):
    return psk_demodulate(samples, 4)


def psk8(samples: np.ndarray):
    return psk_demodulate(samples, 8)
