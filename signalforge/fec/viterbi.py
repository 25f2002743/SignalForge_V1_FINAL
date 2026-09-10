from __future__ import annotations
import numpy as np


def viterbi_decode_soft(llr: np.ndarray, generators=(0o7, 0o5), constraint_length: int = 3) -> np.ndarray:
    """
    Rate-1/2 hard/soft-compatible Viterbi implementation for short constraint lengths.
    Positive LLR means bit 0 is preferred; negative means bit 1.
    """
    llr = np.asarray(llr, dtype=float).reshape(-1)
    n_out = len(generators)
    if n_out != 2:
        raise ValueError("V1 Viterbi supports rate-1/2 only.")
    if llr.size % n_out:
        llr = llr[: llr.size - (llr.size % n_out)]

    states = 1 << (constraint_length - 1)
    inf = 1e30
    metric = np.full(states, inf)
    metric[0] = 0.0
    prev_state = np.full((llr.size // n_out, states), -1, dtype=int)
    prev_bit = np.full((llr.size // n_out, states), -1, dtype=np.int8)

    def parity(v):
        return int(v.bit_count() & 1)

    for t in range(llr.size // n_out):
        new_metric = np.full(states, inf)
        pair = llr[t*n_out:(t+1)*n_out]
        for s in range(states):
            if metric[s] >= inf:
                continue
            for bit in (0, 1):
                reg = (s << 1) | bit
                ns = reg & (states - 1)
                expected = np.array([parity(reg & g) for g in generators])
                # Branch metric: LLR>0 favors 0, LLR<0 favors 1.
                bm = float(np.sum(np.where(expected == 0, -pair, pair)))
                candidate = metric[s] + bm
                if candidate < new_metric[ns]:
                    new_metric[ns] = candidate
                    prev_state[t, ns] = s
                    prev_bit[t, ns] = bit
        metric = new_metric

    state = int(np.argmin(metric))
    out = np.zeros(llr.size // n_out, dtype=np.uint8)
    for t in range(out.size - 1, -1, -1):
        out[t] = prev_bit[t, state] if prev_bit[t, state] >= 0 else 0
        state = prev_state[t, state] if prev_state[t, state] >= 0 else 0
    return out
