from __future__ import annotations
import numpy as np

from signalforge.core.signal_record import SignalRecord
from signalforge.dsp.frequency import estimate_coarse_cfo, mix_frequency
from signalforge.dsp.filters import lowpass
from signalforge.dsp.matched_filter import matched_filter
from signalforge.dsp.synchronizer import estimate_best_timing_offset, symbol_sample


def run_dsp_chain(
    record: SignalRecord,
    symbol_rate_hz: float,
    modulation: str,
    filter_cutoff_hz: float | None = None,
    samples_per_symbol: int = 4,
) -> dict:
    if not record.is_complex:
        raise ValueError("V1 DSP chain expects complex baseband I/Q.")
    if symbol_rate_hz <= 0:
        raise ValueError("symbol_rate_hz must be > 0.")

    x = record.samples.astype(np.complex128)
    cfo = estimate_coarse_cfo(x, record.sample_rate)
    x_cfo = mix_frequency(x, record.sample_rate, cfo)

    if filter_cutoff_hz is None:
        filter_cutoff_hz = min(
            record.sample_rate * 0.45,
            max(symbol_rate_hz * 1.5, symbol_rate_hz),
        )
    x_f = lowpass(x_cfo, record.sample_rate, filter_cutoff_hz)

    sps_float = record.sample_rate / symbol_rate_hz
    # Matched filter requires integer SPS; resampling is intentionally not hidden here.
    sps = max(2, int(round(sps_float)))
    x_mf = matched_filter(x_f, sps)

    offset, timing_metric = estimate_best_timing_offset(x_mf, sps)
    symbols = symbol_sample(x_mf, sps, offset)

    phase_diagnostics = None
    if modulation == "BPSK":
        symbols, phase_diagnostics = _carrier_correct_bpsk(symbols)

    result = {
        "input_samples": int(x.size),
        "estimated_cfo_hz": float(cfo),
        "filter_cutoff_hz": float(filter_cutoff_hz),
        "samples_per_symbol_estimate": float(sps_float),
        "integer_samples_per_symbol": int(sps),
        "timing_offset_samples": float(offset),
        "timing_metric": float(timing_metric),
        "symbols": symbols,
        "carrier_diagnostics": phase_diagnostics,
        "stages": ["coarse_cfo", "lowpass", "rrc_matched_filter", "coarse_timing", "carrier_correction"],
    }
    return result


def _carrier_correct_bpsk(symbols: np.ndarray) -> tuple[np.ndarray, dict]:
    from signalforge.dsp.carrier import costas_bpsk
    return costas_bpsk(symbols)
