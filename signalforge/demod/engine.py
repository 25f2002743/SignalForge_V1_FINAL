from __future__ import annotations

import numpy as np

from signalforge.core.signal_record import SignalRecord
from signalforge.demod.result import DemodResult
from signalforge.demod.psk import bpsk, qpsk, psk8
from signalforge.demod.qam import qam16_demodulate
from signalforge.demod.fsk import fsk_demodulate


def demodulate(
    record: SignalRecord,
    modulation: str,
    symbols: np.ndarray,
    samples_per_symbol: int = 4,
) -> DemodResult:
    modulation = modulation.upper()

    if modulation == "BPSK":
        hard, soft, metrics = bpsk(symbols)
    elif modulation == "QPSK":
        hard, soft, metrics = qpsk(symbols)
    elif modulation == "8PSK":
        hard, soft, metrics = psk8(symbols)
    elif modulation == "16QAM":
        hard, soft, metrics = qam16_demodulate(symbols)
    elif modulation == "FSK":
        hard, soft, metrics = fsk_demodulate(
            record.samples,
            record.sample_rate,
            samples_per_symbol=max(1, int(samples_per_symbol)),
        )
    else:
        return DemodResult(
            modulation=modulation,
            hard_bits=np.array([], dtype=np.uint8),
            status="UNSUPPORTED",
            evidence=["V1 demodulator does not implement this modulation."],
        )

    confidence = float(metrics.get("confidence", np.clip(
        1.0 - float(metrics.get("evm_rms", 1.0)), 0, 1
    )))

    result = DemodResult(
        modulation=modulation,
        hard_bits=hard,
        soft_bits=soft,
        symbols=np.asarray(symbols),
        confidence=confidence,
        metrics=metrics,
        evidence=[
            "Symbol decisions produced by modulation-specific detector.",
            "Soft metrics are decoder-facing reliability evidence, not calibrated probabilities.",
        ],
    )
    return result
