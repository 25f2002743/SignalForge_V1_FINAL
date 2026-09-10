from __future__ import annotations

from pathlib import Path
from signalforge.core.signal_record import SignalRecord
from signalforge.ingestion.iq_parser import read_iq
from signalforge.ingestion.wav_parser import read_wav


def load_signal(path: str | Path, **kwargs) -> SignalRecord:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix == ".wav":
        return read_wav(p, real_signal=kwargs.get("real_signal", False))
    if suffix in {".iq", ".raw", ".bin"}:
        return read_iq(
            p,
            sample_rate=kwargs.get("sample_rate"),
            dtype=kwargs.get("dtype"),
            iq_layout=kwargs.get("iq_layout"),
            center_frequency=kwargs.get("center_frequency"),
            timestamp=kwargs.get("timestamp"),
        )
    raise ValueError(f"Unsupported capture extension: {suffix}")
