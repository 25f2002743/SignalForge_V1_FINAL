from __future__ import annotations

from pathlib import Path
import wave
import numpy as np

from signalforge.core.signal_record import SignalRecord
from signalforge.ingestion.metadata import load_sidecar_metadata


_PCM_WIDTH_TO_DTYPE = {1: np.int8, 2: np.int16, 4: np.int32}


def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        vals = np.where(vals & 0x800000, vals - 0x1000000, vals)
        data = vals.astype(np.float32)
    else:
        dtype = _PCM_WIDTH_TO_DTYPE[sample_width]
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    return data.reshape(-1, channels)


def read_wav(path: str | Path, real_signal: bool = False) -> SignalRecord:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()
        raw = wf.readframes(nframes)

    if comptype != "NONE":
        raise ValueError(f"Compressed WAV is not supported: {comptype}")
    if channels < 1:
        raise ValueError("WAV has no channels.")
    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes.")

    matrix = _decode_pcm(raw, sample_width, channels)

    metadata = load_sidecar_metadata(path)
    center_frequency = metadata.get("center_frequency")
    timestamp = metadata.get("timestamp")

    if channels == 1 or real_signal:
        samples = matrix[:, 0].astype(np.float32)
        channel_info = {"mode": "real", "channels": channels}
    elif channels == 2:
        # V1 convention: stereo WAV can be interpreted as I/Q.
        samples = (matrix[:, 0] + 1j * matrix[:, 1]).astype(np.complex64)
        channel_info = {"mode": "iq", "channels": 2, "mapping": "left=I,right=Q"}
    else:
        raise ValueError(
            "Multi-channel WAV with >2 channels requires explicit channel configuration."
        )

    record = SignalRecord(
        samples=samples,
        sample_rate=float(sample_rate),
        center_frequency=center_frequency,
        timestamp=timestamp,
        source_format="wav",
        dtype=str(samples.dtype),
        channel_info=channel_info,
        metadata={
            "num_channels": channels,
            "sample_width_bytes": sample_width,
            "compression": comptype,
            **metadata,
        },
    )
    record.add_provenance("wav_decode", path=str(path))
    return record
