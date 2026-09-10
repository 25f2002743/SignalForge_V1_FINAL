from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from signalforge.core.signal_record import SignalRecord
from signalforge.ingestion.metadata import load_sidecar_metadata


DTYPES = {
    "int16": np.int16,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


def _read_values(path: Path, dtype_name: str) -> np.ndarray:
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported IQ dtype: {dtype_name}. Choose {list(DTYPES)}.")
    data = np.fromfile(path, dtype=DTYPES[dtype_name])
    if data.size == 0:
        raise ValueError("IQ file contains no samples.")
    return data


def _to_complex(values: np.ndarray, layout: str) -> np.ndarray:
    layout = layout.lower()
    if values.size % 2:
        raise ValueError("IQ file must contain an even number of scalar values.")

    if layout == "iqiq":
        i = values[0::2]
        q = values[1::2]
    elif layout == "iiqq":
        half = values.size // 2
        i = values[:half]
        q = values[half:]
    else:
        raise ValueError("iq_layout must be 'iqiq' or 'iiqq'.")

    return (i.astype(np.float64) + 1j * q.astype(np.float64)).astype(np.complex128)


def read_iq(
    path: str | Path,
    sample_rate: float | None = None,
    dtype: str | None = None,
    iq_layout: str | None = None,
    center_frequency: float | None = None,
    timestamp: str | None = None,
) -> SignalRecord:
    path = Path(path)
    metadata: dict[str, Any] = load_sidecar_metadata(path)

    sample_rate = sample_rate or metadata.get("sample_rate")
    dtype = dtype or metadata.get("dtype")
    iq_layout = iq_layout or metadata.get("iq_layout", "iqiq")
    center_frequency = (
        center_frequency if center_frequency is not None
        else metadata.get("center_frequency")
    )
    timestamp = timestamp or metadata.get("timestamp")

    if sample_rate is None:
        raise ValueError("sample_rate is required for raw IQ unless present in sidecar metadata.")
    if dtype is None:
        raise ValueError("dtype is required for raw IQ unless present in sidecar metadata.")

    values = _read_values(path, dtype)
    samples = _to_complex(values, iq_layout)

    record = SignalRecord(
        samples=samples,
        sample_rate=float(sample_rate),
        center_frequency=center_frequency,
        timestamp=timestamp,
        source_format="iq",
        dtype=str(samples.dtype),
        channel_info={"mode": "iq", "layout": iq_layout, "scalar_dtype": dtype},
        metadata={**metadata, "scalar_dtype": dtype, "iq_layout": iq_layout},
    )
    record.add_provenance("iq_decode", path=str(path), dtype=dtype, iq_layout=iq_layout)
    return record
