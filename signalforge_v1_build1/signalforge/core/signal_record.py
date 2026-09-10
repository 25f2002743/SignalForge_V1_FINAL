from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
import numpy as np


@dataclass
class SignalRecord:
    """Canonical signal contract consumed by all downstream V1 modules."""

    samples: np.ndarray
    sample_rate: float
    center_frequency: Optional[float] = None
    timestamp: Optional[str] = None
    source_format: str = "unknown"
    dtype: str = ""
    channel_info: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    provenance: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    decoding_results: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples)
        if arr.ndim != 1:
            raise ValueError("SignalRecord.samples must be a 1-D array.")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("SignalRecord.samples must be numeric.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0.")
        self.samples = arr
        self.dtype = self.dtype or str(arr.dtype)
        self.duration = len(arr) / self.sample_rate

    @property
    def num_samples(self) -> int:
        return int(self.samples.size)

    @property
    def is_complex(self) -> bool:
        return np.iscomplexobj(self.samples)

    def add_provenance(self, operation: str, **details: Any) -> None:
        self.provenance.append({"operation": operation, **details})

    def summary(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "sample_rate_hz": self.sample_rate,
            "duration_s": self.duration,
            "center_frequency_hz": self.center_frequency,
            "source_format": self.source_format,
            "dtype": self.dtype,
            "complex": self.is_complex,
            "channel_info": self.channel_info,
            "metadata": self.metadata,
        }
