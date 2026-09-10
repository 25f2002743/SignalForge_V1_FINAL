from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class DemodResult:
    modulation: str
    hard_bits: np.ndarray
    soft_bits: np.ndarray | None = None
    symbols: np.ndarray | None = None
    symbol_indices: np.ndarray | None = None
    confidence: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "modulation": self.modulation,
            "num_bits": int(self.hard_bits.size),
            "confidence": float(self.confidence),
            "metrics": self.metrics,
            "status": self.status,
            "evidence": self.evidence,
        }
