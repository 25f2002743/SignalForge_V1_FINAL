from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FECResult:
    code: str
    corrected_bits: np.ndarray
    success: bool
    confidence: float
    corrected_count: int = 0
    metrics: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "code": self.code,
            "bit_count": int(self.corrected_bits.size),
            "success": bool(self.success),
            "confidence": float(self.confidence),
            "corrected_count": int(self.corrected_count),
            "metrics": self.metrics,
            "evidence": self.evidence,
        }
