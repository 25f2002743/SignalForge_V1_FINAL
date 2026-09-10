from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DeinterleaveResult:
    method: str
    bits: np.ndarray
    score: float
    status: str = "SUCCESS"
    metrics: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "method": self.method,
            "bit_count": int(self.bits.size),
            "score": float(self.score),
            "status": self.status,
            "metrics": self.metrics,
            "evidence": self.evidence,
        }
