from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ParameterEstimate:
    name: str
    value: Any
    confidence: Optional[float]
    evidence: list[str]
    status: str  # KNOWN / DERIVED / AMBIGUOUS / UNOBSERVABLE

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "status": self.status,
        }
