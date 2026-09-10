from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    id: str
    modulation: str
    symbol_rate_hz: float | None = None
    samples_per_symbol: float | None = None
    carrier_offset_hz: float | None = None
    timing_offset: float | None = None
    pulse_shape: str | None = None
    fec: str | None = None
    interleaver: str | None = None
    advisor_score: float = 0.0
    evidence_score: float = 0.0
    status: str = "GENERATED"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modulation": self.modulation,
            "symbol_rate_hz": self.symbol_rate_hz,
            "samples_per_symbol": self.samples_per_symbol,
            "carrier_offset_hz": self.carrier_offset_hz,
            "timing_offset": self.timing_offset,
            "pulse_shape": self.pulse_shape,
            "fec": self.fec,
            "interleaver": self.interleaver,
            "advisor_score": self.advisor_score,
            "evidence_score": self.evidence_score,
            "status": self.status,
            "evidence": self.evidence,
            "contradictions": self.contradictions,
        }
