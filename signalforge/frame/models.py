from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameCandidate:
    start_bit: int
    end_bit: int
    header_start: int | None = None
    header_end: int | None = None
    payload_start: int | None = None
    payload_end: int | None = None
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self):
        return self.end_bit - self.start_bit

    def as_dict(self):
        return {
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "length": self.length,
            "header": [self.header_start, self.header_end],
            "payload": [self.payload_start, self.payload_end],
            "score": self.score,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }
