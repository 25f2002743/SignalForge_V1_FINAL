from __future__ import annotations
from itertools import product

from signalforge.core.signal_record import SignalRecord
from signalforge.hypotheses.model import Hypothesis


def _symbol_rates(record: SignalRecord) -> list[float | None]:
    candidates = record.parameters.get("symbol_rate_candidates", {}).get("value", [])
    rates = [float(x["symbol_rate_hz"]) for x in candidates if x.get("symbol_rate_hz", 0) > 0]
    return rates[:3] or [None]


def generate(record: SignalRecord, max_hypotheses: int = 12) -> list[Hypothesis]:
    advisor = record.hypotheses
    rates = _symbol_rates(record)
    ranked = sorted(advisor, key=lambda x: x.get("advisor_score", 0), reverse=True)

    hypotheses = []
    counter = 1
    for candidate, rate in product(ranked[:5], rates):
        hypotheses.append(
            Hypothesis(
                id=f"H{counter}",
                modulation=candidate["modulation"],
                symbol_rate_hz=rate,
                advisor_score=float(candidate.get("advisor_score", 0)),
            )
        )
        counter += 1
        if len(hypotheses) >= max_hypotheses:
            break

    record.hypotheses = [h.as_dict() for h in hypotheses]
    record.add_provenance("hypothesis_generation", count=len(hypotheses))
    return hypotheses
