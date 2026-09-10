from __future__ import annotations

from signalforge.core.signal_record import SignalRecord
from signalforge.modulation.features import modulation_features
from signalforge.modulation.classifiers import rule_based_advisor, rank_candidates


def advise(record: SignalRecord) -> dict:
    features = modulation_features(record.samples, record.sample_rate)
    probabilities = rule_based_advisor(features)
    ranked = rank_candidates(probabilities)

    result = {
        "features": features,
        "candidates": ranked,
        "method": "V1 rule-based evidence advisor",
        "interpretation": "candidate ranking only; final modulation requires executable hypothesis validation",
    }

    record.observations["modulation_features"] = features
    record.hypotheses.extend(
        [{"modulation": x["modulation"], "advisor_score": x["score"]} for x in ranked]
    )
    record.add_provenance("modulation_advisor", method="rule_based_v1")
    return result
