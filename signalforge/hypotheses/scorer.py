from __future__ import annotations
import numpy as np

from signalforge.hypotheses.model import Hypothesis


def score_hypothesis(h: Hypothesis, advisor_weight: float = 0.35) -> float:
    """
    Initial V1 ranking score. Later DSP/FEC/reconstruction evidence is additive.
    """
    base = float(np.clip(h.advisor_score, 0, 1))
    evidence = float(np.clip(h.evidence_score, 0, 1))
    score = advisor_weight * base + (1.0 - advisor_weight) * evidence
    if h.contradictions:
        score -= min(0.5, 0.1 * len(h.contradictions))
    return float(np.clip(score, 0, 1))


def rank_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    for h in hypotheses:
        h.evidence_score = max(h.evidence_score, h.advisor_score)
        h.evidence_score = score_hypothesis(h, advisor_weight=0.5)
    return sorted(hypotheses, key=lambda x: x.evidence_score, reverse=True)
