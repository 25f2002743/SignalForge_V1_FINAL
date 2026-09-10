from __future__ import annotations
import numpy as np

from signalforge.fec.result import FECResult
from signalforge.fec.detector import rank_fec_candidates
from signalforge.fec.viterbi import viterbi_decode_soft
from signalforge.fec.reed_solomon import reed_solomon_available
from signalforge.fec.ldpc import ldpc_available
from signalforge.fec.concatenated import concatenated_available


def decode_candidates(bits: np.ndarray, soft_bits: np.ndarray | None = None, top_k: int = 4) -> list[FECResult]:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    candidates = rank_fec_candidates(b, soft_bits)
    results = []

    for c in candidates[:top_k]:
        if c["code"] == "CONVOLUTIONAL":
            if soft_bits is None:
                llr = np.where(b == 0, 1.0, -1.0)
            else:
                llr = np.asarray(soft_bits, dtype=float)
            corrected = viterbi_decode_soft(llr)
            results.append(FECResult(
                code="CONVOLUTIONAL",
                corrected_bits=corrected,
                success=True,
                confidence=float(c["score"]),
                metrics={"decoder": "V1 rate-1/2 Viterbi"},
                evidence=["Decoder executed as a hypothesis; success is not protocol proof."],
            ))
        elif c["code"] == "RS":
            results.append(FECResult("RS", b, False, 0.0, metrics=reed_solomon_available()))
        elif c["code"] == "LDPC":
            results.append(FECResult("LDPC", b, False, 0.0, metrics=ldpc_available()))
        else:
            results.append(FECResult("CONCATENATED", b, False, 0.0, metrics=concatenated_available()))

    return results
