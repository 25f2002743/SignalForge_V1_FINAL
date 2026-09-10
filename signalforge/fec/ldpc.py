from __future__ import annotations
import numpy as np


def ldpc_available() -> dict:
    return {
        "implemented": False,
        "code": "LDPC",
        "status": "PLUGIN_REQUIRED",
        "reason": "V1 core exposes an LDPC adapter boundary; validated protocol-specific parity-check matrices/decoders are required.",
    }


def decode_ldpc(llr: np.ndarray, *args, **kwargs):
    raise NotImplementedError(ldpc_available()["reason"])
