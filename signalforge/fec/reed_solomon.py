from __future__ import annotations
import numpy as np


def reed_solomon_available() -> dict:
    return {
        "implemented": False,
        "code": "RS",
        "status": "PLUGIN_REQUIRED",
        "reason": "V1 core keeps RS field arithmetic isolated; a validated RS implementation/plugin can be attached here.",
    }


def decode_rs(bits: np.ndarray, *args, **kwargs):
    raise NotImplementedError(reed_solomon_available()["reason"])
