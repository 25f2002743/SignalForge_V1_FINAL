from __future__ import annotations


def concatenated_available() -> dict:
    return {
        "implemented": False,
        "code": "CONCATENATED",
        "status": "COMPOSITION_REQUIRED",
        "reason": "Compose validated inner/outer decoders once protocol ordering and parameters are known.",
    }
