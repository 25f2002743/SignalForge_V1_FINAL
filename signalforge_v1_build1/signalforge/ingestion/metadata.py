from __future__ import annotations
from pathlib import Path
import json
from typing import Any


def load_sidecar_metadata(path: str | Path) -> dict[str, Any]:
    """Load optional <capture>.json metadata without altering the capture."""
    p = Path(path)
    candidates = [
        p.with_suffix(p.suffix + ".json"),
        p.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Metadata file must contain a JSON object: {candidate}")
            return data
    return {}
