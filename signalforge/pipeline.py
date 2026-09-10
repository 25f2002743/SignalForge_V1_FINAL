from __future__ import annotations
from signalforge.gui.app import analyze_file


class SignalForgePipeline:
    """Stable public V1 API around the end-to-end analysis pipeline."""

    def analyze(self, path: str, **kwargs) -> dict:
        return analyze_file(path, **kwargs)


def analyze(path: str, **kwargs) -> dict:
    return SignalForgePipeline().analyze(path, **kwargs)
