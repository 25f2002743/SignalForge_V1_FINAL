from __future__ import annotations

import argparse
import json
from signalforge.ingestion.loader import load_signal


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SignalForge V1 — Build 1 ingestion smoke test")
    p.add_argument("file", help="Path to .wav, .iq, .raw or .bin capture")
    p.add_argument("--sample-rate", type=float, help="Sample rate for raw IQ")
    p.add_argument("--dtype", choices=["int16", "int32", "float32", "float64"])
    p.add_argument("--iq-layout", choices=["iqiq", "iiqq"], default=None)
    p.add_argument("--center-frequency", type=float)
    p.add_argument("--real-signal", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    record = load_signal(
        args.file,
        sample_rate=args.sample_rate,
        dtype=args.dtype,
        iq_layout=args.iq_layout,
        center_frequency=args.center_frequency,
        real_signal=args.real_signal,
    )
    print(json.dumps(record.summary(), indent=2, default=str))


if __name__ == "__main__":
    main()
