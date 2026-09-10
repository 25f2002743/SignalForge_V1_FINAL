# SignalForge V1 — Build 1
Foundation + Ingestion + Canonical SignalRecord

## Scope
- WAV ingestion: mono/stereo PCM and float where supported by Python's wave module
- IQ/raw ingestion: int16/int32/float32/float64
- IQIQ and IIQQ layouts
- Real-signal WAV mode
- Metadata/configuration support
- Canonical immutable-ish SignalRecord contract
- Basic normalization and validation
- Provenance tracking
- CLI smoke-test entry point

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m signalforge.app.main --help
python -m signalforge.app.main path/to/capture.wav
python -m signalforge.app.main path/to/capture.iq --sample-rate 1000000 --dtype int16 --iq-layout iqiq
```

The raw input is never modified. Derived/normalized samples are separate arrays.
