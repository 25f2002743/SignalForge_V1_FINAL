# SignalForge V1 — Final Integration

SignalForge is an automated .IQ/.WAV signal-analysis prototype aligned with the
PS 26147 workflow.

## End-to-end pipeline

Input
→ ingestion
→ preprocessing
→ signal observatory
→ parameter extraction
→ modulation hypotheses
→ synchronization
→ demodulation
→ de-interleaving
→ FEC
→ bitstream analysis
→ frame/header/payload candidates
→ validation
→ GUI/export.

## Run

Install the Python dependency:

    pip install -r requirements.txt

Launch:

    python run_signalforge.py

or:

    python -m signalforge

## V1 behavior

The system is evidence-driven. Unknown protocol details are not silently invented.
Candidate modulation, de-interleaving, FEC and framing results remain hypotheses until
supported by measurable evidence.

RS/LDPC/concatenated FEC paths are explicit adapter boundaries when protocol-specific
parameters are unavailable.

## Validation

The package includes unit tests for demodulation, de-interleaving, Viterbi, framing,
correlation/quality utilities, plus an end-to-end smoke-test structure.

## Output

The GUI provides an analysis summary and JSON export. Observatory functions expose
time-domain, FFT/spectrum, waterfall and constellation datasets for richer plotting
frontends.

This is a V1 engineering prototype, not a claim of universal automatic protocol
identification for arbitrary terrestrial signals.
