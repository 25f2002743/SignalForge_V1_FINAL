import numpy as np
from signalforge.core.signal_record import SignalRecord
from signalforge.observatory.engine import observe
from signalforge.parameters.estimate import extract_parameters


def test_parameter_extraction_tone():
    fs = 20000
    t = np.arange(20000) / fs
    x = np.exp(2j * np.pi * 2000 * t)
    r = SignalRecord(x, fs)
    observe(r, waterfall_fft_size=256)
    p = extract_parameters(r)

    assert p["sample_rate"].status == "KNOWN"
    assert p["duration"].value == 1.0
    assert "bandwidth" in p
    assert "spectral_peaks" in p
    assert "envelope" in p
    assert "frequency_offset_candidates" in p
