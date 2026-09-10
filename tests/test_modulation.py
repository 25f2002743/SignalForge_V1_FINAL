import numpy as np
from signalforge.core.signal_record import SignalRecord
from signalforge.observatory.engine import observe
from signalforge.parameters.estimate import extract_parameters
from signalforge.modulation.advisor import advise
from signalforge.hypotheses.generator import generate


def test_modulation_advisor_and_hypotheses():
    fs = 10000
    t = np.arange(20000) / fs
    x = np.exp(2j * np.pi * 1000 * t)
    r = SignalRecord(x, fs)
    observe(r, waterfall_fft_size=256)
    extract_parameters(r)
    result = advise(r)
    assert len(result["candidates"]) == 5
    hs = generate(r)
    assert len(hs) > 0
    assert hs[0].id == "H1"
