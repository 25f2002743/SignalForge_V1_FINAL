import numpy as np
from signalforge.dsp.frequency import estimate_coarse_cfo
from signalforge.dsp.matched_filter import rrc_taps
from signalforge.dsp.synchronizer import symbol_sample


def test_coarse_cfo_tone():
    fs = 10000
    f = 1000
    t = np.arange(10000) / fs
    x = np.exp(2j * np.pi * f * t)
    est = estimate_coarse_cfo(x, fs)
    assert abs(est - f) < 1.0


def test_rrc_normalized():
    taps = rrc_taps(4)
    assert taps.size > 0
    assert abs(np.sqrt(np.sum(taps*taps)) - 1.0) < 1e-6


def test_symbol_sampling():
    x = np.arange(100, dtype=float)
    y = symbol_sample(x, 4, 0)
    assert np.allclose(y[:5], [0, 4, 8, 12, 16])
