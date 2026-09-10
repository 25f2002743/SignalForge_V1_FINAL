import numpy as np
from signalforge.core.signal_record import SignalRecord
from signalforge.preprocessing.pipeline import prepare


def test_prepare_removes_dc_and_normalizes():
    x = np.ones(1000, dtype=np.float64) * 7
    x += np.sin(np.arange(1000) / 10)
    r = SignalRecord(x, 1000)
    out = prepare(r)
    assert abs(np.mean(out.samples)) < 1e-10
    assert np.max(np.abs(out.samples)) <= 1.0000001
