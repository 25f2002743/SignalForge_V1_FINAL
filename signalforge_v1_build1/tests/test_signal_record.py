import numpy as np
from signalforge.core.signal_record import SignalRecord


def test_signal_record_duration():
    r = SignalRecord(np.ones(1000, dtype=np.complex64), 1000)
    assert r.num_samples == 1000
    assert r.duration == 1.0
    assert r.is_complex
