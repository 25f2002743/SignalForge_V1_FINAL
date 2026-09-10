import numpy as np
from signalforge.ingestion.iq_parser import _to_complex


def test_iqiq_layout():
    x = np.array([1, 10, 2, 20, 3, 30], dtype=np.int16)
    y = _to_complex(x, "iqiq")
    assert np.allclose(y, [1+10j, 2+20j, 3+30j])


def test_iiqq_layout():
    x = np.array([1, 2, 3, 10, 20, 30], dtype=np.int16)
    y = _to_complex(x, "iiqq")
    assert np.allclose(y, [1+10j, 2+20j, 3+30j])
