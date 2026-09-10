import numpy as np
from signalforge.observatory.plots import time_domain, spectrum, waterfall, constellation


def test_observatory_outputs():
    x = np.exp(1j * 2*np.pi*0.05*np.arange(4096))
    assert time_domain(x, 1000)["i"].size > 0
    assert spectrum(x, 1000)["power_db"].size > 0
    assert waterfall(x, 1000)["power_db"].ndim == 2
    assert constellation(x)["count"] > 0
