import numpy as np
from signalforge.demod.psk import bpsk, qpsk
from signalforge.demod.qam import qam16_demodulate


def test_bpsk_demod():
    symbols = np.array([1, -1, 1, -1], dtype=np.complex128)
    bits, soft, metrics = bpsk(symbols)
    assert bits.tolist() == [0, 1, 0, 1]
    assert soft.size == bits.size


def test_qpsk_demod_has_two_bits_per_symbol():
    symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    bits, soft, _ = qpsk(symbols)
    assert bits.size == 8
    assert soft.size == 8


def test_16qam_demod():
    symbols = np.array([
        (-3-3j)/np.sqrt(10),
        (-1+1j)/np.sqrt(10),
        (3-1j)/np.sqrt(10),
        (1+3j)/np.sqrt(10),
    ])
    bits, soft, metrics = qam16_demodulate(symbols)
    assert bits.size == 16
    assert soft.size == 16
    assert metrics["evm_rms"] < 1e-6
