import numpy as np
from signalforge.deinterleave.block import block_deinterleave
from signalforge.deinterleave.convolutional import convolutional_deinterleave
from signalforge.fec.viterbi import viterbi_decode_soft


def test_block_deinterleave_shape():
    b = np.arange(12, dtype=np.uint8)
    out = block_deinterleave(b, 3, 4)
    assert out.size == 12


def test_convolutional_deinterleave_preserves_bits():
    b = np.random.default_rng(1).integers(0, 2, 32, dtype=np.uint8)
    out = convolutional_deinterleave(b, 4)
    assert sorted(out.tolist()) == sorted(b.tolist())


def test_viterbi_output():
    bits = np.zeros(40, dtype=float) + 5.0
    out = viterbi_decode_soft(bits)
    assert out.size == 20
    assert np.all(out == 0)
