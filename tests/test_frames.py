import numpy as np
from signalforge.frame.patterns import find_pattern
from signalforge.frame.parser import bits_to_bytes, byte_entropy
from signalforge.validation.score import candidate_quality


def test_find_pattern_with_one_error():
    b = np.array([0,1,1,0,1,0,1,1,0], dtype=np.uint8)
    matches = find_pattern(b, "11010", max_errors=1)
    assert matches


def test_bits_to_bytes():
    assert bits_to_bytes(np.array([0,1,0,0,0,0,0,1], dtype=np.uint8)) == b"A"


def test_entropy_nonnegative():
    assert byte_entropy(b"AAAA") >= 0


def test_quality_range():
    q = candidate_quality(1, .5, .5, 0)
    assert 0 <= q <= 1
