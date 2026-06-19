"""Nested masking (path consistency) and the entropy-injected prior."""
import pytest

torch = pytest.importorskip("torch")

from siflow.masking import nested_masks, entropy_inject, sample_st
from siflow.schedule import NoiseSchedule

MASK = 100
V = 100  # tokens 0..99; mask id sits outside


def test_nested_and_values():
    sc = NoiseSchedule()
    x0 = torch.randint(0, V, (8, 64))
    s = torch.full((8,), 0.2)
    t = torch.full((8,), 0.7)
    x_s, x_t, keep_s, keep_t, _ = nested_masks(x0, s, t, MASK, sc)
    # revealed(s) subset revealed(t): keep_s True implies keep_t True
    assert (keep_t | ~keep_s).all()
    assert (x_s[~keep_s] == MASK).all() and (x_t[~keep_t] == MASK).all()
    assert (x_s[keep_s] == x0[keep_s]).all()


def test_entropy_inject():
    sc = NoiseSchedule()
    x0 = torch.randint(0, V, (4, 32))
    s = torch.zeros(4)
    t = torch.full((4,), 0.5)
    x_s, _, keep_s, _, _ = nested_masks(x0, s, t, MASK, sc)
    xi = entropy_inject(x_s, keep_s, MASK, V, lam=1.0)
    assert (xi[~keep_s] != MASK).all()        # every masked slot got a real token
    assert (xi[keep_s] == x_s[keep_s]).all()  # revealed slots untouched


def test_sample_st_forcing():
    s, t = sample_st(2000, "cpu", p0=1.0, p1=0.0)
    assert (s == 0).all() and (t > s).all()
    s2, t2 = sample_st(2000, "cpu", p0=0.0, p1=1.0)
    assert (t2 == 1).all() and (t2 > s2).all()
