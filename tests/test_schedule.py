"""Torch-free tests for the noise schedule (runs anywhere numpy is available)."""
import numpy as np

from siflow.schedule import NoiseSchedule, n_keep, mask_frac


def test_boundaries():
    assert n_keep(0.0, 256) == 0          # fully masked prior
    assert n_keep(1.0, 256) == 256        # clean
    assert mask_frac(1.0) == 0.0


def test_monotone_keep():
    L = 256
    ks = np.array([n_keep(s, L) for s in np.linspace(0, 1, 64)])
    assert (np.diff(ks) >= 0).all(), "n_keep must be non-decreasing (nesting invariant)"


def test_nesting_invariant():
    L = 128
    for s, t in [(0.1, 0.4), (0.0, 1.0), (0.5, 0.51), (0.3, 0.9)]:
        assert n_keep(s, L) <= n_keep(t, L)


def test_kinds():
    for kind in ("loglinear", "linear", "cosine"):
        sc = NoiseSchedule(kind=kind)
        assert sc.mask_frac(0.0) > 0.9 and sc.mask_frac(1.0) < 1e-6
