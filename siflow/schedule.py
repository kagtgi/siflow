"""Noise schedules for SIFLOW.

Time conventions
----------------
We use the paper's *generation* time ``s in [0, 1]``:

* ``s = 0``  -> fully masked prior  (everything is ``[M]``)
* ``s = 1``  -> clean data          (nothing is masked)

``mask_frac(s)`` is the fraction of positions that are masked at level ``s``;
it is *monotonically decreasing* (1 at ``s=0``, 0 at ``s=1``). The masked
diffusion forward time used by MDLM is ``t_forward = 1 - s``.

The MDLM teacher we distil has ``time_conditioning = false`` -- its predictive
distribution depends only on *which* tokens are masked, not on a time embedding.
The schedule therefore only needs to decide *how many* tokens are masked at each
level; any monotone schedule is admissible. ``loglinear`` reproduces MDLM's
default (its ``sigma`` is log-linear, which makes the masking fraction linear in
forward time).

This module is intentionally torch-free (pure numpy / Python floats) so the
nesting and monotonicity guarantees that ``masking.nested_masks`` relies on can
be unit-tested without a GPU / torch install.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_KINDS = ("loglinear", "linear", "cosine")


def _as_array(x):
    arr = np.asarray(x, dtype=np.float64)
    return arr, arr.ndim == 0


def mask_frac(s, kind: str = "loglinear", eps: float = 1e-3):
    """Fraction of masked positions at generation time ``s`` (decreasing in s).

    Accepts a Python float or any array-like; returns the same shape.
    ``eps`` keeps the prior from being *exactly* fully masked, matching MDLM.
    """
    arr, scalar = _as_array(s)
    arr = np.clip(arr, 0.0, 1.0)
    tf = 1.0 - arr  # forward (masking) time
    if kind == "loglinear":
        # MDLM: sigma(tf) = -log(1 - (1-eps) tf)  ->  move_chance = (1-eps) tf
        frac = (1.0 - eps) * tf
    elif kind == "linear":
        frac = tf
    elif kind == "cosine":
        frac = np.cos(0.5 * np.pi * arr)
    else:
        raise ValueError(f"unknown schedule kind {kind!r}; choose from {_KINDS}")
    frac = np.clip(frac, 0.0, 1.0)
    return float(frac) if scalar else frac


def n_keep(s, L: int, kind: str = "loglinear", eps: float = 1e-3):
    """Number of *unmasked* positions at level ``s`` for a length-``L`` sequence.

    Monotonically *non-decreasing* in ``s`` (the key invariant that makes
    ``masking.nested_masks`` produce nested mask sets).
    """
    keep_frac = 1.0 - mask_frac(s, kind=kind, eps=eps)
    arr, scalar = _as_array(keep_frac)
    counts = np.rint(arr * L).astype(np.int64)
    counts = np.clip(counts, 0, L)
    return int(counts) if scalar else counts


def sigma_forward(t_forward, kind: str = "loglinear", eps: float = 1e-3):
    """MDLM-style ``sigma`` at *forward* time (t_forward=0 clean, =1 masked)."""
    arr, scalar = _as_array(t_forward)
    arr = np.clip(arr, 0.0, 1.0)
    if kind in ("loglinear", "linear"):
        sig = -np.log1p(-(1.0 - eps) * arr)
    elif kind == "cosine":
        mc = np.cos(0.5 * np.pi * (1.0 - arr))  # move-chance, then invert
        sig = -np.log1p(-np.clip(mc, 0.0, 1.0 - 1e-6))
    else:
        raise ValueError(f"unknown schedule kind {kind!r}")
    return float(sig) if scalar else sig


@dataclass(frozen=True)
class NoiseSchedule:
    """Convenience wrapper bundling a schedule kind + eps."""

    kind: str = "loglinear"
    eps: float = 1e-3

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown schedule kind {self.kind!r}; choose from {_KINDS}")

    def mask_frac(self, s):
        return mask_frac(s, kind=self.kind, eps=self.eps)

    def n_keep(self, s, L: int):
        return n_keep(s, L, kind=self.kind, eps=self.eps)

    def sigma_forward(self, t_forward):
        return sigma_forward(t_forward, kind=self.kind, eps=self.eps)
