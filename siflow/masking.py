"""Masking utilities for SIFLOW.

The central routine is :func:`nested_masks`, which builds two masked views
``x_s`` (noisier, more masked) and ``x_t`` (cleaner, fewer masked) of the *same*
clean sequence such that the revealed positions are **nested**::

    revealed(s)  subset of  revealed(t)        (equivalently  mask(t) subset of mask(s))

This guarantees the teacher's two predictive simplex points ``mu_s`` and
``mu_t`` lie on a *single* reverse trajectory of one sequence, so the secant
``(mu_t - mu_s) / (t - s)`` is the chord of one curve -- the path-consistency
property that reviewers (1, 2, 4) asked us to make explicit.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .schedule import NoiseSchedule


def _keep_counts(times: torch.Tensor, L: int, schedule: NoiseSchedule) -> torch.Tensor:
    """Per-example number of kept (unmasked) positions; shape [B], dtype long."""
    counts = schedule.n_keep(times.detach().cpu().numpy(), L)  # numpy, monotone in time
    return torch.as_tensor(np.asarray(counts).reshape(-1), dtype=torch.long, device=times.device)


def reveal_ranks(B: int, L: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """For each example, a random reveal order: ``rank[b, i]`` is the step at
    which position ``i`` becomes unmasked (0 = revealed first)."""
    scores = torch.rand(B, L, device=device, generator=generator)
    perm = scores.argsort(dim=1)        # perm[b, j] = position revealed j-th
    rank = perm.argsort(dim=1)          # inverse permutation = reveal order per position
    return rank


def apply_mask(x0: torch.Tensor, keep_mask: torch.Tensor, mask_index: int) -> torch.Tensor:
    """Return ``x0`` with positions where ``keep_mask`` is False set to ``mask_index``."""
    return torch.where(keep_mask, x0, torch.full_like(x0, mask_index))


def nested_masks(
    x0: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    mask_index: int,
    schedule: NoiseSchedule,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build nested masked views at levels ``s < t``.

    Parameters
    ----------
    x0 : long [B, L]      clean token ids
    s, t : float [B]      generation times with 0 <= s < t <= 1 (s noisier)
    mask_index : int      id of the ``[M]`` token
    schedule : NoiseSchedule

    Returns
    -------
    x_s, x_t : long [B, L]          masked inputs (more / fewer masks)
    keep_s, keep_t : bool [B, L]    True where the position is *unmasked*
    rank : long [B, L]              the shared reveal order (for debugging / reuse)
    """
    assert x0.dim() == 2, "x0 must be [B, L]"
    B, L = x0.shape
    rank = reveal_ranks(B, L, x0.device, generator=generator)
    keep_s_n = _keep_counts(s, L, schedule).view(B, 1)
    keep_t_n = _keep_counts(t, L, schedule).view(B, 1)
    # nesting is guaranteed because n_keep is monotone non-decreasing in time and
    # both views use the *same* reveal order `rank`.
    keep_s = rank < keep_s_n
    keep_t = rank < keep_t_n
    x_s = apply_mask(x0, keep_s, mask_index)
    x_t = apply_mask(x0, keep_t, mask_index)
    return x_s, x_t, keep_s, keep_t, rank


def entropy_inject(
    x: torch.Tensor,
    keep_mask: torch.Tensor,
    mask_index: int,
    vocab_size: int,
    lam: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Entropy-injected prior (Di[M]O): with prob ``lam`` replace a *masked*
    input position by a uniform-random vocabulary token.

    Only positions that are currently masked (``~keep_mask``) are eligible, so
    the revealed/clean tokens are never corrupted. The mask token id itself is
    never injected.
    """
    if lam <= 0.0:
        return x
    masked = ~keep_mask
    coin = torch.rand(x.shape, device=x.device, generator=generator) < lam
    inject = masked & coin
    rand_tok = torch.randint(0, vocab_size, x.shape, device=x.device, generator=generator)
    # avoid accidentally injecting the mask token
    rand_tok = torch.where(rand_tok == mask_index, (rand_tok + 1) % vocab_size, rand_tok)
    return torch.where(inject, rand_tok, x)


def sample_st(
    B: int,
    device,
    p0: float = 0.25,
    p1: float = 0.25,
    min_gap: float = 1e-3,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample interval endpoints ``0 <= s < t <= 1``.

    With prob ``p0`` force ``s = 0`` (trains the one-step generation start), and
    with prob ``p1`` force ``t = 1`` (trains the clean endpoint). Otherwise both
    are uniform with ``t > s``.
    """
    u = torch.rand(B, 2, device=device, generator=generator)
    a = torch.minimum(u[:, 0], u[:, 1])
    b = torch.maximum(u[:, 0], u[:, 1])
    s = a.clone()
    t = b.clone()
    force0 = torch.rand(B, device=device, generator=generator) < p0
    force1 = torch.rand(B, device=device, generator=generator) < p1
    s = torch.where(force0, torch.zeros_like(s), s)
    t = torch.where(force1, torch.ones_like(t), t)
    # guarantee a positive gap
    t = torch.maximum(t, s + min_gap).clamp(max=1.0)
    s = torch.minimum(s, t - min_gap).clamp(min=0.0)
    return s, t
