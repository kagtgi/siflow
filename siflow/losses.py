"""SIFLOW training losses.

All losses operate over the last (vocabulary or reduced-support) dimension and
respect the SUBS ``-inf`` structure -- target probabilities are zero exactly
where the teacher logit is ``-inf`` (mask column, pinned positions), and those
terms contribute nothing to the KL.

``loss_mask`` (bool ``[B, L]``, typically ``~keep_s`` = positions masked at level
``s``) restricts the average to positions where the student actually predicts.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

# Floor for log-probabilities. SUBS encodes structural zeros as finfo.min
# (~-3.4e38); when the student pins ~0 mass on a token the target still supports
# (e.g. an entropy-injected position), the raw KL term p*(logp - log q) overflows
# float32. Flooring log q at a large-but-finite value keeps the (legitimately
# large) penalty representable and the gradient well-behaved.
LOG_FLOOR = -100.0


def beta_schedule(step: int, total_steps: int, beta_max: float = 2.0, anneal_frac: float = 0.5) -> float:
    """Cosine anneal ``beta_max -> 1`` over the first ``anneal_frac`` of training,
    then hold at 1 (the teacher's own sharpness)."""
    if beta_max <= 1.0 or anneal_frac <= 0.0 or total_steps <= 0:
        return 1.0
    horizon = max(1.0, anneal_frac * total_steps)
    frac = min(step / horizon, 1.0)
    return 1.0 + (beta_max - 1.0) * 0.5 * (1.0 + math.cos(math.pi * frac))


def masked_mean(x: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if loss_mask is None:
        return x.mean()
    m = loss_mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def satd_kl(
    z_t: torch.Tensor,
    log_mu_hat: torch.Tensor,
    beta: float,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Simplex-Annealed Temperature Distillation loss::

        KL( sg[softmax(z_t / beta)]  ||  mu_hat )

    averaged over ``loss_mask`` positions. ``z_t`` are teacher SUBS logits at the
    cleaner level ``t`` (the target); ``log_mu_hat`` is the student log-distribution.
    """
    log_p = F.log_softmax(z_t.float() / beta, dim=-1)
    p = log_p.exp()
    term = p * (log_p - log_mu_hat.clamp_min(LOG_FLOOR))
    term = torch.where(p > 0, term, torch.zeros_like(term))
    kl = term.sum(dim=-1)  # [B, L]
    return masked_mean(kl, loss_mask)


def secant_mse(
    mu_hat: torch.Tensor,
    mu_t: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prob-space secant / velocity-matching term.

    Matching ``mu_hat -> mu_t`` is exactly matching the student's prob-space
    displacement ``(mu_hat - mu_s)/(t-s)`` to the secant target
    ``(mu_t - mu_s)/(t-s)`` -- a finite, boundary-safe alternative to the
    logit-space velocity L2 (whose ``-inf`` entries are ill-defined).
    """
    sq = (mu_hat - mu_t).pow(2).sum(dim=-1)  # [B, L]
    return masked_mean(sq, loss_mask)


def mdm_ce(
    log_mu_pred: torch.Tensor,
    x0: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked-diffusion self-consistency: NLL of the true token at masked
    positions (keeps the head on the data manifold). Full-vocab only."""
    nll = -log_mu_pred.gather(-1, x0.unsqueeze(-1)).squeeze(-1).clamp_max(-LOG_FLOOR)  # [B, L]
    return masked_mean(nll, loss_mask)
