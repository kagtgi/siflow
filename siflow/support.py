"""Reduced-support (top-m) representation for large-vocabulary teachers.

Dream (~152k) and DiffusionGemma (~256k) vocabularies make full-vocab student
tensors expensive. We restrict the SIFLOW loss to a per-(example, position)
support of the ``m`` tokens carrying the most mass across *both* endpoints, plus
a single folded ``rest`` bucket::

    support = top-m of (softmax(z_s) + softmax(z_t))         # [..., m]
    z_*_red = [ z_*.gather(support) , logsumexp(z_* over the complement) ]   # [..., m+1]

The velocity head only displaces the ``m`` real tokens; the ``rest`` bucket is
carried along with zero velocity. MDLM (50k vocab) uses the exact full-vocab
path and never needs this.
"""
from __future__ import annotations

from typing import Tuple

import torch


def reduce_to_support(z_s: torch.Tensor, z_t: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(support_idx [.., m], z_s_red [.., m+1], z_t_red [.., m+1])``.

    The last column of each reduced logit tensor is the folded ``rest`` bucket.
    """
    with torch.no_grad():
        combined = torch.softmax(z_s.float(), -1) + torch.softmax(z_t.float(), -1)
        support = combined.topk(m, dim=-1).indices                       # [.., m]
        z_s_sup = z_s.float().gather(-1, support)
        z_t_sup = z_t.float().gather(-1, support)
        rest_s = _rest_logit(z_s.float(), z_s_sup)
        rest_t = _rest_logit(z_t.float(), z_t_sup)
        z_s_red = torch.cat([z_s_sup, rest_s.unsqueeze(-1)], dim=-1)
        z_t_red = torch.cat([z_t_sup, rest_t.unsqueeze(-1)], dim=-1)
    return support, z_s_red, z_t_red


def _rest_logit(z_full: torch.Tensor, z_sup: torch.Tensor) -> torch.Tensor:
    """logsumexp of the complement of the support: log(sum exp z_full - sum exp z_sup)."""
    lse_all = torch.logsumexp(z_full, dim=-1)
    lse_sup = torch.logsumexp(z_sup, dim=-1)
    # log(exp(lse_all) - exp(lse_sup)); clamp the tiny/negative residual to a floor
    diff = lse_all + torch.log1p(-torch.exp((lse_sup - lse_all).clamp(max=-1e-6)))
    floor = torch.finfo(z_full.dtype).min / 2
    return torch.nan_to_num(diff, neginf=floor)
