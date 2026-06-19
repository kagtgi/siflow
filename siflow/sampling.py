"""Reference samplers.

* :func:`teacher_ancestral_sample` -- the standard masked-diffusion ancestral
  sampler used for the *teacher step-curve* baseline (e.g. MDLM at 1024/64/32/8
  steps). NFE == ``num_steps``.

The SIFLOW student's one-/few-step sampler lives on ``Student.generate``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .schedule import NoiseSchedule
from .teacher.base import Teacher


def _alpha(schedule: NoiseSchedule, t_forward: float) -> float:
    """Survival prob (fraction *unmasked*) at forward time ``t_forward``."""
    return float(np.exp(-schedule.sigma_forward(t_forward)))


@torch.no_grad()
def teacher_ancestral_sample(
    teacher: Teacher,
    batch_size: int,
    length: int,
    num_steps: int,
    schedule: Optional[NoiseSchedule] = None,
    sample: bool = True,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Ancestral reverse sampling of a masked DLM.

    At each step ``(t -> s)`` (forward times, ``t > s``) every still-masked
    position is independently unmasked with probability
    ``(alpha_s - alpha_t) / (1 - alpha_t)`` and, if unmasked, drawn from the
    teacher's predicted x0 distribution.
    """
    schedule = schedule or NoiseSchedule()
    dev = teacher.device
    mask_id = teacher.mask_index
    x = torch.full((batch_size, length), mask_id, dtype=torch.long, device=dev)
    ts = torch.linspace(1.0, 0.0, num_steps + 1).tolist()  # forward times, 1 -> 0

    for i in range(num_steps):
        t_f, s_f = ts[i], ts[i + 1]
        a_t, a_s = _alpha(schedule, t_f), _alpha(schedule, s_f)
        unmask_p = 0.0 if (1.0 - a_t) <= 1e-9 else max(0.0, min(1.0, (a_s - a_t) / (1.0 - a_t)))

        logits = teacher.logits(x).float()
        probs = torch.softmax(logits / max(temperature, 1e-3), dim=-1)
        if sample:
            flat = probs.view(-1, probs.shape[-1])
            drawn = torch.multinomial(flat, 1, generator=generator).view(batch_size, length)
        else:
            drawn = probs.argmax(dim=-1)

        masked = x == mask_id
        coin = torch.rand(x.shape, device=dev, generator=generator) < unmask_p
        do_unmask = masked & coin
        x = torch.where(do_unmask, drawn, x)

    # final commit of any leftover masks
    masked = x == mask_id
    if masked.any():
        logits = teacher.logits(x).float()
        drawn = logits.argmax(dim=-1)
        x = torch.where(masked, drawn, x)
    return x


@torch.no_grad()
def teacher_complete(teacher: Teacher, input_ids: torch.Tensor, fill_mask: torch.Tensor,
                     num_steps: int = 8) -> torch.Tensor:
    """Confidence-based teacher completion of ``fill_mask`` positions (for the
    LAMBADA teacher baseline). Reveals most-confident masked positions over
    ``num_steps`` passes; the rest of ``input_ids`` is fixed context."""
    dev = teacher.device
    mask_id = teacher.mask_index
    tokens = input_ids.clone().to(dev)
    tokens[fill_mask] = mask_id
    committed = (~fill_mask).to(dev)
    base = committed.clone()
    num_mask = (~base).sum(dim=1)
    for j in range(1, num_steps + 1):
        probs = torch.softmax(teacher.logits(tokens).float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        pred = torch.where(committed, tokens, pred)
        conf = conf.masked_fill(committed, float("inf"))
        reveal = torch.ceil((j / num_steps) * num_mask).long()
        target_total = base.sum(dim=1) + reveal
        order = conf.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)
        committed = (rank < target_total.view(-1, 1)) | base
        tokens = torch.where(committed, pred, torch.full_like(tokens, mask_id))
    if not committed.all():
        pred = teacher.logits(tokens).float().argmax(dim=-1)
        tokens = torch.where(committed, tokens, pred)
    return tokens
