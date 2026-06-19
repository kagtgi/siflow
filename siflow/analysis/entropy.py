"""Per-position entropy diagnostics (Figure F4; validates SATD).

``onestep_entropy`` measures the entropy of a trained student's one-step output
distribution at masked positions; aggregating across variants (full SIFLOW vs.
hard-label vs. teacher) shows that SATD avoids the collapsed, over-confident
distributions of the hard-label baseline.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..masking import nested_masks, sample_st
from ..schedule import NoiseSchedule


def _entropy(p: torch.Tensor) -> torch.Tensor:
    logp = torch.log(p.clamp_min(1e-12))
    return -(p * logp).sum(dim=-1)


@torch.no_grad()
def onestep_entropy(student, token_dataset, schedule: NoiseSchedule = None,
                    n_examples: int = 256, batch_size: int = 16, seed: int = 0,
                    device=None) -> Dict[str, object]:
    """Entropy of the student's one-step (s=0, t=1) distribution at masked positions."""
    schedule = schedule or NoiseSchedule()
    device = device or student.teacher.device
    L = token_dataset.seq_len
    rng = np.random.default_rng(seed)
    idx_all = rng.permutation(len(token_dataset))[:n_examples]
    ent = []
    for b in range(0, len(idx_all), batch_size):
        idx = idx_all[b: b + batch_size]
        x0 = token_dataset.batch(idx, device=device)
        B = x0.shape[0]
        x_mask = torch.full_like(x0, student.teacher.mask_index)
        z0, h0 = student.teacher.logits_and_hidden(x_mask)
        s = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        pred = student.predict(z0, h0, s, t)
        ent.append(_entropy(pred.mu_hat).flatten().cpu().numpy())
    e = np.concatenate(ent) if ent else np.array([0.0])
    return {"entropy_mean": float(e.mean()), "entropy_std": float(e.std()), "entropy": e.tolist()}


@torch.no_grad()
def target_entropy(teacher, token_dataset, beta: float = 1.0, schedule: NoiseSchedule = None,
                   n_examples: int = 128, batch_size: int = 16, seed: int = 0, device=None) -> Dict[str, float]:
    """Entropy of the SATD soft target ``softmax(z_t / beta)`` (beta>1 softer)."""
    schedule = schedule or NoiseSchedule()
    device = device or teacher.device
    L = token_dataset.seq_len
    rng = np.random.default_rng(seed)
    idx_all = rng.permutation(len(token_dataset))[:n_examples]
    ent = []
    for b in range(0, len(idx_all), batch_size):
        idx = idx_all[b: b + batch_size]
        x0 = token_dataset.batch(idx, device=device)
        B = x0.shape[0]
        g = torch.Generator(device=device).manual_seed(seed + b)
        s, t = sample_st(B, device, generator=g)
        _, x_t, _, keep_t, _ = nested_masks(x0, s, t, teacher.mask_index, schedule, generator=g)
        z_t = teacher.logits(x_t)
        p = torch.softmax(z_t.float() / beta, dim=-1)
        ent.append(_entropy(p).flatten().cpu().numpy())
    e = np.concatenate(ent) if ent else np.array([0.0])
    return {"target_entropy_mean": float(e.mean()), "beta": beta}
