"""The SIFLOW student: a frozen teacher backbone + a trainable velocity head.

Provides the training-time prediction (``predict``) used by the losses, and the
inference-time one-/few-step generator (``generate``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import VelocityHead
from .schedule import NoiseSchedule
from .teacher.base import Teacher

_EPS = 1e-8


@dataclass
class Prediction:
    log_mu_hat: torch.Tensor   # [B, L, V] (or [B, L, m] for gathered support)
    mu_hat: torch.Tensor       # [B, L, V]
    U: torch.Tensor            # velocity


class Student(nn.Module):
    def __init__(self, teacher: Teacher, head: VelocityHead, schedule: Optional[NoiseSchedule] = None):
        super().__init__()
        self.teacher = teacher            # frozen; not registered as a submodule
        self.head = head
        self.schedule = schedule or NoiseSchedule()
        self.space = head.space

    # -- training-time prediction --------------------------------------------
    def predict(
        self,
        z_s: torch.Tensor,
        h_s: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        support_idx: Optional[torch.Tensor] = None,
    ) -> Prediction:
        """Student endpoint distribution implied by the velocity head.

        ``z_s`` are the teacher SUBS logits at level ``s`` (may contain ``-inf``);
        ``h_s`` the teacher hidden state at level ``s``.

        Reduced support (Dream / LLaDA): ``z_s`` has shape ``[B, L, m+1]`` where the
        first ``m`` columns are the union-top-K logits and the last is the folded
        ``rest`` bucket. ``support_idx`` is ``[B, L, m]``; the head only displaces the
        ``m`` real tokens and the ``rest`` bucket is carried along with zero velocity.
        """
        dt = (t - s).clamp_min(1e-6).view(-1, 1, 1)
        U = self.head(h_s, s, t, support_idx=support_idx)
        if support_idx is not None and z_s.shape[-1] == support_idx.shape[-1] + 1:
            U = F.pad(U, (0, 1))  # zero velocity on the 'rest' bucket
        if self.space == "logit":
            z_hat = z_s + dt * U
            log_mu = F.log_softmax(z_hat, dim=-1)
            mu = log_mu.exp()
        else:  # prob space
            mu_s = torch.softmax(z_s, dim=-1)
            mu = (mu_s + dt * U).clamp_min(_EPS)
            mu = mu / mu.sum(dim=-1, keepdim=True)
            log_mu = mu.log()
        return Prediction(log_mu_hat=log_mu, mu_hat=mu, U=U)

    # -- inference ------------------------------------------------------------
    @torch.no_grad()
    def _decode(
        self,
        tokens: torch.Tensor,
        committed: torch.Tensor,
        k: int,
        sample: bool = False,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Shared k-step self-conditioned decoder.

        ``tokens`` holds clean ids at ``committed`` positions and ``mask_index``
        elsewhere; only the un-committed (masked) positions are filled, revealed
        most-confident-first over ``k`` steps. NFE == ``k``.
        """
        dev = self.teacher.device
        mask_id = self.teacher.mask_index
        base = committed.clone()
        num_mask = (~base).sum(dim=1)                        # [B]
        times = torch.linspace(0.0, 1.0, k + 1, device=dev)

        for j in range(1, k + 1):
            s_prev = float(times[j - 1])
            z, h = self.teacher.logits_and_hidden(tokens)
            B = tokens.shape[0]
            U = self.head(h, torch.full((B,), s_prev, device=dev), torch.ones(B, device=dev))
            logits = z.float() + (1.0 - s_prev) * U.float()
            if sample:
                probs = torch.softmax(logits / max(temperature, 1e-3), dim=-1)
                flat = probs.view(-1, probs.shape[-1])
                pred = torch.multinomial(flat, 1, generator=generator).view(B, -1)
                conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            else:
                conf, pred = torch.softmax(logits, dim=-1).max(dim=-1)

            pred = torch.where(committed, tokens, pred)
            conf = conf.masked_fill(committed, float("inf"))     # always keep committed
            reveal_j = torch.ceil(float(times[j]) * num_mask).long()
            target_total = base.sum(dim=1) + reveal_j            # [B]
            order = conf.argsort(dim=1, descending=True)
            rank = order.argsort(dim=1)
            new_committed = (rank < target_total.view(-1, 1)) | base
            tokens = torch.where(new_committed, pred, torch.full_like(tokens, mask_id))
            committed = new_committed

        if not committed.all():                                  # final rounding
            z, h = self.teacher.logits_and_hidden(tokens)
            B = tokens.shape[0]
            U = self.head(h, torch.full((B,), float(times[-2]), device=dev), torch.ones(B, device=dev))
            pred = (z.float() + U.float()).argmax(dim=-1)
            tokens = torch.where(committed, tokens, pred)
        return tokens

    @torch.no_grad()
    def generate(self, batch_size: int, length: int, k: int = 1, sample: bool = False,
                 temperature: float = 1.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Unconditional one-/few-step generation from the fully masked prior."""
        dev = self.teacher.device
        tokens = torch.full((batch_size, length), self.teacher.mask_index, dtype=torch.long, device=dev)
        committed = torch.zeros((batch_size, length), dtype=torch.bool, device=dev)
        return self._decode(tokens, committed, k, sample, temperature, generator)

    @torch.no_grad()
    def complete(self, input_ids: torch.Tensor, fill_mask: torch.Tensor, k: int = 1,
                 sample: bool = False, temperature: float = 1.0,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Conditional completion: fill the ``fill_mask`` positions of ``input_ids``
        (the rest are kept fixed). Used for LAMBADA last-word prediction."""
        dev = self.teacher.device
        tokens = input_ids.clone().to(dev)
        tokens[fill_mask] = self.teacher.mask_index
        committed = (~fill_mask).to(dev)
        return self._decode(tokens, committed, k, sample, temperature, generator)
