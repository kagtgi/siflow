"""The SIFLOW velocity head.

The student is ``frozen teacher backbone + VelocityHead``. The head is the only
trainable module (~1-3M params). Given the teacher hidden state ``h`` at level
``s`` and the interval ``(s, t)``, it predicts a velocity ``U_theta``:

* **logit space (primary):**  ``z_hat_t = z_s + (t - s) * U``,  ``mu_hat = softmax(z_hat_t)``.
  Numerically safe -- the softmax keeps ``mu_hat`` on the simplex and respects the
  SUBS ``-inf`` structure automatically.
* **prob space (ablation):**  ``U`` is projected to the sum-zero tangent and
  ``mu_hat = renorm(clip(mu_s + (t - s) * U))``.

To keep the parameter count tiny, the hidden-space displacement ``delta_h`` is
lifted to a vocab-logit velocity through the **frozen** teacher embedding ``E``
(``U = scale * delta_h @ E.T``). ``scale`` is zero-initialized, so at step 0
``U = 0`` and ``mu_hat_t = mu_s`` -- a safe, well-defined starting point.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _sinusoidal(x: torch.Tensor, dim: int, max_period: float = 1.0e4) -> torch.Tensor:
    """Sinusoidal embedding of a scalar in [0, 1]; returns [..., dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / max(half, 1)
    )
    ang = x.float().unsqueeze(-1) * freqs * (2.0 * math.pi)
    emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim
        emb = torch.cat([emb, torch.zeros(*emb.shape[:-1], 1, device=x.device)], dim=-1)
    return emb


class TimeEmbed2D(nn.Module):
    """Embed the interval endpoints ``(s, t)`` jointly (interval-aware)."""

    def __init__(self, out_dim: int = 128, scalar_dim: int = 64):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * scalar_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        es = _sinusoidal(s, self.scalar_dim)
        et = _sinusoidal(t, self.scalar_dim)
        return self.mlp(torch.cat([es, et], dim=-1))  # [B, out_dim]


class VelocityHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding: torch.Tensor,            # frozen E [V, H], used as un-embed
        bottleneck: int = 1024,
        time_dim: int = 128,
        space: str = "logit",
        mid_blocks: int = 0,
    ):
        super().__init__()
        assert space in ("logit", "prob")
        self.space = space
        self.hidden_dim = hidden_dim
        self.vocab_size = int(embedding.shape[0])
        # frozen un-embedding (registered as buffer so it moves with .to / .state_dict
        # but is never optimized)
        self.register_buffer("E", embedding.detach(), persistent=False)

        self.time = TimeEmbed2D(out_dim=time_dim)
        self.in_proj = nn.Linear(hidden_dim, bottleneck)
        self.film = nn.Linear(time_dim, 2 * bottleneck)
        self.act = nn.GELU()
        # optional extra MLP blocks (controls head depth for the 1- vs 2-layer ablation)
        self.mid = nn.ModuleList(
            [nn.Sequential(nn.Linear(bottleneck, bottleneck), nn.GELU()) for _ in range(mid_blocks)]
        )
        self.out_proj = nn.Linear(bottleneck, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # Zero-init the OUTPUT projection only: this makes the head start as the
        # identity map (delta_h = 0 -> U = 0 -> mu_hat = mu_s) while still letting
        # gradients flow into out_proj on the first step (its input is non-zero).
        # NB: do NOT also gate by a zero-init scalar -- a double zero kills the
        # gradient and the head would never leave U=0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def delta_h(self, h: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Hidden-space displacement, FiLM-conditioned on the interval."""
        c = self.time(s, t)                              # [B, time_dim]
        gamma, beta = self.film(c).chunk(2, dim=-1)      # [B, bottleneck] each
        x = self.in_proj(h)                              # [B, L, bottleneck]
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.act(x)
        for block in self.mid:
            x = x + block(x)                             # residual MLP block
        return self.norm(self.out_proj(x))               # [B, L, H]

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        support_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Velocity ``U``.

        * ``support_idx is None`` -> full ``[B, L, V]`` (MDLM).
        * ``support_idx`` ``[B, L, m]`` -> gathered ``[B, L, m]`` (Dream / LLaDA),
          avoiding materializing the full 152k/256k-wide tensor.
        """
        d = self.delta_h(h, s, t)                        # [B, L, H]
        E = self.E.to(d.dtype)
        if support_idx is None:
            U = torch.matmul(d, E.t())                   # [B, L, V]
        else:
            E_sup = E[support_idx]                        # [B, L, m, H]
            U = (E_sup * d.unsqueeze(2)).sum(dim=-1)      # [B, L, m]
        if self.space == "prob":
            U = U - U.mean(dim=-1, keepdim=True)          # tangent (sum-zero)
        return U
