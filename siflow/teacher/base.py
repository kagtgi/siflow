"""Teacher abstraction for SIFLOW.

A ``Teacher`` is a *frozen* pretrained masked diffusion LM. It exposes, for a
batch of (partially masked) token ids, the per-position predictive distribution
on the vocabulary simplex -- the object SIFLOW distils on.

All teachers return logits **after the SUBS parameterization** so that

* the mask token never receives probability mass, and
* already-revealed (unmasked) positions are pinned to their observed token.

Concrete teachers (MDLM / Dream / DiffusionGemma) subclass this and only differ
in how they load the backbone and where the embedding / hidden states live.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Teacher(ABC):
    name: str
    mask_index: int
    vocab_size: int
    hidden_dim: int
    device: torch.device
    dtype: torch.dtype

    # -- core interface -------------------------------------------------------
    @abstractmethod
    def logits_and_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(subs_logits [B, L, V], hidden [B, L, H])`` with no grad."""

    @property
    @abstractmethod
    def embedding_matrix(self) -> torch.Tensor:
        """Frozen input-embedding matrix ``E`` of shape ``[V, H]``.

        SIFLOW's velocity head reuses ``E`` as a frozen un-embedding: a
        hidden-space displacement ``d`` is lifted to a vocab-logit displacement
        ``d @ E.T``. This keeps the head at ~1-3M trainable params (the lift adds
        none) at the cost of restricting velocities to the row space of ``E`` --
        a deliberate, documented low-rank inductive bias.
        """

    # -- conveniences ---------------------------------------------------------
    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_and_hidden(input_ids)[0]

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_and_hidden(input_ids)[1]

    @torch.no_grad()
    def mu(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predictive simplex points ``softmax(subs_logits)`` -> ``[B, L, V]``."""
        return torch.softmax(self.logits(input_ids).float(), dim=-1)

    def subs(self, raw_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply the SUBS parameterization to raw backbone logits.

        * mask column -> -inf (no mass on ``[M]``)
        * unmasked positions -> one-hot at the observed token

        This matches ``Diffusion._subs_parameterization`` in the MDLM codebase,
        which the HF ``forward`` does **not** apply.
        """
        neg_inf = torch.finfo(raw_logits.dtype).min
        out = raw_logits.clone()
        out[..., self.mask_index] = neg_inf
        unmasked = input_ids != self.mask_index            # [B, L]
        if unmasked.any():
            out[unmasked] = neg_inf
            idx = input_ids[unmasked]
            out[unmasked, idx] = 0.0
        return out
