"""DiffusionGemma teacher (SIFLOW-G).

``google/diffusiongemma-26B-A4B-it`` -- Gemma-4 architecture, 25.2B-param MoE
(~3.8B active), ~50 GB fp16 (fits a single A100-80GB). Gemma tokenizer.
Used only by the offline cache builder; see :class:`HFMaskedDLMTeacher`.

For the MoE backbone we default to eager attention; pass
``attn_implementation="flash_attention_2"`` in the config if available.
"""
from __future__ import annotations

from typing import Optional

import torch

from .hf_dlm import HFMaskedDLMTeacher

_DEFAULT = "google/diffusiongemma-26B-A4B-it"


class GemmaTeacher(HFMaskedDLMTeacher):
    def __init__(
        self,
        name: str = _DEFAULT,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        mask_token: Optional[str | int] = None,
        auto_class: Optional[str] = "AutoModelForMaskedLM",
        attn_implementation: Optional[str] = "eager",
    ):
        super().__init__(
            name=name, device=device, dtype=dtype, mask_token=mask_token,
            auto_class=auto_class, attn_implementation=attn_implementation,
        )
