"""LLaDA-8B teacher (SIFLOW-L).

``GSAI-ML/LLaDA-8B-Base`` -- an 8B open masked diffusion LM (LLaMA-style
architecture, ~16 GB in fp16/bf16), so it fits a single A100-40GB alongside the
tiny velocity head with no quantization. Vocabulary ~126k.

LLaDA has no native HuggingFace ``mask_token_id``: its mask id is the fixed
constant **126336** (see the official ``ML-GSAI/LLaDA`` repo), so we default
``mask_token=126336``. It loads via ``AutoModel``/``AutoModelForCausalLM`` with
``trust_remote_code=True`` and exposes per-position ``.logits``; the final hidden
state is captured by the un-embed forward hook in :class:`HFMaskedDLMTeacher`.
"""
from __future__ import annotations

from typing import Optional

import torch

from .hf_dlm import HFMaskedDLMTeacher

_DEFAULT = "GSAI-ML/LLaDA-8B-Base"
_LLADA_MASK_ID = 126336


class LLaDATeacher(HFMaskedDLMTeacher):
    def __init__(
        self,
        name: str = _DEFAULT,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        mask_token: Optional[str | int] = None,
        auto_class: Optional[str] = "AutoModel",
        attn_implementation: Optional[str] = None,
    ):
        # LLaDA exposes no tokenizer/config mask id -> default to the known constant.
        if mask_token is None:
            mask_token = _LLADA_MASK_ID
        super().__init__(
            name=name, device=device, dtype=dtype, mask_token=mask_token,
            auto_class=auto_class, attn_implementation=attn_implementation,
        )
