"""Dream-7B teacher (SIFLOW-D).

``Dream-org/Dream-v0-Base-7B`` (or the Instruct variant) -- a 7B open masked DLM
initialized from Qwen2.5-7B, ~14 GB in fp16/bf16. Qwen2.5 tokenizer (~152k vocab).
Used only by the offline cache builder; see :class:`HFMaskedDLMTeacher`.
"""
from __future__ import annotations

from typing import Optional

import torch

from .hf_dlm import HFMaskedDLMTeacher

_DEFAULT = "Dream-org/Dream-v0-Base-7B"


class DreamTeacher(HFMaskedDLMTeacher):
    def __init__(
        self,
        name: str = _DEFAULT,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        mask_token: Optional[str | int] = None,
        auto_class: Optional[str] = "AutoModel",
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(
            name=name, device=device, dtype=dtype, mask_token=mask_token,
            auto_class=auto_class, attn_implementation=attn_implementation,
        )
