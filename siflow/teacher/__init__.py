"""Teacher factory."""
from __future__ import annotations

import torch

from .base import Teacher
from .mdlm import MDLMTeacher
from .dream import DreamTeacher
from .llada import LLaDATeacher
from .gemma import GemmaTeacher  # retained for future 80GB / multi-GPU work (not in the 2-NB study)

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16,
           "fp32": torch.float32, "float32": torch.float32}


def build_teacher(cfg, device=None) -> Teacher:
    """Build a teacher from a config node ``cfg.teacher`` with fields:
    ``kind`` in {mdlm, dream, llada, gemma}, ``name``, ``dtype``, optional
    ``mask_token``, ``auto_class``, ``attn_implementation``."""
    tc = cfg.teacher
    dtype = _DTYPES.get(str(getattr(tc, "dtype", "bf16")).lower(), torch.bfloat16)
    kind = str(tc.kind).lower()
    if kind == "mdlm":
        return MDLMTeacher(name=tc.name, device=device, dtype=dtype,
                           mask_index=getattr(tc, "mask_index", None))
    common = dict(
        name=tc.name, device=device, dtype=dtype,
        mask_token=getattr(tc, "mask_token", None),
        attn_implementation=getattr(tc, "attn_implementation", None),
    )
    if kind == "dream":
        return DreamTeacher(auto_class=getattr(tc, "auto_class", "AutoModel"), **common)
    if kind == "llada":
        return LLaDATeacher(auto_class=getattr(tc, "auto_class", "AutoModel"), **common)
    if kind == "gemma":
        return GemmaTeacher(auto_class=getattr(tc, "auto_class", "AutoModelForMaskedLM"), **common)
    raise ValueError(f"unknown teacher kind {kind!r}")


__all__ = ["Teacher", "MDLMTeacher", "DreamTeacher", "LLaDATeacher", "GemmaTeacher", "build_teacher"]
