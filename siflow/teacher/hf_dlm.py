"""Generic Hugging Face masked-DLM teacher (shared by Dream / DiffusionGemma).

These are large teachers used only by the *offline cache builder* (run_5 / run_7);
training (run_6 / run_8) reads the cache and never reloads the backbone.

Both Dream-7B and DiffusionGemma are masked-token predictors: a single forward
on partially-masked ids yields per-position logits over the vocabulary. We apply
the same SUBS parameterization as for MDLM (mask column -> -inf, revealed
positions pinned one-hot) using the teacher's own mask-token id.

Because the exact public class / mask token of these 2026 releases may shift,
loading is defensive: we try several ``Auto*`` classes and several places the
mask id might live, and raise a clear error if none work so the user can set
``auto_class`` / ``mask_token`` in the config.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import Teacher


class HFMaskedDLMTeacher(Teacher):
    #: ordered Auto* classes to try when loading
    _AUTO_CLASSES = ("AutoModelForMaskedLM", "AutoModelForCausalLM", "AutoModel")

    def __init__(
        self,
        name: str,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        mask_token: Optional[str | int] = None,
        auto_class: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ):
        import transformers
        from transformers import AutoTokenizer

        self.name = name
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

        classes = [auto_class] if auto_class else list(self._AUTO_CLASSES)
        last_err: Optional[Exception] = None
        self.model = None
        for cls_name in classes:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                kw = dict(trust_remote_code=True, torch_dtype=dtype)
                if attn_implementation:
                    kw["attn_implementation"] = attn_implementation
                self.model = cls.from_pretrained(name, **kw)
                self._auto_class = cls_name
                break
            except Exception as e:  # noqa: BLE001 - report after exhausting options
                last_err = e
        if self.model is None:
            raise RuntimeError(f"could not load {name} via {classes}: {last_err}")

        self.model.eval().requires_grad_(False).to(self.device)
        self.vocab_size = int(self.model.config.vocab_size)
        self.hidden_dim = int(getattr(self.model.config, "hidden_size",
                                      getattr(self.model.config, "hidden_dim", 0)))
        emb = self.model.get_input_embeddings().weight.detach()
        if self.hidden_dim == 0:
            self.hidden_dim = int(emb.shape[1])
        self._embedding = emb
        self.mask_index = self._resolve_mask_index(mask_token)

    def _resolve_mask_index(self, mask_token: Optional[str | int]) -> int:
        if isinstance(mask_token, int):
            return mask_token
        if isinstance(mask_token, str):
            mid = self.tokenizer.convert_tokens_to_ids(mask_token)
            if mid is not None and mid >= 0:
                return int(mid)
        for attr in ("mask_token_id",):
            mid = getattr(self.tokenizer, attr, None)
            if mid is not None:
                return int(mid)
        mid = getattr(self.model.config, "mask_token_id", None)
        if mid is not None:
            return int(mid)
        raise RuntimeError(
            f"{self.name}: no mask token found; pass mask_token=<str|int> in the config")

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._embedding

    @torch.no_grad()
    def logits_and_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids, output_hidden_states=True, return_dict=True)
        raw = getattr(out, "logits", None)
        if raw is None:
            raise RuntimeError(
                f"{self.name} ({self._auto_class}) returned no logits; "
                "set auto_class to a *ForMaskedLM/ForCausalLM variant in the config")
        hidden = out.hidden_states[-1]
        subs = self.subs(raw.float(), input_ids)
        return subs, hidden
