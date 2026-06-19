"""MDLM teacher (``kuleshov-group/mdlm-owt``).

Critical integration note
-------------------------
The Hugging Face ``MDLM.forward`` returns **raw backbone logits**. It does NOT
apply the SUBS parameterization (that lives only in the GitHub ``Diffusion``
wrapper). We therefore re-apply SUBS ourselves via ``Teacher.subs``. Getting
this wrong silently trains the head on the wrong target, so ``tests/test_subs``
checks it explicitly.

* tokenizer: plain GPT-2 (no native mask token)
* ``vocab_size = 50258`` -> ``mask_index = 50257`` (the extra, last id)
* ``hidden_dim = 768``, ``time_conditioning = false`` (effectively an x0 predictor)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import Teacher

_DEFAULT = "kuleshov-group/mdlm-owt"


class MDLMTeacher(Teacher):
    def __init__(
        self,
        name: str = _DEFAULT,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        mask_index: Optional[int] = None,
    ):
        from transformers import AutoModelForMaskedLM

        self.name = name
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
        self.model.eval().requires_grad_(False).to(self.device, dtype=dtype)

        self.vocab_size = int(self.model.config.vocab_size)
        self.hidden_dim = int(getattr(self.model.config, "hidden_dim", 768))
        self.mask_index = self.vocab_size - 1 if mask_index is None else int(mask_index)

        emb = self._find_embedding()
        assert emb.shape[0] == self.vocab_size, (
            f"embedding rows {emb.shape[0]} != vocab {self.vocab_size}")
        self._embedding = emb

    # -- embedding lookup (robust across MDLM revisions) ----------------------
    def _find_embedding(self) -> torch.Tensor:
        try:
            mod = self.model.get_input_embeddings()
            if mod is not None and hasattr(mod, "weight"):
                return mod.weight.detach()
        except (AttributeError, NotImplementedError):
            pass
        # fall back to common attribute paths in the DiT backbone
        for path in ("backbone.vocab_embed.embedding.weight",
                     "backbone.vocab_embed.weight",
                     "vocab_embed.embedding.weight"):
            obj = self.model
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, torch.Tensor):
                return obj.detach()
        raise RuntimeError("could not locate MDLM input embedding matrix")

    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self._embedding

    @torch.no_grad()
    def logits_and_hidden(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        try:
            out = self.model(input_ids, output_hidden_states=True, return_dict=True)
            raw, hiddens = out.logits, out.hidden_states
        except TypeError:
            out = self.model(input_ids, output_hidden_states=True)
            raw, hiddens = (out[0], out[1]) if isinstance(out, tuple) else (out.logits, out.hidden_states)
        if hiddens is None:
            raise RuntimeError("MDLM did not return hidden states; needs output_hidden_states=True")
        hidden = hiddens[-1]
        subs = self.subs(raw.float(), input_ids)
        return subs, hidden
