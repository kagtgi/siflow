"""Generic Hugging Face masked-DLM teacher (shared by Dream-7B and LLaDA-8B).

Both are masked-token predictors: a single forward on partially-masked ids yields
per-position logits over the vocabulary. We apply the same SUBS parameterization
as for MDLM (mask column -> -inf, revealed positions pinned one-hot) using the
teacher's own mask-token id, and -- for these large vocabularies -- regress on a
reduced top-m support during training (see ``siflow/support.py``).

The teacher runs *live* on a single A100-40GB alongside the tiny velocity head;
no offline cache is required. Because the exact public class / mask token of these
releases may shift, loading is defensive: we try several ``Auto*`` classes and
several places the mask id might live (raising a clear error so the user can set
``auto_class`` / ``mask_token`` in the config), and the final hidden state the
velocity head needs is captured robustly -- via ``output_hidden_states`` when the
forward supports it, else via a forward hook on the un-embed (lm_head) module.
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

        from ..flash_compat import ensure_flash_attn
        ensure_flash_attn()  # some remote backbones import flash_attn with no fallback

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

        # Robust hidden-state capture. Some custom masked-DLM forwards (e.g. LLaDA)
        # do not honour ``output_hidden_states`` and only return ``.logits``. The
        # velocity head needs the final hidden state, so we also register a forward
        # hook on the output-embedding (un-embed / lm_head) module and grab its
        # input -- which IS the final-layer hidden state [B, L, H]. Whichever path
        # produces a tensor first wins in ``logits_and_hidden``.
        self._hidden_cache: dict = {}
        self._hook_handle = None
        try:
            head_mod = self.model.get_output_embeddings()
        except Exception:  # noqa: BLE001 - not all models expose this
            head_mod = None
        if head_mod is not None:
            def _grab_hidden(_m, inp, _out):
                if inp and torch.is_tensor(inp[0]):
                    self._hidden_cache["h"] = inp[0].detach()
            self._hook_handle = head_mod.register_forward_hook(_grab_hidden)

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
        self._hidden_cache.pop("h", None)
        try:
            out = self.model(input_ids, output_hidden_states=True, return_dict=True)
        except TypeError:
            # custom forward that does not accept output_hidden_states/return_dict
            out = self.model(input_ids)
        raw = getattr(out, "logits", None)
        if raw is None and isinstance(out, (tuple, list)) and len(out):
            raw = out[0]
        if raw is None:
            raise RuntimeError(
                f"{self.name} ({self._auto_class}) returned no logits; "
                "set auto_class to a *ForMaskedLM/ForCausalLM variant in the config")
        hs = getattr(out, "hidden_states", None)
        hidden = hs[-1] if hs is not None else self._hidden_cache.get("h")
        if hidden is None:
            raise RuntimeError(
                f"{self.name}: could not obtain a hidden state (no output_hidden_states "
                "and no un-embed input captured); the velocity head needs it")
        subs = self.subs(raw.float(), input_ids)
        return subs, hidden
