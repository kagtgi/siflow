"""Generative perplexity under a held-out GPT-2-Large scorer.

Gen-PPL measures fluency but rewards degenerate repetition, so always report it
alongside diversity / MAUVE (see ``siflow.eval.diversity``).
"""
from __future__ import annotations

from typing import List

import torch


def decode_ids(ids: torch.Tensor, tokenizer, skip_special: bool = True) -> List[str]:
    """Decode generated token ids (in the *generator's* tokenizer) to text."""
    out = []
    for row in ids.tolist():
        out.append(tokenizer.decode(row, skip_special_tokens=skip_special).strip())
    return out


class GPT2Scorer:
    def __init__(self, name: str = "gpt2-large", device=None, dtype=torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(self.device, dtype=dtype).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def perplexity(self, texts: List[str], batch_size: int = 8, max_len: int = 512) -> dict:
        """Corpus-level generative perplexity (exp of total CE / total tokens)."""
        texts = [t for t in texts if t and t.strip()]
        total_nll, total_tok = 0.0, 0
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids = enc.input_ids.to(self.device)
            attn = enc.attention_mask.to(self.device)
            logits = self.model(ids, attention_mask=attn).logits.float()
            # shift for next-token prediction
            sl = logits[:, :-1]
            tgt = ids[:, 1:]
            m = attn[:, 1:].bool()
            ce = torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), tgt.reshape(-1), reduction="none"
            ).view(tgt.shape)
            total_nll += float((ce * m).sum())
            total_tok += int(m.sum())
        if total_tok == 0:
            return {"gen_ppl": float("nan"), "n_tokens": 0}
        import math

        return {"gen_ppl": math.exp(total_nll / total_tok), "n_tokens": total_tok}
