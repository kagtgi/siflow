"""LAMBADA zero-shot last-word accuracy for masked DLMs.

LAMBADA tests broad-discourse coherence: predict the final word of a passage.
A masked DLM scores it by revealing the context, masking only the target word's
token span, and reading out the student's prediction. Accuracy as a function of
the refinement budget ``k`` is the paper's direct evidence that self-conditioned
refinement repairs jointly inconsistent one-step samples.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _load_lambada(max_examples: Optional[int]):
    from datasets import load_dataset

    for name, conf in (("lambada", None), ("EleutherAI/lambada_openai", "en"), ("cimec/lambada", None)):
        try:
            ds = load_dataset(name, conf, split="test") if conf else load_dataset(name, split="test")
            texts = [ex["text"] for ex in ds]
            return texts[:max_examples] if max_examples else texts
        except Exception:  # noqa: BLE001 - try the next mirror
            continue
    raise RuntimeError("could not load any LAMBADA dataset variant")


@torch.no_grad()
def lambada_accuracy(
    student,
    tokenizer,
    k_list: List[int] = (1, 2, 4, 8),
    device=None,
    max_examples: int = 500,
    batch_size: int = 16,
    max_len: int = 256,
    complete_fn=None,
) -> Dict[str, float]:
    """Return ``{f"lambada_acc@k{k}": acc}`` for each ``k``.

    ``complete_fn(input_ids, fill_mask, k) -> filled_ids`` defaults to
    ``student.complete``; pass a teacher completer to score the teacher baseline.
    """
    device = device or student.teacher.device
    mask_id = student.teacher.mask_index
    if complete_fn is None:
        complete_fn = lambda ids, fill, k: student.complete(ids, fill, k=k)  # noqa: E731
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    texts = _load_lambada(max_examples)

    # pre-tokenize: context ids + target last-word token span
    items = []
    for t in texts:
        t = t.strip()
        if " " not in t:
            continue
        prefix, last = t.rsplit(" ", 1)
        ctx = tokenizer(prefix + " ", add_special_tokens=False)["input_ids"]
        tgt = tokenizer(last, add_special_tokens=False)["input_ids"]
        if not tgt or len(ctx) + len(tgt) > max_len:
            continue
        items.append((ctx, tgt))

    results: Dict[str, float] = {}
    for k in k_list:
        correct = 0
        total = 0
        for i in range(0, len(items), batch_size):
            chunk = items[i: i + batch_size]
            L = max(len(c) + len(g) for c, g in chunk)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            fill = torch.zeros((len(chunk), L), dtype=torch.bool)
            for r, (ctx, tgt) in enumerate(chunk):
                seq = ctx + tgt
                ids[r, : len(seq)] = torch.tensor(seq)
                fill[r, len(ctx): len(ctx) + len(tgt)] = True  # mask only the target word
            ids = ids.to(device)
            fill = fill.to(device)
            out = complete_fn(ids, fill, k)
            for r, (ctx, tgt) in enumerate(chunk):
                pred = out[r, len(ctx): len(ctx) + len(tgt)].tolist()
                correct += int(pred == list(tgt))
                total += 1
        results[f"lambada_acc@k{k}"] = correct / max(total, 1)
    return results
