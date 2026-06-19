"""Diversity / repetition metrics (detect entropy & mode collapse)."""
from __future__ import annotations

from typing import Dict, List


def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def diversity_metrics(texts: List[str], self_bleu: bool = False, self_bleu_sample: int = 200) -> Dict[str, float]:
    """Corpus distinct-n (n=1..4), mean within-sample 4-gram repetition, and
    (optionally) self-BLEU. Whitespace tokenization keeps it tokenizer-agnostic."""
    out: Dict[str, float] = {}
    toks = [t.split() for t in texts if t and t.strip()]

    for n in (1, 2, 3, 4):
        seen, tot = set(), 0
        for tk in toks:
            grams = _ngrams(tk, n)
            seen.update(grams)
            tot += len(grams)
        out[f"distinct_{n}"] = len(seen) / max(tot, 1)

    reps = []
    for tk in toks:
        g = _ngrams(tk, 4)
        if g:
            reps.append(1.0 - len(set(g)) / len(g))
    out["rep_4"] = sum(reps) / max(len(reps), 1)

    if self_bleu and len(toks) >= 2:
        try:
            import sacrebleu

            sample = [" ".join(tk) for tk in toks[:self_bleu_sample]]
            scores = []
            for i, hyp in enumerate(sample):
                refs = sample[:i] + sample[i + 1:]
                scores.append(sacrebleu.sentence_bleu(hyp, refs).score)
            out["self_bleu"] = sum(scores) / len(scores)
        except Exception:  # noqa: BLE001
            pass
    return out
