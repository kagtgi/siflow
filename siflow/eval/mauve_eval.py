"""MAUVE: distributional gap between generated and human (reference) text."""
from __future__ import annotations

from typing import List, Optional


def compute_mauve(
    gen_texts: List[str],
    ref_texts: List[str],
    featurize_model_name: str = "gpt2-large",
    max_text_length: int = 256,
    device_id: Optional[int] = 0,
    verbose: bool = False,
) -> float:
    """Return the MAUVE score (higher is better). Requires ``mauve-text``."""
    import mauve

    gen_texts = [t for t in gen_texts if t and t.strip()]
    ref_texts = [t for t in ref_texts if t and t.strip()]
    n = min(len(gen_texts), len(ref_texts))
    out = mauve.compute_mauve(
        p_text=gen_texts[:n],
        q_text=ref_texts[:n],
        featurize_model_name=featurize_model_name,
        max_text_length=max_text_length,
        device_id=device_id if device_id is not None else -1,
        verbose=verbose,
    )
    return float(out.mauve)
