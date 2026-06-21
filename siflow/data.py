"""Corpus tokenization and the live (cache-free) token dataset.

``build_token_chunks`` streams a HF text dataset, tokenizes it, packs the stream
into non-overlapping length-``L`` chunks and writes a compact ``uint16`` ``.npy``
(GPT-2 / most masked-DLM vocabularies fit in 16 bits). The MDLM teacher is cheap
enough to run live, so primary training reads these clean chunks and masks them
on the fly -- no giant simplex cache is needed for the primary study.
"""
from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np

# Streamed in order until one works. OpenWebText is preferred (the paper's corpus)
# but it is a *script* dataset, which modern `datasets` (>=3.0, common on Colab)
# refuses unless trust_remote_code is set -- and may drop entirely. So we fall back
# to large, script-free, no-auth Parquet corpora. (path, config_name, text_key)
_TEXT_DATASETS = [
    ("Skylion007/openwebtext", None, "text"),         # true OWT (needs a script)
    ("HuggingFaceFW/fineweb", "sample-10BT", "text"),  # reliable large Parquet stream
    ("allenai/c4", "en", "text"),
    ("stas/openwebtext-10k", None, "text"),            # tiny OWT Parquet (last resort)
]


def _id_dtype(tokenizer):
    """Smallest uint that holds this tokenizer's ids. GPT-2 (~50k) fits uint16;
    Qwen/Dream (~152k) and LLaMA/LLaDA (~126k) do NOT -> uint32, or their ids would
    silently overflow/wrap. Unknown size -> uint32 (safe)."""
    nvocab = 0
    try:
        nvocab = len(tokenizer)
    except TypeError:
        nvocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    return np.uint16 if 0 < nvocab <= 65536 else np.uint32


def _open_text_stream(preferred: Optional[str], split: str, streaming: bool):
    """Return ``(iterator_of_examples, text_key, source_name)`` for the first text
    dataset that actually streams. Tries the preferred dataset (with and without
    ``trust_remote_code``), then the script-free fallbacks."""
    from datasets import load_dataset

    cands = []
    seen = set()
    for path, name, key in ([(preferred, None, "text")] if preferred else []) + _TEXT_DATASETS:
        if path and path not in seen:
            seen.add(path)
            cands.append((path, name, key))
    last_err: Optional[Exception] = None
    for path, name, key in cands:
        for trust in (False, True):
            try:
                kw = {"split": split, "streaming": streaming}
                if name:
                    kw["name"] = name
                if trust:
                    kw["trust_remote_code"] = True
                ds = load_dataset(path, **kw)
                it = iter(ds)
                first = next(it)                       # force a real fetch (validates access)
                if key not in first:                   # find any string field
                    key = next((k for k, v in first.items() if isinstance(v, str)), key)
                tag = f"{path}" + (f":{name}" if name else "")
                print(f"[data] streaming corpus: {tag}")
                return itertools.chain([first], it), key
            except Exception as e:  # noqa: BLE001 - try the next dataset / trust flag
                last_err = e
    raise RuntimeError(f"could not stream any text dataset; last error: {last_err}")


def build_token_chunks(
    tokenizer,
    seq_len: int,
    target_seqs: int,
    out_path: str,
    dataset: str = "Skylion007/openwebtext",
    split: str = "train",
    text_key: str = "text",
    streaming: bool = True,
    add_eos: bool = True,
    skip_seqs: int = 0,
) -> int:
    """Stream + tokenize + pack into ``[N, seq_len]`` uint16; save to ``out_path``.

    ``skip_seqs`` discards the first N packed chunks before collecting -- use it to
    carve a disjoint validation set from the same stream (e.g. train uses
    ``skip_seqs=0``, val uses ``skip_seqs=<train size>``). Falls back across several
    public corpora so it runs on any recent ``datasets`` without auth.

    Returns the number of chunks written (``N``)."""
    eos = tokenizer.eos_token_id if add_eos and tokenizer.eos_token_id is not None else None
    idt = _id_dtype(tokenizer)  # uint16 (GPT-2) vs uint32 (Dream/LLaDA) -- avoid id overflow

    stream, text_key = _open_text_stream(dataset, split, streaming)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    buf: list[int] = []
    chunks: list[np.ndarray] = []
    n = 0
    skipped = 0
    for ex in stream:
        text = ex.get(text_key) or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len:
            chunk = np.asarray(buf[:seq_len], dtype=idt)
            del buf[:seq_len]
            if skipped < skip_seqs:
                skipped += 1
                continue
            chunks.append(chunk)
            n += 1
            if n >= target_seqs:
                break
        if n >= target_seqs:
            break

    arr = np.stack(chunks) if chunks else np.zeros((0, seq_len), dtype=idt)
    if arr.shape[0] == 0:
        raise RuntimeError(
            f"no token chunks produced for {out_path}: every corpus stream yielded no "
            f"usable text (skip_seqs={skip_seqs}). Check network access / lower skip_seqs.")
    np.save(out_path, arr)
    return int(arr.shape[0])


class TokenChunkDataset:
    """Memory-mapped ``[N, L]`` uint16 token chunks; ``__getitem__`` -> long [L]."""

    def __init__(self, npy_path: str):
        self.path = npy_path
        self.data = np.load(npy_path, mmap_mode="r")
        assert self.data.ndim == 2, f"expected [N, L], got {self.data.shape}"

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.data.shape[1])

    def __getitem__(self, i: int):
        import torch

        return torch.as_tensor(np.asarray(self.data[i], dtype=np.int64))

    def batch(self, idx, device=None):
        import torch

        rows = np.asarray(self.data[np.asarray(idx)], dtype=np.int64)
        t = torch.as_tensor(rows)
        return t.to(device) if device is not None else t


def infinite_batches(dataset: "TokenChunkDataset", batch_size: int, seed: int = 0, device=None):
    """Yield random batches forever (reshuffling each epoch)."""
    rng = np.random.default_rng(seed)
    N = len(dataset)
    order = rng.permutation(N)
    pos = 0
    while True:
        if pos + batch_size > N:
            order = rng.permutation(N)
            pos = 0
        idx = order[pos: pos + batch_size]
        pos += batch_size
        yield dataset.batch(idx, device=device)
