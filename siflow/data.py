"""Corpus tokenization and the live (cache-free) token dataset.

``build_token_chunks`` streams a HF text dataset, tokenizes it, packs the stream
into non-overlapping length-``L`` chunks and writes a compact ``uint16`` ``.npy``
(GPT-2 / most masked-DLM vocabularies fit in 16 bits). The MDLM teacher is cheap
enough to run live, so primary training reads these clean chunks and masks them
on the fly -- no giant simplex cache is needed for the primary study.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


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
    ``skip_seqs=0``, val uses ``skip_seqs=<train size>``).

    Returns the number of chunks written (``N``)."""
    from datasets import load_dataset

    eos = tokenizer.eos_token_id if add_eos and tokenizer.eos_token_id is not None else None

    ds = load_dataset(dataset, split=split, streaming=streaming)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    buf: list[int] = []
    chunks: list[np.ndarray] = []
    n = 0
    skipped = 0
    for ex in ds:
        text = ex.get(text_key) or ""
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len:
            chunk = np.asarray(buf[:seq_len], dtype=np.uint16)
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

    arr = np.stack(chunks) if chunks else np.zeros((0, seq_len), dtype=np.uint16)
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
