#!/usr/bin/env python
"""Build the (optional) reduced-support simplex cache for Dream / Gemma.

    python scripts/build_cache.py --config siflow/config/dream.yaml \
        --tokens runs/data/dream_256.npy --out runs/cache/dream --n 50000 --m 128

Resumable: re-running skips shards already written, so a 12h timeout is recovered
by simply re-running the same command/cell.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siflow.config import load_config  # noqa: E402
from siflow.schedule import NoiseSchedule  # noqa: E402
from siflow.teacher import build_teacher  # noqa: E402
from siflow.data import TokenChunkDataset  # noqa: E402
from siflow.cache import build_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", required=True, help="[N,L] token chunks in the TEACHER tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    teacher = build_teacher(cfg)
    ds = TokenChunkDataset(args.tokens)
    sched = NoiseSchedule(kind=cfg.schedule.kind, eps=float(cfg.schedule.eps))
    man = build_cache(
        teacher, ds, args.out, n_examples=args.n, m=args.m, schedule=sched,
        p0=float(cfg.train.p0), p1=float(cfg.train.p1), lam_ent=float(cfg.train.lam_ent),
        batch_size=args.batch, shard_size=args.shard_size, seed=int(cfg.seed),
    )
    print(f"[build_cache] manifest: {man}")


if __name__ == "__main__":
    main()
