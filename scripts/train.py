#!/usr/bin/env python
"""Train a SIFLOW velocity head.

    python scripts/train.py --config siflow/config/mdlm.yaml \
        --set data.tokens_path=runs/data/owt_gpt2_256.npy train.total_steps=20000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siflow.config import load_config  # noqa: E402
from siflow.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="OmegaConf dotlist overrides, e.g. seed=1")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    out = train(cfg)
    print(f"[train] checkpoints in: {out}")


if __name__ == "__main__":
    main()
