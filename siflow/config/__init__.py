"""Configuration loading for SIFLOW (OmegaConf-based).

Usage
-----
    from siflow.config import load_config
    cfg = load_config("siflow/config/mdlm.yaml",
                      overrides=["train.total_steps=5000", "seed=1"])

Each experiment yaml may set ``defaults: default.yaml`` (a sibling file) which is
merged underneath it. CLI-style dotlist overrides are applied last.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from omegaconf import OmegaConf, DictConfig

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # try as given, then relative to this config dir, then basename in this dir
    for cand in (path, os.path.join(_HERE, path), os.path.join(_HERE, os.path.basename(path))):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"config not found: {path}")


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    """Load ``path``, merge its ``defaults`` base if present, apply overrides."""
    path = _resolve(path)
    cfg = OmegaConf.load(path)
    base_name = cfg.pop("defaults", None) if "defaults" in cfg else None
    if base_name:
        base = OmegaConf.load(_resolve(str(base_name)))
        cfg = OmegaConf.merge(base, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def config_hash(cfg: DictConfig) -> str:
    """Short, stable hash of a resolved config (for run provenance)."""
    import hashlib

    blob = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
