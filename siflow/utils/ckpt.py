"""Checkpoint save / resume (head + EMA + optimizer + scheduler + step + RNG).

Designed for Colab's 12h timeouts: ``save`` writes both a ``latest.pt`` and a
step-tagged copy; ``resume`` restores everything so a fresh session continues
exactly where the previous one stopped.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save(
    out_dir: str,
    step: int,
    head: torch.nn.Module,
    ema=None,
    optimizer=None,
    scheduler=None,
    cfg=None,
    extra: Optional[Dict[str, Any]] = None,
    keep_tagged: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    blob: Dict[str, Any] = {
        "step": step,
        "head": head.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cfg": None if cfg is None else _to_container(cfg),
        "extra": extra or {},
    }
    latest = os.path.join(out_dir, "latest.pt")
    tmp = latest + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, latest)  # atomic-ish so a crash mid-write can't corrupt latest
    if keep_tagged:
        torch.save(blob, os.path.join(out_dir, f"step_{step:07d}.pt"))
    return latest


def _to_container(cfg):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # noqa: BLE001
        return None


def load(out_dir: str, map_location="cpu") -> Optional[Dict[str, Any]]:
    latest = os.path.join(out_dir, "latest.pt")
    if not os.path.exists(latest):
        return None
    return torch.load(latest, map_location=map_location, weights_only=False)


def resume(out_dir: str, head, ema=None, optimizer=None, scheduler=None, map_location="cpu") -> int:
    """Restore in place; return the step to resume *from* (0 if no checkpoint)."""
    blob = load(out_dir, map_location=map_location)
    if blob is None:
        return 0
    head.load_state_dict(blob["head"])
    if ema is not None and blob.get("ema") is not None:
        ema.load_state_dict(blob["ema"])
    if optimizer is not None and blob.get("optimizer") is not None:
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob.get("scheduler") is not None:
        scheduler.load_state_dict(blob["scheduler"])
    if blob.get("torch_rng") is not None:
        torch.set_rng_state(blob["torch_rng"].cpu() if hasattr(blob["torch_rng"], "cpu") else blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng") is not None:
        try:
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        except Exception:  # noqa: BLE001 - device-count mismatch across sessions
            pass
    return int(blob["step"]) + 1
