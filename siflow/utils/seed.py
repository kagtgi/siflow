"""Seeding helpers."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_generator(seed: int, device="cpu"):
    """A device-local torch.Generator (for reproducible masking / sampling)."""
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
