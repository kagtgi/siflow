"""Google Drive / output-path helpers.

On Colab, call ``mount()`` once; elsewhere these are no-ops that just resolve a
local base directory. All long-running notebooks write checkpoints, caches and
results under ``base_dir()`` so a timed-out session can resume from Drive.
"""
from __future__ import annotations

import os
from typing import Optional

_DEFAULT_COLAB = "/content/drive/MyDrive/siflow"
_ENV = "SIFLOW_BASE"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def mount(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive if running on Colab. Returns True if mounted."""
    if not in_colab():
        return False
    from google.colab import drive as _gd  # type: ignore

    _gd.mount(mount_point)
    return True


def base_dir(override: Optional[str] = None) -> str:
    """Resolve the artifact base dir: explicit override > $SIFLOW_BASE >
    Colab Drive default > ./runs."""
    base = override or os.environ.get(_ENV)
    if base is None:
        base = _DEFAULT_COLAB if in_colab() else os.path.abspath("runs")
    os.makedirs(base, exist_ok=True)
    return base


def path(*parts: str, override: Optional[str] = None) -> str:
    p = os.path.join(base_dir(override), *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
