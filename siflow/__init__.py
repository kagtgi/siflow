"""SIFLOW: average-velocity distillation of masked diffusion LMs on the simplex.

Submodules are imported lazily so that torch-free utilities (e.g. ``siflow.schedule``)
can be used without importing the whole torch stack.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
