"""Lightweight JSONL + console logging (no W&B dependency required)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional


def log(msg: str) -> None:
    print(f"[siflow] {msg}", file=sys.stderr, flush=True)


class JsonlLogger:
    """Append structured records to a ``.jsonl`` file and echo a short line."""

    def __init__(self, path: str, echo_every: int = 1):
        self.path = path
        self.echo_every = max(1, echo_every)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._n = 0
        self._t0 = time.time()

    def write(self, record: Dict[str, Any], echo_keys: Optional[list] = None) -> None:
        record = {"wall_s": round(time.time() - self._t0, 2), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._n += 1
        if self._n % self.echo_every == 0:
            keys = echo_keys or list(record.keys())
            parts = []
            for k in keys:
                v = record.get(k)
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            log(" ".join(parts))
