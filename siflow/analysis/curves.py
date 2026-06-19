"""Training-curve helpers (parse train_log.jsonl for Figure F6)."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(records: List[Dict], key: str, x: str = "step") -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in records:
        if key in r and x in r:
            xs.append(r[x])
            ys.append(r[key])
    return xs, ys
