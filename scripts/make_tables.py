#!/usr/bin/env python
"""Turn ``results/*.json`` into ``paper/tables_auto.tex`` (LaTeX row macros).

The paper ``\\input``s this file and uses the macros inside table environments,
so re-running this after experiments fills Tables 2-4 with no hand-editing.
Rows whose numbers come from a cited paper (not reproduced here) are emitted with
a dagger ``$^\\dagger$`` and left as ``--`` for the user to transcribe.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Optional

# Reported-only baselines (numbers come from the cited papers; marked with a dagger).
REPORTED = [
    ("T3D$^\\dagger$~\\citep{t3d2026}", "4--8"),
    ("IMDM$^\\dagger$~\\citep{yoo2026imdm}", "4--8"),
    ("FMLM$^\\dagger$~\\citep{lee2026fmlm}", "1"),
    ("DLM-One$^\\dagger$~\\citep{chen2025dlmone}", "1"),
]


def _fmt(vals: List[Optional[float]], prec=2, pct=False) -> str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "--"
    scale = 100.0 if pct else 1.0
    mu = mean(vals) * scale
    if len(vals) > 1:
        return f"{mu:.{prec}f}$\\pm${pstdev(vals) * scale:.{prec}f}"
    return f"{mu:.{prec}f}"


def load_results(results_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "schema.example.json":
            continue
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _collect(results, method_filter, exclude_ablations=False):
    """Group metric dicts by (method, key) -> per-metric list across seeds/files."""
    by = defaultdict(lambda: defaultdict(list))
    for r in results:
        if method_filter and method_filter not in r.get("method", ""):
            continue
        if exclude_ablations and str(r.get("run_id", "")).startswith("abl_"):
            continue  # ablation runs share method "SIFLOW" but belong in Table 3, not Table 2
        for key, m in r.get("metrics", {}).items():
            for mk, mv in m.items():
                by[(r["method"], key)][mk].append(mv)
    return by


def _step_key(key):
    """Numeric value from a 'k=4' / 'steps=64' metric key (for sorting)."""
    tail = str(key).split("=")[-1]
    try:
        return float(tail)
    except ValueError:
        return float("inf")


def _row(label, steps, by, key, cells=("gen_ppl", "mauve", "lambada_acc", "tok_per_s")):
    def cell(mk):
        vals = by.get(key, {}).get(mk, [])
        if mk == "mauve":
            return _fmt(vals, prec=3)
        if mk == "lambada_acc":
            return _fmt(vals, prec=1, pct=True)
        if mk == "tok_per_s":
            return _fmt(vals, prec=0)
        return _fmt(vals, prec=2)
    return f"{label} & {steps} & " + " & ".join(cell(c) for c in cells) + r" \\"


def main_rows(results) -> str:
    rows = []
    # AR + teacher curve + SDTT (reproduced)
    ar = _collect(results, "AR-")
    for (method, key), by_metric in ar.items():
        rows.append(_row(method.replace("AR-", "AR GPT-2 "), "$L$", {key: by_metric}, key))
    teach = _collect(results, "teacher")
    for (method, key), by_metric in sorted(teach.items(), key=lambda kv: -_step_key(kv[0][1])):
        steps = key.split("=")[-1]
        rows.append(_row("MDLM teacher~\\citep{sahoo2024mdlm}", steps, {key: by_metric}, key))
    sdtt = _collect(results, "SDTT")
    for (method, key), by_metric in sdtt.items():
        rows.append(_row("SDTT~\\citep{deschenaux2025sdtt}", key.split("=")[-1], {key: by_metric}, key))
    # reported baselines
    for label, steps in REPORTED:
        rows.append(f"{label} & {steps} & -- & -- & -- & -- " + r"\\")
    # SIFLOW main rows (ablation runs excluded -> they go to Table 3)
    sf = _collect(results, "SIFLOW", exclude_ablations=True)
    for (method, key), by_metric in sorted(sf.items(), key=lambda kv: _step_key(kv[0][1])):
        if method != "SIFLOW":
            continue
        rows.append(_row("\\textbf{\\method{} (ours)}", key.split("=")[-1], {key: by_metric}, key))
    return "\n".join(rows) if rows else "-- & -- & -- & -- & -- & -- \\\\"


def ablation_rows(results) -> str:
    """Ablation rows pulled from runs whose run_id starts with 'abl_'."""
    rows = []
    for r in results:
        rid = r.get("run_id", "")
        if not rid.startswith("abl_"):
            continue
        label = rid.replace("abl_", "").replace("_", " ")
        m = r.get("metrics", {}).get("k=1", {})
        ppl = _fmt([m.get("gen_ppl")])
        mauve = _fmt([m.get("mauve")], prec=3)
        lam = _fmt([m.get("lambada_acc")], prec=1, pct=True)
        tps = _fmt([m.get("tok_per_s")], prec=0)
        rows.append(f"\\quad {label} & {ppl} & {mauve} & {lam} & {tps} " + r"\\")
    return "\n".join(rows) if rows else "\\quad (run run\\_4 to populate) & -- & -- & -- & -- \\\\"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables_auto.tex")
    args = ap.parse_args()
    results = load_results(args.results)

    body = []
    body.append("% AUTO-GENERATED by scripts/make_tables.py -- do not edit by hand.")
    body.append(r"\newcommand{\SiFlowMainRows}{%")
    body.append(main_rows(results))
    body.append("}")
    body.append(r"\newcommand{\SiFlowAblationRows}{%")
    body.append(ablation_rows(results))
    body.append("}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(body) + "\n")
    print(f"[make_tables] wrote {args.out} from {len(results)} result files")


if __name__ == "__main__":
    main()
