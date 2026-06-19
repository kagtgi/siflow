#!/usr/bin/env python
"""Render paper figures from ``results/*.json`` + analysis dumps into ``paper/figures/``.

Produces (when the underlying data exists):
  pareto.pdf            quality (Gen-PPL / MAUVE) vs throughput (tok/s) / NFE
  straightness.pdf      histogram of path-length ratios (F1)
  entropy.pdf           per-position one-step entropy by variant (F4)
  lambada_vs_k.pdf      LAMBADA accuracy vs refinement budget k (F5)
  training_curves.pdf   SATD / vel / MDM loss over steps (F6)

matplotlib only (Agg), vector PDF, no seaborn/font CDNs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(results_dir) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "schema.example.json":
            continue
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def fig_pareto(results, out):
    plt.figure(figsize=(4.2, 3.2))
    plotted = False
    for r in results:
        xs, ys, labels = [], [], []
        for key, m in r.get("metrics", {}).items():
            if m.get("tok_per_s") and m.get("gen_ppl"):
                xs.append(m["tok_per_s"])
                ys.append(m["gen_ppl"])
                labels.append(key)
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            plt.plot(xs, ys, "o-", label=r.get("method", "?"), markersize=4)
            plotted = True
    plt.xscale("log")
    plt.xlabel("throughput (tok/s, log)")
    plt.ylabel("Gen-PPL $\\downarrow$")
    plt.title("Quality–throughput frontier")
    if plotted:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_straightness(results, out):
    ratios = None
    for r in results:
        st = r.get("analysis", {}).get("straightness")
        if st and st.get("ratios"):
            ratios = st["ratios"]
            mean_r = st.get("ratio_mean")
            break
    if ratios is None:
        return
    plt.figure(figsize=(4.0, 3.0))
    plt.hist(ratios, bins=40, color="#2a824b", alpha=0.85)
    plt.axvline(1.0, color="k", ls="--", lw=1, label="straight (R=1)")
    if mean_r:
        plt.axvline(mean_r, color="#c3372d", lw=1.5, label=f"mean={mean_r:.2f}")
    plt.xlabel("path-length ratio $R$")
    plt.ylabel("count")
    plt.title("Simplex-trajectory straightness")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_entropy(results, out):
    series = []
    for r in results:
        ent = r.get("analysis", {}).get("onestep_entropy")
        if ent and ent.get("entropy"):
            series.append((r.get("run_id", r.get("method", "?")), ent["entropy"]))
    if not series:
        return
    plt.figure(figsize=(4.2, 3.0))
    for label, e in series:
        plt.hist(e, bins=40, alpha=0.5, label=label, density=True)
    plt.xlabel("per-position entropy (nats)")
    plt.ylabel("density")
    plt.title("One-step output entropy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_lambada(results, out):
    plt.figure(figsize=(4.0, 3.0))
    plotted = False
    for r in results:
        if r.get("method") != "SIFLOW":
            continue
        ks, accs = [], []
        for key, m in r.get("metrics", {}).items():
            if key.startswith("k=") and m.get("lambada_acc") is not None:
                ks.append(int(key.split("=")[1]))
                accs.append(100.0 * m["lambada_acc"])
        if ks:
            order = sorted(range(len(ks)), key=lambda i: ks[i])
            plt.plot([ks[i] for i in order], [accs[i] for i in order], "o-", label=r.get("run_id", "SIFLOW"))
            plotted = True
    plt.xlabel("refinement budget $k$ (NFE)")
    plt.ylabel("LAMBADA acc (%)")
    plt.title("Token-dependency recovery")
    if plotted:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_training(train_log, out):
    if not train_log or not os.path.exists(train_log):
        return
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from siflow.analysis.curves import load_jsonl, series

    rows = load_jsonl(train_log)
    plt.figure(figsize=(4.2, 3.0))
    for key in ("satd", "vel", "mdm"):
        xs, ys = series(rows, key)
        if xs:
            plt.plot(xs, ys, label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--train-log", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = _load(args.results)

    fig_pareto(results, os.path.join(args.out_dir, "pareto.pdf"))
    fig_straightness(results, os.path.join(args.out_dir, "straightness.pdf"))
    fig_entropy(results, os.path.join(args.out_dir, "entropy.pdf"))
    fig_lambada(results, os.path.join(args.out_dir, "lambada_vs_k.pdf"))
    fig_training(args.train_log, os.path.join(args.out_dir, "training_curves.pdf"))
    print(f"[make_figures] wrote PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()
