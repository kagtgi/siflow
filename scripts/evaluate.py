#!/usr/bin/env python
"""Evaluate one system and write a results JSON (schema: results/schema.example.json).

Examples
--------
SIFLOW student (k sweep + analyses):
    python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \
        --ref-tokens runs/data/owt_gpt2_val.npy --n-samples 1000 --out results/mdlm.json

Teacher step-curve baseline:
    python scripts/evaluate.py --config siflow/config/mdlm.yaml --system teacher \
        --steps 8 32 64 1024 --ref-tokens runs/data/owt_gpt2_val.npy --out results/mdlm_teacher.json

AR GPT-2 reference:
    python scripts/evaluate.py --config siflow/config/mdlm.yaml --system ar \
        --ar-model gpt2 --ref-tokens runs/data/owt_gpt2_val.npy --out results/ar_gpt2.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from siflow.config import load_config, config_hash  # noqa: E402
from siflow.utils import drive, ckpt as ckpt_io  # noqa: E402
from siflow.schedule import NoiseSchedule  # noqa: E402


# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _tokenizer(cfg, teacher=None):
    from transformers import AutoTokenizer

    if str(cfg.teacher.kind).lower() == "mdlm":
        return AutoTokenizer.from_pretrained("gpt2")
    if teacher is not None and getattr(teacher, "tokenizer", None) is not None:
        return teacher.tokenizer
    return AutoTokenizer.from_pretrained(cfg.teacher.name, trust_remote_code=True)


def _ref_texts(path, tokenizer, n):
    import numpy as np

    if not path or not os.path.exists(path):
        return []
    arr = np.load(path, mmap_mode="r")
    rows = np.asarray(arr[: n], dtype=np.int64).tolist()
    return [tokenizer.decode(r, skip_special_tokens=True).strip() for r in rows]


def _skeleton(cfg, method, source="reproduced"):
    return {
        "run_id": str(cfg.run_id), "method": method, "teacher": str(cfg.teacher.name),
        "tokenizer": "gpt2" if str(cfg.teacher.kind).lower() == "mdlm" else str(cfg.teacher.name),
        "source": source, "cite": None, "seed": int(cfg.seed),
        "git_sha": _git_sha(), "config_hash": config_hash(cfg),
        "metrics": {}, "analysis": {}, "train": {},
    }


def _quality_block(texts, ref_texts, scorer, no_mauve, device_id):
    from siflow.eval.diversity import diversity_metrics

    block = {}
    block.update(scorer.perplexity(texts))
    block.update(diversity_metrics(texts))
    if not no_mauve and ref_texts:
        try:
            from siflow.eval.mauve_eval import compute_mauve

            block["mauve"] = compute_mauve(texts, ref_texts, device_id=device_id)
        except Exception as e:  # noqa: BLE001
            block["mauve"] = None
            block["mauve_error"] = str(e)
    return block


# --------------------------------------------------------------------------- #
def eval_siflow(cfg, args, device):
    from siflow.train import build_student
    from siflow.teacher import build_teacher
    from siflow.eval import decode_ids
    from siflow.eval.gen_ppl import GPT2Scorer
    from siflow.eval.lambada import lambada_accuracy
    from siflow.eval.throughput import student_throughput
    from siflow.analysis import path_length_ratio, onestep_entropy
    from siflow.data import TokenChunkDataset

    teacher = build_teacher(cfg, device=device)
    student = build_student(cfg, device, teacher=teacher)
    out_dir = args.ckpt_dir or (cfg.out_dir or drive.path("ckpt", str(cfg.run_id)))
    blob = ckpt_io.load(out_dir, map_location=device)
    assert blob is not None, f"no checkpoint in {out_dir}"
    state = blob.get("ema") or blob.get("head")
    student.head.load_state_dict(state)
    student.head.eval()

    tok = _tokenizer(cfg, teacher)
    ref = _ref_texts(args.ref_tokens, tok, args.n_samples)
    scorer = GPT2Scorer(args.scorer, device=device)
    res = _skeleton(cfg, method="SIFLOW")

    L = int(cfg.data.seq_len)
    for k in args.k_list:
        texts = []
        bs = args.gen_batch
        while len(texts) < args.n_samples:
            ids = student.generate(min(bs, args.n_samples - len(texts)), L, k=k, sample=args.sample)
            texts.extend(decode_ids(ids, tok))
        m = _quality_block(texts, ref, scorer, args.no_mauve, args.device_id)
        m.update(student_throughput(student, L, k, batch_size=args.gen_batch))
        res["metrics"][f"k={k}"] = m

    # LAMBADA across the same k sweep (single call)
    lam = lambada_accuracy(student, tok, k_list=tuple(args.k_list), device=device,
                           max_examples=args.max_lambada)
    for k in args.k_list:
        res["metrics"][f"k={k}"]["lambada_acc"] = lam.get(f"lambada_acc@k{k}")

    # analyses
    if args.ref_tokens and os.path.exists(args.ref_tokens):
        ds = TokenChunkDataset(args.ref_tokens)
        sched = NoiseSchedule(kind=cfg.schedule.kind, eps=float(cfg.schedule.eps))
        res["analysis"]["straightness"] = path_length_ratio(
            teacher, ds, schedule=sched, n_examples=args.straightness_n, device=device)
        res["analysis"]["onestep_entropy"] = onestep_entropy(student, ds, schedule=sched,
                                                             n_examples=args.straightness_n, device=device)
    if blob.get("step") is not None:
        res["train"]["total_steps"] = int(blob["step"]) + 1
    return res


def eval_teacher(cfg, args, device):
    from siflow.teacher import build_teacher
    from siflow.sampling import teacher_ancestral_sample, teacher_complete
    from siflow.eval import decode_ids
    from siflow.eval.gen_ppl import GPT2Scorer
    from siflow.eval.lambada import lambada_accuracy
    from siflow.eval.throughput import teacher_throughput

    teacher = build_teacher(cfg, device=device)
    tok = _tokenizer(cfg, teacher)
    ref = _ref_texts(args.ref_tokens, tok, args.n_samples)
    scorer = GPT2Scorer(args.scorer, device=device)
    sched = NoiseSchedule(kind=cfg.schedule.kind, eps=float(cfg.schedule.eps))
    res = _skeleton(cfg, method="teacher")

    L = int(cfg.data.seq_len)
    for steps in args.steps:
        texts = []
        while len(texts) < args.n_samples:
            n = min(args.gen_batch, args.n_samples - len(texts))
            ids = teacher_ancestral_sample(teacher, n, L, steps, schedule=sched)
            texts.extend(decode_ids(ids, tok))
        m = _quality_block(texts, ref, scorer, args.no_mauve, args.device_id)
        m.update(teacher_throughput(teacher, teacher_ancestral_sample, L, steps,
                                    batch_size=args.gen_batch, schedule=sched))
        lam = lambada_accuracy(teacher, tok, k_list=(steps,), device=device, max_examples=args.max_lambada,
                               complete_fn=lambda ids, fill, k, _t=teacher: teacher_complete(_t, ids, fill, k))
        m["lambada_acc"] = lam.get(f"lambada_acc@k{steps}")
        res["metrics"][f"steps={steps}"] = m
    return res


def eval_ar(cfg, args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from siflow.eval.gen_ppl import GPT2Scorer
    from siflow.eval.diversity import diversity_metrics

    name = args.ar_model
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    ref = _ref_texts(args.ref_tokens, tok, args.n_samples)
    scorer = GPT2Scorer(args.scorer, device=device)
    res = _skeleton(cfg, method=f"AR-{name}")
    L = int(cfg.data.seq_len)

    texts = []
    while len(texts) < args.n_samples:
        n = min(args.gen_batch, args.n_samples - len(texts))
        prompt = torch.full((n, 1), tok.bos_token_id or tok.eos_token_id, device=device)
        out = model.generate(prompt, do_sample=True, max_new_tokens=L, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
        texts.extend([tok.decode(r, skip_special_tokens=True).strip() for r in out])
    m = _quality_block(texts, ref, scorer, args.no_mauve, args.device_id)
    m["nfe"] = L
    res["metrics"]["ar"] = m
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--system", choices=["siflow", "teacher", "ar"], default="siflow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--k-list", type=int, nargs="*", default=[1, 2, 4, 8], dest="k_list")
    ap.add_argument("--steps", type=int, nargs="*", default=[8, 32, 64, 1024])
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--ref-tokens", default=None)
    ap.add_argument("--scorer", default="gpt2-large")
    ap.add_argument("--ar-model", default="gpt2")
    ap.add_argument("--max-lambada", type=int, default=500)
    ap.add_argument("--straightness-n", type=int, default=256)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--no-mauve", action="store_true")
    ap.add_argument("--sample", action="store_true", help="sample tokens (else argmax)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn = {"siflow": eval_siflow, "teacher": eval_teacher, "ar": eval_ar}[args.system]
    res = fn(cfg, args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()
