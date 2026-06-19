#!/usr/bin/env python
"""Generate the run_0 .. run_8 Colab notebooks (valid nbformat-4 JSON).

Re-run this whenever the pipeline changes:  python notebooks/_build.py
Set REPO_URL to the pushed GitHub repo so cell 1 clones the right place.
"""
from __future__ import annotations

import json
import os

REPO_URL = "https://github.com/kagtgi/siflow.git"  # public push target
HERE = os.path.dirname(os.path.abspath(__file__))

_uid = [0]


def _cell(kind, src):
    _uid[0] += 1
    base = {"metadata": {}, "source": src, "id": f"c{_uid[0]:03d}"}
    if kind == "markdown":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


def md(src):
    return _cell("markdown", src)


def code(src):
    return _cell("code", src)


SETUP = f"""# --- 1. Clone + install (run once per session) ---
REPO_URL = "{REPO_URL}"   # <-- edit to your fork if needed
import os
if not os.path.isdir("siflow"):
    !git clone $REPO_URL siflow
%cd siflow
!git pull -q
!pip -q install -e .
!pip -q install -r requirements-colab.txt
print("setup done")"""

MOUNT = """# --- 2. Mount Drive + set artifact base (all sessions share this) ---
from siflow.utils import drive
drive.mount()
import os
os.environ["SIFLOW_BASE"] = "/content/drive/MyDrive/siflow"
BASE = drive.base_dir()
print("artifacts ->", BASE)"""


def header(title, what, needs):
    return [
        md(f"# {title}\n\n{what}\n\n**Hardware:** single A100-80GB, <12h. "
           "All artifacts are written to Google Drive so the next notebook resumes.\n\n"
           f"**Needs from previous notebooks:** {needs}"),
        code(SETUP),
        code(MOUNT),
    ]


def save_cell(paths):
    cmds = "\n".join(f"!cp -r {p} {{BASE}}/ 2>/dev/null || true" for p in paths)
    return code("# --- Save outputs to Drive (so the next notebook can resume) ---\n" + cmds +
                "\nprint('saved to', BASE)")


def build():
    notebooks = {}

    # ---------------- run_0: smoke ----------------
    notebooks["run_0_smoke"] = header(
        "SIFLOW · run_0 · Smoke test",
        "Verifies the install, the full unit-test suite (incl. the SUBS check that is the "
        "riskiest MDLM integration point), and an end-to-end MDLM load + one-step generation. "
        "Run this first; if it passes, the long runs are safe.",
        "nothing (entry point)") + [
        code("# Full unit-test suite (runs the torch tests on the Colab GPU)\n!python -m pytest tests/ -q"),
        code("""# End-to-end: load MDLM, verify SUBS gives ~0 mass on the mask token, 1-step generate
import torch
from siflow.teacher import MDLMTeacher
from siflow.head import VelocityHead
from siflow.student import Student
from transformers import AutoTokenizer

teacher = MDLMTeacher(dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("gpt2")
ids = torch.full((2, 32), teacher.mask_index, device=teacher.device)
z, h = teacher.logits_and_hidden(ids)
p_mask = torch.softmax(z.float(), -1)[..., teacher.mask_index].max().item()
print("max prob on mask token (should be ~0):", p_mask)

head = VelocityHead(teacher.hidden_dim, teacher.embedding_matrix, bottleneck=1024).to(teacher.device)
student = Student(teacher, head)
out = student.generate(2, 32, k=1)
print(tok.decode(out[0].tolist()))
print("smoke OK")"""),
        code("""import json, os
os.makedirs("results", exist_ok=True)
json.dump({"smoke": "ok", "mask_prob": float(p_mask)}, open("results/smoke_ok.json", "w"))"""),
        save_cell(["results"]),
    ]

    # ---------------- run_1: MDLM data ----------------
    notebooks["run_1_mdlm_data_cache"] = header(
        "SIFLOW · run_1 · MDLM data prep",
        "Streams + tokenizes OpenWebText into length-256 GPT-2 chunks (train + a disjoint val "
        "split used as the MAUVE reference). The MDLM teacher is cheap enough to run live, so "
        "**no simplex cache is needed** for the primary study.",
        "run_0 passed") + [
        code("""from transformers import AutoTokenizer
from siflow.data import build_token_chunks

tok = AutoTokenizer.from_pretrained("gpt2")
N_TRAIN = 200_000   # ~51M tokens; reused across epochs
N_VAL   = 5_000

n = build_token_chunks(tok, 256, N_TRAIN, f"{BASE}/data/owt_gpt2_256.npy",
                       dataset="Skylion007/openwebtext", split="train")
print("train chunks:", n)
nv = build_token_chunks(tok, 256, N_VAL, f"{BASE}/data/owt_gpt2_val.npy",
                        dataset="Skylion007/openwebtext", split="train", skip_seqs=N_TRAIN)
print("val chunks:", nv)"""),
        save_cell([]),
        md("Data lives on Drive at `{BASE}/data/`. Proceed to **run_2** to train the SIFLOW head."),
    ]

    # ---------------- run_2: MDLM train ----------------
    notebooks["run_2_mdlm_train"] = header(
        "SIFLOW · run_2 · Train MDLM SIFLOW head",
        "Trains the velocity head (frozen MDLM backbone) for 20k steps with SATD + secant + MDM "
        "losses. Checkpoints to Drive every 1k steps — if the session times out, just re-run this "
        "notebook and it resumes from the latest checkpoint.",
        "run_1 (owt_gpt2_256.npy)") + [
        code("""!python scripts/train.py --config siflow/config/mdlm.yaml --set \\
    data.tokens_path={BASE}/data/owt_gpt2_256.npy \\
    out_dir={BASE}/ckpt/mdlm run_id=siflow_mdlm train.total_steps=20000"""),
        code("""# peek at the loss curve
from siflow.analysis.curves import load_jsonl, series
import matplotlib.pyplot as plt
rows = load_jsonl(f"{BASE}/ckpt/mdlm/train_log.jsonl")
for k in ("satd", "vel", "mdm"):
    xs, ys = series(rows, k)
    if xs: plt.plot(xs, ys, label=k)
plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.show()"""),
        md("Checkpoint is at `{BASE}/ckpt/mdlm/latest.pt`. Proceed to **run_3** for evaluation + figures."),
    ]

    # ---------------- run_3: MDLM eval ----------------
    notebooks["run_3_mdlm_eval_figures"] = header(
        "SIFLOW · run_3 · MDLM eval + figures (fills Table 2)",
        "Evaluates the SIFLOW student (Gen-PPL, MAUVE, LAMBADA, diversity, throughput across "
        "k=1,2,4,8), the MDLM teacher step-curve, AR GPT-2, and (optionally) SDTT@8; then builds "
        "the straightness/Pareto/entropy figures and auto-fills the LaTeX tables.",
        "run_2 (ckpt/mdlm), run_1 (val tokens)") + [
        code("""# SIFLOW student (k sweep + straightness + entropy analyses)
!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/mdlm --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
    --n-samples 1000 --out results/mdlm.json"""),
        code("""# MDLM teacher step-curve baseline
!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system teacher \\
    --steps 8 32 64 1024 --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
    --n-samples 1000 --out results/mdlm_teacher.json"""),
        code("""# AR GPT-2 reference
!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system ar --ar-model gpt2 \\
    --ref-tokens {BASE}/data/owt_gpt2_val.npy --n-samples 1000 --out results/ar_gpt2.json"""),
        code("""# (optional) SDTT@8 baseline -- same GPT-2 tokenizer / MDLM arch, directly comparable.
# Skips gracefully if the package / checkpoint is unavailable.
try:
    !pip -q install git+https://github.com/jdeschena/sdtt.git
    import torch, json
    from sdtt import load_small_student
    from siflow.eval.gen_ppl import GPT2Scorer, decode_ids
    from siflow.eval.diversity import diversity_metrics
    from transformers import AutoTokenizer
    m = load_small_student(loss="kld", round=7).cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    texts = []
    while len(texts) < 1000:
        s = m.sample(n_samples=64, num_steps=8, seq_len=256)
        texts += decode_ids(s if torch.is_tensor(s) else torch.tensor(s), tok)
    scorer = GPT2Scorer("gpt2-large")
    res = {"run_id": "sdtt", "method": "SDTT", "source": "reproduced",
           "metrics": {"steps=8": {**scorer.perplexity(texts), **diversity_metrics(texts), "nfe": 8}}}
    json.dump(res, open("results/sdtt.json", "w"), indent=2)
    print("SDTT done")
except Exception as e:
    print("SDTT baseline skipped:", e)"""),
        code("""# Figures + auto-filled tables
!python scripts/make_figures.py --results results --train-log {BASE}/ckpt/mdlm/train_log.jsonl
!python scripts/make_tables.py --results results
print(open("paper/tables_auto.tex").read()[:1500])"""),
        save_cell(["results", "paper/figures", "paper/tables_auto.tex"]),
    ]

    # ---------------- run_4: ablations ----------------
    notebooks["run_4_mdlm_ablations"] = header(
        "SIFLOW · run_4 · MDLM ablations (fills Table 3)",
        "Retrains short (5k-step) head variants from the same data and evaluates each at k=1,8: "
        "no-average-velocity (Di[M]O-style), hard-label (no SATD anneal), no entropy prior, "
        "+identity target, prob-space head, and head-depth. Split into two runs if a session "
        "times out (each variant resumes independently).",
        "run_1 (tokens)") + [
        code("""ABLATIONS = {
  "abl_no_avg_velocity": "ablation.no_avg_velocity=true",
  "abl_hard_label":      "ablation.hard_label=true",
  "abl_no_entropy_prior":"train.lam_ent=0.0",
  "abl_identity_target": "train.w_id=0.5",
  "abl_prob_space":      "head.space=prob",
  "abl_head_depth1":     "head.mid_blocks=1",
}
for rid, override in ABLATIONS.items():
    print("=== train", rid, "===")
    !python scripts/train.py --config siflow/config/mdlm.yaml --set \\
        data.tokens_path={BASE}/data/owt_gpt2_256.npy out_dir={BASE}/ckpt/{rid} \\
        run_id={rid} train.total_steps=5000 {override}
    !python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \\
        --ckpt-dir {BASE}/ckpt/{rid} --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
        --n-samples 500 --k-list 1 8 --set run_id={rid} --out results/{rid}.json"""),
        code("!python scripts/make_tables.py --results results\nprint(open('paper/tables_auto.tex').read())"),
        save_cell(["results", "paper/tables_auto.tex"]),
    ]

    # ---------------- run_5: Dream setup/cache ----------------
    notebooks["run_5_dream_cache"] = header(
        "SIFLOW · run_5 · Dream-7B setup (SIFLOW-D)",
        "Downloads Dream-7B, verifies a masked forward, and tokenizes data in **Dream's own "
        "tokenizer**. Default path trains live in run_6 (Dream-7B ~14GB fits on A100-80GB). "
        "An optional cell precomputes a reduced-support cache if you prefer a cache->train split.",
        "run_0 passed") + [
        md("> **Note:** if `tokenizer.mask_token_id` is unset for Dream, set `teacher.mask_token` "
           "in `siflow/config/dream.yaml` (the model card documents the mask token). "
           "If a raw forward doesn't expose `.logits`, set `teacher.auto_class` to the documented class."),
        code("""# Verify the teacher loads and a masked forward is sensible
import torch
from siflow.teacher import DreamTeacher
teacher = DreamTeacher(dtype=torch.bfloat16)
print("vocab", teacher.vocab_size, "hidden", teacher.hidden_dim, "mask_id", teacher.mask_index)
ids = torch.full((1, 16), teacher.mask_index, device=teacher.device)
z = teacher.logits(ids)
print("argmax tokens:", z.argmax(-1)[0][:8].tolist())"""),
        code("""# Tokenize data in Dream's tokenizer (train + val)
from siflow.data import build_token_chunks
tokz = teacher.tokenizer
n  = build_token_chunks(tokz, 256, 60_000, f"{BASE}/data/dream_256.npy",
                        dataset="Skylion007/openwebtext", split="train")
nv = build_token_chunks(tokz, 256, 2_000, f"{BASE}/data/dream_val.npy",
                        dataset="Skylion007/openwebtext", split="train", skip_seqs=60_000)
print("dream chunks:", n, nv)"""),
        code("""# (OPTIONAL) Precompute a reduced-support cache instead of training live.
# Resumable at shard granularity. Skip this cell to train live in run_6.
# !python scripts/build_cache.py --config siflow/config/dream.yaml \\
#     --tokens {BASE}/data/dream_256.npy --out {BASE}/cache/dream --n 50000 --m 128 --batch 8"""),
        save_cell([]),
    ]

    # ---------------- run_6: Dream train+eval ----------------
    notebooks["run_6_dream_train_eval"] = header(
        "SIFLOW · run_6 · Dream-7B train + eval (SIFLOW-D)",
        "Trains a head-only SIFLOW-D student on the Dream-7B backbone (live, reduced top-m "
        "support) and evaluates it. Fills the SIFLOW-D rows of Table 2.",
        "run_5 (dream_256.npy + Dream weights cached)") + [
        code("""!python scripts/train.py --config siflow/config/dream.yaml --set \\
    data.tokens_path={BASE}/data/dream_256.npy \\
    out_dir={BASE}/ckpt/dream run_id=siflow_dream train.total_steps=15000"""),
        code("""!python scripts/evaluate.py --config siflow/config/dream.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/dream --ref-tokens {BASE}/data/dream_val.npy \\
    --n-samples 500 --k-list 1 4 --out results/dream.json"""),
        code("!python scripts/make_tables.py --results results"),
        save_cell(["results", "paper/tables_auto.tex"]),
    ]

    # ---------------- run_7: Gemma setup/cache ----------------
    notebooks["run_7_gemma_cache"] = header(
        "SIFLOW · run_7 · DiffusionGemma setup (SIFLOW-G)",
        "Downloads DiffusionGemma-26B-A4B (~50GB fp16, fits A100-80GB) and tokenizes data in its "
        "tokenizer. Trains live in run_8. Optional reduced-support cache cell included.",
        "run_0 passed") + [
        md("> **Note:** confirm the documented HF class and mask token for "
           "`google/diffusiongemma-26B-A4B-it` and set `teacher.auto_class` / `teacher.mask_token` "
           "in `siflow/config/gemma.yaml` if needed. The MoE backbone uses eager attention by default."),
        code("""import torch
from siflow.teacher import GemmaTeacher
teacher = GemmaTeacher(dtype=torch.bfloat16)
print("vocab", teacher.vocab_size, "hidden", teacher.hidden_dim, "mask_id", teacher.mask_index)
ids = torch.full((1, 16), teacher.mask_index, device=teacher.device)
print("argmax:", teacher.logits(ids).argmax(-1)[0][:8].tolist())"""),
        code("""from siflow.data import build_token_chunks
tokz = teacher.tokenizer
n  = build_token_chunks(tokz, 256, 40_000, f"{BASE}/data/gemma_256.npy",
                        dataset="Skylion007/openwebtext", split="train")
nv = build_token_chunks(tokz, 256, 2_000, f"{BASE}/data/gemma_val.npy",
                        dataset="Skylion007/openwebtext", split="train", skip_seqs=40_000)
print("gemma chunks:", n, nv)"""),
        code("""# (OPTIONAL) reduced-support cache
# !python scripts/build_cache.py --config siflow/config/gemma.yaml \\
#     --tokens {BASE}/data/gemma_256.npy --out {BASE}/cache/gemma --n 30000 --m 128 --batch 4"""),
        save_cell([]),
    ]

    # ---------------- run_8: Gemma train+eval + final aggregate ----------------
    notebooks["run_8_gemma_train_eval"] = header(
        "SIFLOW · run_8 · DiffusionGemma train + eval + final paper artifacts (SIFLOW-G)",
        "Trains the head-only SIFLOW-G student on the DiffusionGemma backbone and evaluates it, "
        "then regenerates ALL tables + figures from every results JSON collected so far. "
        "Copy `paper/tables_auto.tex` and `paper/figures/` back into the paper and recompile.",
        "run_7 (gemma data + weights); ideally run_3/4/6 results on Drive too") + [
        code("""!python scripts/train.py --config siflow/config/gemma.yaml --set \\
    data.tokens_path={BASE}/data/gemma_256.npy \\
    out_dir={BASE}/ckpt/gemma run_id=siflow_gemma train.total_steps=12000"""),
        code("""!python scripts/evaluate.py --config siflow/config/gemma.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/gemma --ref-tokens {BASE}/data/gemma_val.npy \\
    --n-samples 400 --k-list 1 4 --out results/gemma.json"""),
        code("""# Pull every result collected across sessions back from Drive, then regenerate everything
!mkdir -p results && cp -r {BASE}/results/* results/ 2>/dev/null || true
!python scripts/make_tables.py --results results
!python scripts/make_figures.py --results results --train-log {BASE}/ckpt/mdlm/train_log.jsonl
print(open("paper/tables_auto.tex").read())"""),
        save_cell(["results", "paper/figures", "paper/tables_auto.tex"]),
        md("**Done.** Drop `paper/tables_auto.tex` and `paper/figures/*.pdf` into the paper tree and "
           "recompile `paper/siflow_aaai.tex` — Tables 2–4 and the figures are now populated."),
    ]

    # write out
    for name, cells in notebooks.items():
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
                "accelerator": "GPU",
                "colab": {"provenance": []},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    build()
