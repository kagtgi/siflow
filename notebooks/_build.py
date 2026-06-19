#!/usr/bin/env python
"""Generate the run_0 .. run_8 Colab notebooks (valid nbformat-4 JSON).

Handoff model (default): each part writes artifacts under one local base dir,
zips the relevant pieces and AUTO-DOWNLOADS them; the next part UPLOADS those
zip(s) and extracts them back. Flip USE_DRIVE=True in a notebook to persist on
Google Drive instead and skip the import/download steps.

Re-run after changes:  python notebooks/_build.py
"""
from __future__ import annotations

import json
import os

REPO_URL = "https://github.com/kagtgi/siflow.git"
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


SETUP = f"""# === 1. Clone + install (run once per session, ~2 min) ===
REPO_URL = "{REPO_URL}"
import os
if not os.path.isdir("siflow"):
    !git clone $REPO_URL siflow
%cd siflow
!git pull -q
!pip -q install -e .
!pip -q install -r requirements-colab.txt
print("setup done")"""

CONFIG = """# === 2. Where do artifacts live? ===
# Default: a local folder + zip download/upload between parts (no Drive needed).
# Set USE_DRIVE = True to persist on Google Drive instead (then the import and
# download steps below become no-ops).
USE_DRIVE = False

import os
if USE_DRIVE:
    from siflow.utils import drive
    drive.mount()
    os.environ["SIFLOW_BASE"] = "/content/drive/MyDrive/siflow"
    BASE = drive.base_dir()
else:
    BASE = "/content/artifacts"
    os.makedirs(BASE, exist_ok=True)
print("artifacts ->", BASE)"""

HF_LOGIN = """# === Hugging Face login ===
# Required for the gated DiffusionGemma weights; recommended for Dream too.
# Get a token at https://huggingface.co/settings/tokens (read scope).
from huggingface_hub import login
login()"""


def import_section(needs):
    """needs: list of (zipname, description). Empty -> no import section."""
    if not needs:
        return []
    lines = ["### Import the previous part(s)\n",
             "This part needs the output zip(s) you downloaded earlier. Run the cell below — a file "
             "picker opens; select **all** of these at once:\n"]
    for z, d in needs:
        lines.append(f"- `{z}` — {d}")
    lines.append("\n*(If a long run here stopped early at the 11h guard, also re-upload **this** "
                 "part's own checkpoint zip to resume.)* Using Drive instead? Set `USE_DRIVE=True` "
                 "above and skip this.")
    return [
        md("\n".join(lines)),
        code("# === Import previous outputs (pick the .zip files listed above) ===\n"
             "if not USE_DRIVE:\n"
             "    from siflow.utils.io import import_zips\n"
             "    import_zips(BASE)\n"
             "else:\n"
             "    print('USE_DRIVE: reading prior outputs from', BASE)"),
    ]


def export_cell(zipname, include):
    inc = ", ".join(repr(p) for p in include)
    return code(
        "# === Save + auto-download this part's output ===\n"
        "from siflow.utils.io import export_and_download\n"
        f"if not USE_DRIVE:\n"
        f"    export_and_download(BASE, {zipname!r}, include=[{inc}])\n"
        "else:\n"
        "    print('USE_DRIVE: outputs already persisted under', BASE)")


def header(title, what, needs, hf=False, training=False):
    runtime = ("\n\n**Runtime:** designed to finish well under one Colab session. Training stops and "
               "checkpoints automatically at an 11h guard — if that happens, just re-run this notebook "
               "(re-import its checkpoint) and it resumes." if training else
               "\n\n**Runtime:** comfortably under one Colab session.")
    cells = [
        md(f"# {title}\n\n{what}{runtime}\n\n"
           "**How to use:** run every cell top-to-bottom. Cells 1–2 set up the environment and the "
           "artifact location; the last cell downloads this part's output zip for the next notebook."),
        code(SETUP),
        code(CONFIG),
    ]
    if hf:
        cells.append(code(HF_LOGIN))
    cells += import_section(needs)
    return cells


def build():
    nbs = {}

    # ---- run_0 smoke ----
    nbs["run_0_smoke"] = header(
        "SIFLOW · run_0 · Smoke test",
        "Verifies the install, the unit-test suite (incl. the SUBS check), and an end-to-end MDLM "
        "load + one-step generation. Run this first.",
        needs=[]) + [
        code("!python -m pytest tests/ -q"),
        code("""import torch
from siflow.teacher import MDLMTeacher
from siflow.head import VelocityHead
from siflow.student import Student
from transformers import AutoTokenizer

teacher = MDLMTeacher(dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("gpt2")
ids = torch.full((2, 32), teacher.mask_index, device=teacher.device)
z, _ = teacher.logits_and_hidden(ids)
print("max prob on mask token (should be ~0):",
      torch.softmax(z.float(), -1)[..., teacher.mask_index].max().item())
head = VelocityHead(teacher.hidden_dim, teacher.embedding_matrix, bottleneck=1024).to(teacher.device)
print(tok.decode(Student(teacher, head).generate(2, 32, k=1)[0].tolist()))
print("smoke OK")"""),
    ]

    # ---- run_1 data ----
    nbs["run_1_mdlm_data_cache"] = header(
        "SIFLOW · run_1 · MDLM data prep",
        "Tokenizes OpenWebText into length-256 GPT-2 chunks (train + a disjoint val split used as the "
        "MAUVE reference). Lower `N_TRAIN` for a quick smoke.",
        needs=[]) + [
        code("""from transformers import AutoTokenizer
from siflow.data import build_token_chunks

tok = AutoTokenizer.from_pretrained("gpt2")
N_TRAIN = 200_000   # ~51M tokens (set 20_000 for a quick smoke)
N_VAL   = 5_000

print("train chunks:", build_token_chunks(tok, 256, N_TRAIN, f"{BASE}/data/owt_gpt2_256.npy",
      dataset="Skylion007/openwebtext", split="train"))
print("val chunks:",   build_token_chunks(tok, 256, N_VAL, f"{BASE}/data/owt_gpt2_val.npy",
      dataset="Skylion007/openwebtext", split="train", skip_seqs=N_TRAIN))"""),
        export_cell("run_1_data.zip", ["data/owt_gpt2_256.npy", "data/owt_gpt2_val.npy"]),
    ]

    # ---- run_2 train ----
    nbs["run_2_mdlm_train"] = header(
        "SIFLOW · run_2 · Train MDLM SIFLOW head",
        "Trains the velocity head (frozen MDLM backbone) for 20k steps (~3–4h on A100). Checkpoints "
        "every 1k steps; the 11h guard stops + checkpoints if needed.",
        needs=[("run_1_data.zip", "tokenized OpenWebText from run_1")], training=True) + [
        code("""!python scripts/train.py --config siflow/config/mdlm.yaml --set \\
    data.tokens_path={BASE}/data/owt_gpt2_256.npy \\
    out_dir={BASE}/ckpt/mdlm run_id=siflow_mdlm train.total_steps=20000"""),
        code("""from siflow.analysis.curves import load_jsonl, series
import matplotlib.pyplot as plt
rows = load_jsonl(f"{BASE}/ckpt/mdlm/train_log.jsonl")
for k in ("satd", "vel", "mdm"):
    xs, ys = series(rows, k)
    if xs: plt.plot(xs, ys, label=k)
plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.show()"""),
        export_cell("run_2_mdlm_ckpt.zip", ["ckpt/mdlm"]),
    ]

    # ---- run_3 eval ----
    nbs["run_3_mdlm_eval_figures"] = header(
        "SIFLOW · run_3 · MDLM eval + figures (fills Table 2)",
        "Evaluates the SIFLOW student (k=1,2,4,8), the MDLM teacher step-curve, AR GPT-2, and "
        "optionally SDTT@8; builds the figures and auto-fills the LaTeX tables.",
        needs=[("run_1_data.zip", "val tokens for MAUVE + analyses"),
               ("run_2_mdlm_ckpt.zip", "the trained MDLM head")], training=False) + [
        code("""!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/mdlm --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
    --n-samples 1000 --out {BASE}/results/mdlm.json"""),
        code("""!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system teacher \\
    --steps 8 32 64 1024 --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
    --n-samples 1000 --out {BASE}/results/mdlm_teacher.json"""),
        code("""!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system ar --ar-model gpt2 \\
    --ref-tokens {BASE}/data/owt_gpt2_val.npy --n-samples 1000 --out {BASE}/results/ar_gpt2.json"""),
        code("""# (optional) SDTT@8 baseline -- skips gracefully if unavailable
try:
    !pip -q install git+https://github.com/jdeschena/sdtt.git
    import torch, json
    from sdtt import load_small_student
    from siflow.eval.gen_ppl import GPT2Scorer, decode_ids
    from siflow.eval.diversity import diversity_metrics
    from transformers import AutoTokenizer
    m = load_small_student(loss="kld", round=7).cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2"); texts = []
    while len(texts) < 1000:
        s = m.sample(n_samples=64, num_steps=8, seq_len=256)
        texts += decode_ids(s if torch.is_tensor(s) else torch.tensor(s), tok)
    sc = GPT2Scorer("gpt2-large")
    json.dump({"run_id":"sdtt","method":"SDTT","source":"reproduced",
               "metrics":{"steps=8":{**sc.perplexity(texts),**diversity_metrics(texts),"nfe":8}}},
              open(f"{BASE}/results/sdtt.json","w"), indent=2)
    print("SDTT done")
except Exception as e:
    print("SDTT skipped:", e)"""),
        code("""!python scripts/make_figures.py --results {BASE}/results --train-log {BASE}/ckpt/mdlm/train_log.jsonl --out-dir {BASE}/figures
!python scripts/make_tables.py --results {BASE}/results --out {BASE}/tables_auto.tex
print(open(f"{BASE}/tables_auto.tex").read()[:1500])"""),
        export_cell("run_3_results.zip", ["results", "figures", "tables_auto.tex"]),
    ]

    # ---- run_4 ablations ----
    nbs["run_4_mdlm_ablations"] = header(
        "SIFLOW · run_4 · MDLM ablations (fills Table 3)",
        "Retrains short (5k-step) head variants and evaluates each at k=1,8. ~5–7h total; each "
        "variant resumes independently if interrupted.",
        needs=[("run_1_data.zip", "tokenized data + val"),
               ("run_3_results.zip", "(optional) keeps run_3's main-table rows alongside the ablations")],
        training=True) + [
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
        run_id={rid} train.total_steps=5000 train.max_hours=1.5 {override}
    !python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \\
        --ckpt-dir {BASE}/ckpt/{rid} --ref-tokens {BASE}/data/owt_gpt2_val.npy \\
        --n-samples 500 --k-list 1 8 --set run_id={rid} --out {BASE}/results/{rid}.json"""),
        code("!python scripts/make_tables.py --results {BASE}/results --out {BASE}/tables_auto.tex\nprint(open(f'{BASE}/tables_auto.tex').read())"),
        export_cell("run_4_ablations.zip", ["results", "tables_auto.tex"]),
    ]

    # ---- run_5 dream setup ----
    nbs["run_5_dream_cache"] = header(
        "SIFLOW · run_5 · Dream-7B setup (SIFLOW-D)",
        "Downloads Dream-7B (~14GB), verifies a masked forward, and tokenizes data in Dream's "
        "tokenizer. Trains live in run_6 (no cache needed). ~1h.",
        needs=[], hf=True) + [
        md("> If `tokenizer.mask_token_id` is unset for Dream, set `teacher.mask_token` in "
           "`siflow/config/dream.yaml`. If a raw forward lacks `.logits`, set `teacher.auto_class`."),
        code("""import torch
from siflow.teacher import DreamTeacher
teacher = DreamTeacher(dtype=torch.bfloat16)
print("vocab", teacher.vocab_size, "hidden", teacher.hidden_dim, "mask_id", teacher.mask_index)
print("argmax:", teacher.logits(torch.full((1,16), teacher.mask_index, device=teacher.device)).argmax(-1)[0][:8].tolist())"""),
        code("""from siflow.data import build_token_chunks
tokz = teacher.tokenizer
print("dream chunks:",
      build_token_chunks(tokz, 256, 60_000, f"{BASE}/data/dream_256.npy", dataset="Skylion007/openwebtext", split="train"),
      build_token_chunks(tokz, 256, 2_000, f"{BASE}/data/dream_val.npy", dataset="Skylion007/openwebtext", split="train", skip_seqs=60_000))"""),
        export_cell("run_5_dream_data.zip", ["data/dream_256.npy", "data/dream_val.npy"]),
    ]

    # ---- run_6 dream train+eval ----
    nbs["run_6_dream_train_eval"] = header(
        "SIFLOW · run_6 · Dream-7B train + eval (SIFLOW-D)",
        "Head-only SIFLOW-D on the Dream-7B backbone (live, reduced top-m support), then eval. "
        "~6–8h. Fills the SIFLOW-D rows of Table 2.",
        needs=[("run_5_dream_data.zip", "Dream-tokenized data"),
               ("run_3_results.zip / run_4_ablations.zip", "(optional) to keep prior rows in the table")],
        hf=True, training=True) + [
        code("""!python scripts/train.py --config siflow/config/dream.yaml --set \\
    data.tokens_path={BASE}/data/dream_256.npy \\
    out_dir={BASE}/ckpt/dream run_id=siflow_dream train.total_steps=12000"""),
        code("""!python scripts/evaluate.py --config siflow/config/dream.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/dream --ref-tokens {BASE}/data/dream_val.npy \\
    --n-samples 500 --k-list 1 4 --out {BASE}/results/dream.json"""),
        code("!python scripts/make_tables.py --results {BASE}/results --out {BASE}/tables_auto.tex"),
        export_cell("run_6_dream.zip", ["ckpt/dream", "results/dream.json"]),
    ]

    # ---- run_7 gemma setup ----
    nbs["run_7_gemma_cache"] = header(
        "SIFLOW · run_7 · DiffusionGemma setup (SIFLOW-G)",
        "Downloads DiffusionGemma-26B-A4B (~50GB, fits A100-80GB) and tokenizes data in its tokenizer. "
        "Trains live in run_8. ~1.5h (download-bound).",
        needs=[], hf=True) + [
        md("> Confirm the documented HF class + mask token for `google/diffusiongemma-26B-A4B-it` and "
           "set `teacher.auto_class` / `teacher.mask_token` in `siflow/config/gemma.yaml` if needed."),
        code("""import torch
from siflow.teacher import GemmaTeacher
teacher = GemmaTeacher(dtype=torch.bfloat16)
print("vocab", teacher.vocab_size, "hidden", teacher.hidden_dim, "mask_id", teacher.mask_index)"""),
        code("""from siflow.data import build_token_chunks
tokz = teacher.tokenizer
print("gemma chunks:",
      build_token_chunks(tokz, 256, 40_000, f"{BASE}/data/gemma_256.npy", dataset="Skylion007/openwebtext", split="train"),
      build_token_chunks(tokz, 256, 2_000, f"{BASE}/data/gemma_val.npy", dataset="Skylion007/openwebtext", split="train", skip_seqs=40_000))"""),
        export_cell("run_7_gemma_data.zip", ["data/gemma_256.npy", "data/gemma_val.npy"]),
    ]

    # ---- run_8 gemma train+eval + final ----
    nbs["run_8_gemma_train_eval"] = header(
        "SIFLOW · run_8 · DiffusionGemma train + eval + FINAL paper artifacts (SIFLOW-G)",
        "Head-only SIFLOW-G on the DiffusionGemma backbone (6k steps, ~7h), then regenerates ALL "
        "tables + figures from every results JSON you import. The downloaded zip holds the final "
        "`tables_auto.tex` + `figures/` for the paper.",
        needs=[("run_7_gemma_data.zip", "Gemma-tokenized data (required)"),
               ("run_3_results.zip", "MDLM main-table rows"),
               ("run_4_ablations.zip", "ablation rows"),
               ("run_6_dream.zip", "SIFLOW-D rows")],
        hf=True, training=True) + [
        code("""!python scripts/train.py --config siflow/config/gemma.yaml --set \\
    data.tokens_path={BASE}/data/gemma_256.npy \\
    out_dir={BASE}/ckpt/gemma run_id=siflow_gemma train.total_steps=6000"""),
        code("""!python scripts/evaluate.py --config siflow/config/gemma.yaml --system siflow \\
    --ckpt-dir {BASE}/ckpt/gemma --ref-tokens {BASE}/data/gemma_val.npy \\
    --n-samples 400 --k-list 1 4 --out {BASE}/results/gemma.json"""),
        code("""!python scripts/make_tables.py  --results {BASE}/results --out {BASE}/tables_auto.tex
!python scripts/make_figures.py --results {BASE}/results --out-dir {BASE}/figures
print(open(f"{BASE}/tables_auto.tex").read())"""),
        export_cell("run_8_final_paper_artifacts.zip", ["results", "figures", "tables_auto.tex"]),
        md("**Done.** Unzip `run_8_final_paper_artifacts.zip` into `paper/` (drop `tables_auto.tex` and "
           "`figures/*.pdf` in) and recompile `paper/siflow_aaai.tex` — Tables 2–4 and the figures populate."),
    ]

    for name, cells in nbs.items():
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
        with open(os.path.join(HERE, f"{name}.ipynb"), "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    build()
