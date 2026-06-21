#!/usr/bin/env python
"""Generate the TWO Colab notebooks that produce every paper result.

Design: a single A100-40GB, < 12h per notebook, no quantization.
  * nb1_mdlm.ipynb           -- MDLM-170M full study: data -> train -> eval ->
                                 ablations -> figures/tables (Table 2 main + Table 3).
  * nb2_large_teachers.ipynb -- Dream-7B (-D) and LLaDA-8B (-L) head-only train+eval
                                 sequentially (each fits 40GB fp16), then regenerate
                                 the FINAL combined tables/figures.

Every stage is guarded by an existence check and training auto-resumes from its
checkpoint, so if a Colab session ends early you just re-run the notebook and it
skips finished work. Handoff is by zip (auto-download at the end / file-picker
upload at the top); flip USE_DRIVE=True to persist on Google Drive instead.

Re-run after changes:  python notebooks/_build.py
"""
from __future__ import annotations

import glob
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
# Default: a local folder + zip download/upload between the 2 notebooks (no Drive).
# Set USE_DRIVE = True to persist on Google Drive instead (the import + download
# steps then become no-ops and everything survives a disconnect automatically).
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
# Recommended for the Dream-7B / LLaDA-8B weights (faster, avoids rate limits).
# Get a token at https://huggingface.co/settings/tokens (read scope).
from huggingface_hub import login
login()"""


def import_section(needs):
    """needs: list of (zipname, description). Empty -> no import section."""
    if not needs:
        return []
    lines = ["### Import the previous part\n",
             "Run the cell below — a file picker opens; select the zip(s) you downloaded earlier:\n"]
    for z, d in needs:
        lines.append(f"- `{z}` — {d}")
    lines.append("\n*(Re-running this notebook after a disconnect? Also re-upload **this** part's own "
                 "output zip — finished stages are skipped and training resumes from its checkpoint.)* "
                 "Using Drive? Set `USE_DRIVE=True` above and skip this.")
    return [
        md("\n".join(lines)),
        code("# === Import previous outputs (pick the .zip file(s) listed above) ===\n"
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


def header(title, what, needs, hf=False):
    runtime = ("\n\n**Runtime:** designed to finish in one Colab session on a single **A100-40GB**. "
               "Every stage is checkpointed and guarded by an existence check, and training has an 11h "
               "wall-clock guard — if a session ends early, just re-run this notebook (re-upload its own "
               "zip) and it skips finished stages and resumes training.")
    cells = [
        md(f"# {title}\n\n{what}{runtime}\n\n"
           "**How to use:** run every cell top-to-bottom. Cells 1–2 set up the environment and the "
           "artifact location; the last cell downloads this part's output zip."),
        code(SETUP),
        code(CONFIG),
    ]
    if hf:
        cells.append(code(HF_LOGIN))
    cells += import_section(needs)
    return cells


# --------------------------------------------------------------------------- #
SMOKE = """# === Quick smoke: unit tests + MDLM load + SUBS sanity ===
!python -m pytest tests/ -q"""

SMOKE_PY = """import torch
from siflow.teacher import MDLMTeacher
teacher = MDLMTeacher(dtype=torch.bfloat16)
ids = torch.full((2, 32), teacher.mask_index, device=teacher.device)
z, _ = teacher.logits_and_hidden(ids)
print("P(mask) on mask token (should be ~0):",
      torch.softmax(z.float(), -1)[..., teacher.mask_index].max().item())
del teacher, z
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("smoke OK")"""


def build_nb1():
    cells = header(
        "SIFLOW · Notebook 1 · MDLM full study (Table 2 main + Table 3)",
        "Runs the entire primary study on the MDLM-170M teacher on a single A100-40GB: tokenize "
        "OpenWebText, train the velocity head, evaluate the k-sweep + teacher step-curve + AR GPT-2 "
        "(+ optional SDTT@8), build all figures, then the ablation suite — and auto-fills the LaTeX "
        "tables. Downloads `nb1_mdlm_outputs.zip` for Notebook 2.",
        needs=[]) + [
        code(SMOKE),
        code(SMOKE_PY),
        code("""# === Sizes (shrink any of these for a fast end-to-end smoke) ===
N_TRAIN   = 160_000   # training sequences (set 20_000 for a quick smoke)
N_VAL     = 5_000
STEPS     = 12000     # MDLM head training steps
ABL_STEPS = 3000      # steps per ablation variant
print("sizes set")"""),
        code("""# === Data: tokenize OpenWebText (train + disjoint val for MAUVE) ===
import os
from transformers import AutoTokenizer
from siflow.data import build_token_chunks
tok = AutoTokenizer.from_pretrained("gpt2")
if not os.path.exists(f"{BASE}/data/owt_gpt2_256.npy"):
    build_token_chunks(tok, 256, N_TRAIN, f"{BASE}/data/owt_gpt2_256.npy",
                       dataset="Skylion007/openwebtext", split="train")
if not os.path.exists(f"{BASE}/data/owt_gpt2_val.npy"):
    build_token_chunks(tok, 256, N_VAL, f"{BASE}/data/owt_gpt2_val.npy",
                       dataset="Skylion007/openwebtext", split="train", skip_seqs=N_TRAIN)
print("data ready")"""),
        code("""# === Train the MDLM velocity head (auto-resumes from checkpoint) ===
ip = get_ipython()
ip.system(f"python scripts/train.py --config siflow/config/mdlm.yaml --set "
          f"data.tokens_path={BASE}/data/owt_gpt2_256.npy "
          f"out_dir={BASE}/ckpt/mdlm run_id=siflow_mdlm train.total_steps={STEPS}")"""),
        code("""# === Training curves (skips cleanly if the log isn't there yet) ===
import os
_log = f"{BASE}/ckpt/mdlm/train_log.jsonl"
if os.path.exists(_log):
    from siflow.analysis.curves import load_jsonl, series
    import matplotlib.pyplot as plt
    rows = load_jsonl(_log)
    for k in ("satd", "vel", "mdm"):
        xs, ys = series(rows, k)
        if xs: plt.plot(xs, ys, label=k)
    plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.title("MDLM head training"); plt.show()
else:
    print("no train log at", _log)"""),
        code("""# === Evaluate SIFLOW (k-sweep), MDLM teacher step-curve, and AR GPT-2 ===
import os
ip = get_ipython()
if not os.path.exists(f"{BASE}/results/mdlm.json"):
    ip.system(f"python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow "
              f"--ckpt-dir {BASE}/ckpt/mdlm --ref-tokens {BASE}/data/owt_gpt2_val.npy "
              f"--n-samples 1000 --k-list 1 2 4 8 --straightness-n 128 --out {BASE}/results/mdlm.json")
if not os.path.exists(f"{BASE}/results/mdlm_teacher.json"):
    ip.system(f"python scripts/evaluate.py --config siflow/config/mdlm.yaml --system teacher "
              f"--steps 8 32 64 1024 --ref-tokens {BASE}/data/owt_gpt2_val.npy "
              f"--n-samples 1000 --out {BASE}/results/mdlm_teacher.json")
if not os.path.exists(f"{BASE}/results/ar_gpt2.json"):
    ip.system(f"python scripts/evaluate.py --config siflow/config/mdlm.yaml --system ar --ar-model gpt2 "
              f"--ref-tokens {BASE}/data/owt_gpt2_val.npy --n-samples 1000 --out {BASE}/results/ar_gpt2.json")
print("eval done")"""),
        code("""# === (optional) SDTT@8 baseline -- skips gracefully if unavailable ===
import os
if not os.path.exists(f"{BASE}/results/sdtt.json"):
    try:
        !pip -q install git+https://github.com/jdeschena/sdtt.git
        import torch, json
        from sdtt import load_small_student
        from siflow.eval import decode_ids
        from siflow.eval.gen_ppl import GPT2Scorer
        from siflow.eval.diversity import diversity_metrics
        from transformers import AutoTokenizer
        m = load_small_student(loss="kld", round=7).cuda().eval()
        tok = AutoTokenizer.from_pretrained("gpt2"); texts = []
        while len(texts) < 1000:
            s = m.sample(n_samples=64, num_steps=8, seq_len=256)
            texts += decode_ids(s if torch.is_tensor(s) else torch.tensor(s), tok)
        sc = GPT2Scorer("gpt2-large")
        json.dump({"run_id": "sdtt", "method": "SDTT", "source": "reproduced",
                   "metrics": {"steps=8": {**sc.perplexity(texts), **diversity_metrics(texts), "nfe": 8}}},
                  open(f"{BASE}/results/sdtt.json", "w"), indent=2)
        print("SDTT done")
    except Exception as e:
        print("SDTT skipped:", e)"""),
        code("""# === Figures + tables (Table 2 main rows so far) ===
ip = get_ipython()
ip.system(f"python scripts/make_figures.py --results {BASE}/results "
          f"--train-log {BASE}/ckpt/mdlm/train_log.jsonl --out-dir {BASE}/figures")
ip.system(f"python scripts/make_tables.py --results {BASE}/results --out {BASE}/tables_auto.tex")
print(open(f"{BASE}/tables_auto.tex").read()[:1500])"""),
        export_cell("nb1_mdlm_outputs.zip", ["results", "figures", "tables_auto.tex", "ckpt/mdlm"]),
        code("""# === Ablations (Table 3) -- each variant is guarded + resumable ===
import os
ip = get_ipython()
ABLATIONS = {
  "abl_no_avg_velocity": "ablation.no_avg_velocity=true",
  "abl_hard_label":      "ablation.hard_label=true",
  "abl_no_entropy_prior":"train.lam_ent=0.0",
  "abl_identity_target": "train.w_id=0.5",
  "abl_prob_space":      "head.space=prob",
  "abl_head_depth1":     "head.mid_blocks=1",
}
for rid, override in ABLATIONS.items():
    if os.path.exists(f"{BASE}/results/{rid}.json"):
        print("skip (done):", rid); continue
    print("=== train", rid, "===")
    ip.system(f"python scripts/train.py --config siflow/config/mdlm.yaml --set "
              f"data.tokens_path={BASE}/data/owt_gpt2_256.npy out_dir={BASE}/ckpt/{rid} "
              f"run_id={rid} train.total_steps={ABL_STEPS} train.max_hours=1.0 {override}")
    ip.system(f"python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow "
              f"--ckpt-dir {BASE}/ckpt/{rid} --ref-tokens {BASE}/data/owt_gpt2_val.npy "
              f"--n-samples 400 --k-list 1 8 --straightness-n 0 --set run_id={rid} "
              f"--out {BASE}/results/{rid}.json")
print("ablations done")"""),
        code("""# === Regenerate tables (now with the ablation rows) + re-export ===
ip = get_ipython()
ip.system(f"python scripts/make_tables.py --results {BASE}/results --out {BASE}/tables_auto.tex")
print(open(f"{BASE}/tables_auto.tex").read())"""),
        export_cell("nb1_mdlm_outputs.zip", ["results", "figures", "tables_auto.tex", "ckpt/mdlm"]),
        md("**Done with Notebook 1.** Keep `nb1_mdlm_outputs.zip` — Notebook 2 imports it so the final "
           "tables include these primary (MDLM) + ablation rows."),
    ]
    return cells


# LLaDA gets --no-mauve to stay comfortably inside one session; everything else identical.
# Each artifact is guarded independently so a mid-session disconnect resumes cleanly:
# tokens -> head train -> SIFLOW eval -> teacher-reference eval. Training auto-resumes
# from its checkpoint; the teacher is a fresh subprocess each call, so VRAM is freed
# between Dream and LLaDA (only one 7-8B model is ever resident).
def _large_teacher_block(tag, kind, cfg, model_name, steps_var, ntok_var, extra_eval=""):
    return f'''# === SIFLOW-{tag}: {model_name} (head-only on a single 40GB card; guarded + resumable) ===
import os
from transformers import AutoTokenizer
from siflow.data import build_token_chunks
ip = get_ipython()
if os.path.exists(f"{{BASE}}/results/{kind}.json") and os.path.exists(f"{{BASE}}/results/{kind}_teacher.json"):
    print("SIFLOW-{tag} already complete; skipping")
else:
    tk = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)
    if not os.path.exists(f"{{BASE}}/data/{kind}_256.npy"):
        build_token_chunks(tk, 256, {ntok_var}, f"{{BASE}}/data/{kind}_256.npy", split="train")
    if not os.path.exists(f"{{BASE}}/data/{kind}_val.npy"):
        build_token_chunks(tk, 256, 1_500, f"{{BASE}}/data/{kind}_val.npy", split="train", skip_seqs={ntok_var})
    del tk
    if not os.path.exists(f"{{BASE}}/results/{kind}.json"):
        ip.system(f"python scripts/train.py --config siflow/config/{cfg}.yaml --set "
                  f"data.tokens_path={{BASE}}/data/{kind}_256.npy out_dir={{BASE}}/ckpt/{kind} "
                  f"run_id=siflow_{kind} train.total_steps={{{steps_var}}}")
        ip.system(f"python scripts/evaluate.py --config siflow/config/{cfg}.yaml --system siflow --variant {tag} "
                  f"--ckpt-dir {{BASE}}/ckpt/{kind} --ref-tokens {{BASE}}/data/{kind}_val.npy --gen-batch 16 "
                  f"--n-samples 500 --k-list 1 4 --straightness-n 0 {extra_eval}--out {{BASE}}/results/{kind}.json")
    if not os.path.exists(f"{{BASE}}/results/{kind}_teacher.json"):
        ip.system(f"python scripts/evaluate.py --config siflow/config/{cfg}.yaml --system teacher --variant {tag} "
                  f"--steps 64 --ref-tokens {{BASE}}/data/{kind}_val.npy --gen-batch 16 --n-samples 300 {extra_eval}"
                  f"--out {{BASE}}/results/{kind}_teacher.json")
    print("SIFLOW-{tag} done")'''


def build_nb2():
    cells = header(
        "SIFLOW · Notebook 2 · Dream-7B & LLaDA-8B (Table 2 -D / -L) + final artifacts",
        "Head-only SIFLOW on two larger backbones, each fitting a single A100-40GB in fp16 (no "
        "quantization): **Dream-7B** (~14 GB, -D) then **LLaDA-8B** (~16 GB, -L), trained and "
        "evaluated sequentially with the teacher freed between them. Then regenerates the FINAL "
        "combined tables + figures from every results JSON (the imported MDLM rows plus -D/-L). "
        "Downloads `nb2_final_paper_artifacts.zip` for the paper.",
        needs=[("nb1_mdlm_outputs.zip", "all MDLM + ablation results so the final tables include the primary rows")],
        hf=True) + [
        code("""# === Sizes (shrink for a smoke; total stays well under one 12h session) ===
DREAM_STEPS = 8000
LLADA_STEPS = 8000
N_DREAM_TOK = 40_000
N_LLADA_TOK = 40_000
print("sizes set")"""),
        code(_large_teacher_block("D", "dream", "dream", "Dream-org/Dream-v0-Base-7B",
                                  "DREAM_STEPS", "N_DREAM_TOK")),
        code(_large_teacher_block("L", "llada", "llada", "GSAI-ML/LLaDA-8B-Base",
                                  "LLADA_STEPS", "N_LLADA_TOK", extra_eval="--no-mauve ")),
        code("""# === FINAL combined tables + figures (MDLM + Dream-7B + LLaDA-8B) ===
ip = get_ipython()
ip.system(f"python scripts/make_tables.py  --results {BASE}/results --out {BASE}/tables_auto.tex")
ip.system(f"python scripts/make_figures.py --results {BASE}/results --out-dir {BASE}/figures")
print(open(f"{BASE}/tables_auto.tex").read())"""),
        export_cell("nb2_final_paper_artifacts.zip",
                    ["results", "figures", "tables_auto.tex", "ckpt/dream", "ckpt/llada"]),
        md("**Done.** Unzip `nb2_final_paper_artifacts.zip` into `paper/` (drop `tables_auto.tex` and "
           "`figures/*.pdf` in) and recompile `paper/siflow_aaai.tex` — Tables 2–3 and the figures populate "
           "with MDLM, Dream-7B (-D) and LLaDA-8B (-L) rows."),
    ]
    return cells


def build():
    # remove the old 9-notebook layout so only the 2 current notebooks remain
    for old in glob.glob(os.path.join(HERE, "run_*.ipynb")):
        os.remove(old)
        print("removed", os.path.basename(old))

    nbs = {"nb1_mdlm": build_nb1(), "nb2_large_teachers": build_nb2()}
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
