# SIFLOW — Running the notebooks on Colab (step by step)

This walks you through running the whole pipeline on a **single Colab A100-80GB**, one notebook
per session, with every output saved to Google Drive so the next notebook (or a re-run after a
timeout) just picks up where you left off.

You do **not** run anything locally. Each notebook clones this repo, installs it, mounts your
Drive, does its part, and copies its outputs to `MyDrive/siflow/`.

---

## 0. One-time prerequisites

1. **Colab with an A100-80GB.** You need Colab Pro / Pro+ (or Colab Enterprise). In each notebook:
   `Runtime → Change runtime type → A100 GPU` and, if offered, **High-RAM**. Confirm with the first
   cell's output (or run `!nvidia-smi` — you want ~80 GB).
2. **Google Drive space.** The defaults are light: tokenized data + head checkpoints + results total
   well under **1 GB**. (Only the *optional* Dream/Gemma simplex caches are large — tens of GB — and
   you can skip them; the default path trains live.)
3. **A Hugging Face account + token** (free). Needed because:
   - **DiffusionGemma** (`google/diffusiongemma-26B-A4B-it`) is a **gated** Gemma model — open its
     model page once, click *Agree and access*, then use a token.
   - Dream-7B / MDLM / OpenWebText / GPT-2 are public, but a token avoids rate limits.
   Get a token at <https://huggingface.co/settings/tokens> (read scope is enough).

> **Everything lands in `MyDrive/siflow/`** — `data/`, `ckpt/<run_id>/`, `results/`, `figures/`,
> `tables_auto.tex`. You can delete that folder to start over.

---

## 1. How to open a notebook in Colab

Click any of these (they open the notebook straight from GitHub):

- run_0 — <https://colab.research.google.com/github/kagtgi/siflow/blob/main/notebooks/run_0_smoke.ipynb>
- run_1 — `.../notebooks/run_1_mdlm_data_cache.ipynb`
- run_2 — `.../notebooks/run_2_mdlm_train.ipynb`
- run_3 — `.../notebooks/run_3_mdlm_eval_figures.ipynb`
- run_4 — `.../notebooks/run_4_mdlm_ablations.ipynb`
- run_5 — `.../notebooks/run_5_dream_cache.ipynb`
- run_6 — `.../notebooks/run_6_dream_train_eval.ipynb`
- run_7 — `.../notebooks/run_7_gemma_cache.ipynb`
- run_8 — `.../notebooks/run_8_gemma_train_eval.ipynb`

(Swap the filename in the URL, or browse `notebooks/` on GitHub and click **“Open in Colab”**.)

---

## 2. The two cells every notebook starts with

**Cell 1 — clone + install** (run once per session; ~2 min):
```python
REPO_URL = "https://github.com/kagtgi/siflow.git"
import os
if not os.path.isdir("siflow"):
    !git clone $REPO_URL siflow
%cd siflow
!git pull -q
!pip -q install -e .
!pip -q install -r requirements-colab.txt
```

**Cell 2 — mount Drive + set the artifact base** (this is what makes outputs persist):
```python
from siflow.utils import drive
drive.mount()                                  # approve the Drive popup
import os
os.environ["SIFLOW_BASE"] = "/content/drive/MyDrive/siflow"
BASE = drive.base_dir()
print("artifacts ->", BASE)                    # /content/drive/MyDrive/siflow
```

After these, just run the remaining cells **top to bottom**. The **last cell** of each notebook
copies that notebook's outputs to `BASE` — don't skip it.

**(For gated models — only needed for run_7/run_8, and good practice for run_5/6):** add and run
this once, right after Cell 1:
```python
from huggingface_hub import login
login()   # paste your HF token; or: !huggingface-cli login
```

---

## 3. Run the notebooks in order

Run them **sequentially**. Each row says what it needs from earlier ones and what it leaves on Drive.

| # | Notebook | ~Time | Reads from Drive | Writes to Drive |
|---|----------|-------|------------------|-----------------|
| 0 | `run_0_smoke` | 15 min | — | `results/smoke_ok.json` |
| 1 | `run_1_mdlm_data_cache` | 1–2 h | — | `data/owt_gpt2_256.npy`, `data/owt_gpt2_val.npy` |
| 2 | `run_2_mdlm_train` | 6–9 h | `data/owt_gpt2_256.npy` | `ckpt/mdlm/latest.pt`, `ckpt/mdlm/train_log.jsonl` |
| 3 | `run_3_mdlm_eval_figures` | 4–6 h | `ckpt/mdlm`, `data/owt_gpt2_val.npy` | `results/mdlm*.json`, `results/ar_gpt2.json`, `figures/`, `tables_auto.tex` |
| 4 | `run_4_mdlm_ablations` | 8–10 h | `data/*` | `results/abl_*.json` |
| 5 | `run_5_dream_cache` | 6–10 h | — | `data/dream_256.npy`, `data/dream_val.npy` |
| 6 | `run_6_dream_train_eval` | 6–9 h | `data/dream_*` | `ckpt/dream`, `results/dream.json` |
| 7 | `run_7_gemma_cache` | 8–11 h | — | `data/gemma_256.npy`, `data/gemma_val.npy` |
| 8 | `run_8_gemma_train_eval` | 6–9 h | `data/gemma_*` + all prior `results/` | `ckpt/gemma`, `results/gemma.json`, regenerated `tables_auto.tex` + `figures/` |

**Minimum path to a complete primary result (the AAAI Table 2 + figures):** run **0 → 1 → 2 → 3**,
then **4** for the ablation table. Notebooks **5–8** add the SIFLOW-D / SIFLOW-G rows and can be done
later (each is the heavy large-teacher path).

You can split a single notebook across sessions — see §5.

---

## 4. Where the outputs are (and how they chain)

```
MyDrive/siflow/
├── data/         owt_gpt2_256.npy, owt_gpt2_val.npy, dream_256.npy, ...   (run_1 / run_5 / run_7)
├── ckpt/
│   ├── mdlm/     latest.pt, step_*.pt, train_log.jsonl                    (run_2; resumable)
│   ├── dream/  · gemma/  · abl_*/                                          (run_6 / run_8 / run_4)
├── results/      mdlm.json, mdlm_teacher.json, ar_gpt2.json, abl_*.json,  (run_3/4/6/8)
│                 dream.json, gemma.json
├── figures/      pareto.pdf, straightness.pdf, entropy.pdf, lambada_vs_k.pdf
└── tables_auto.tex
```

- **Training** (`run_2/6/8`) checkpoints to `ckpt/<run_id>/latest.pt` every 1k steps.
- **Eval** (`run_3/4/6/8`) writes one JSON per system into `results/` and copies them to Drive, so they
  **accumulate** across sessions. `run_8` pulls *all* of `results/` back and regenerates the final
  `tables_auto.tex` + `figures/` from everything collected.

To regenerate the tables/figures at **any** point from everything collected so far:
```python
!mkdir -p results && cp -r {BASE}/results/* results/ 2>/dev/null || true
!python scripts/make_tables.py  --results results
!python scripts/make_figures.py --results results --train-log {BASE}/ckpt/mdlm/train_log.jsonl
!cp paper/tables_auto.tex {BASE}/ ; cp -r paper/figures {BASE}/
```

---

## 5. Resuming after a 12-hour timeout (the important part)

If a session disconnects mid-run, **nothing is lost**:

1. Reopen the **same** notebook in a fresh A100 session.
2. Run **Cell 1** (clone+install) and **Cell 2** (mount Drive) again.
3. Re-run the same work cell.
   - **Training cells** detect `ckpt/<run_id>/latest.pt` and resume from the saved step automatically.
   - **Tokenization** (`run_1`) is quick — just let it finish; or lower the target counts (§6).
   - **Optional cache builds** (`run_5/7`) resume at **shard** granularity (already-written shards are skipped).

So the rule is simply: *reconnect → rerun the two setup cells → rerun the work cell.*

---

## 6. Quick smoke of the *whole* pipeline (recommended first)

Before committing to the long runs, do a tiny end-to-end pass to confirm everything works on your
account. In `run_2` (and `run_3`) override the sizes via the `--set` flags / args:

```python
# run_2 — train only 300 steps, tiny batch
!python scripts/train.py --config siflow/config/mdlm.yaml --set \
    data.tokens_path={BASE}/data/owt_gpt2_256.npy \
    out_dir={BASE}/ckpt/mdlm_smoke run_id=siflow_mdlm_smoke \
    train.total_steps=300 train.batch_size=16 train.micro_batch=16
```
```python
# run_3 — evaluate the smoke checkpoint on few samples, skip MAUVE for speed
!python scripts/evaluate.py --config siflow/config/mdlm.yaml --system siflow \
    --ckpt-dir {BASE}/ckpt/mdlm_smoke --ref-tokens {BASE}/data/owt_gpt2_val.npy \
    --n-samples 64 --k-list 1 4 --no-mauve --out results/mdlm_smoke.json
```
In `run_1`, you can also shrink the corpus: change `N_TRAIN = 200_000` to e.g. `20_000`.
If the smoke pass is green, rerun with the full defaults.

Useful knobs (pass as `--set key=value` to `scripts/train.py`, or args to `scripts/evaluate.py`):

| Knob | Meaning |
|------|---------|
| `train.total_steps` | training length (20k full / a few hundred for smoke) |
| `train.micro_batch` | **lower this if you hit OOM** (16 → 8 → 4) |
| `head.space=prob` / `head.mid_blocks=1` | head ablations |
| `ablation.no_avg_velocity=true` / `ablation.hard_label=true` | method ablations |
| `--n-samples`, `--k-list`, `--no-mauve` | eval cost vs. completeness |

---

## 7. Producing the final paper

After your runs, `MyDrive/siflow/tables_auto.tex` and `MyDrive/siflow/figures/*.pdf` hold the populated
results. Drop them into the paper tree and recompile:

```bash
# locally, or in any LaTeX environment
cp tables_auto.tex   paper/tables_auto.tex
cp figures/*.pdf     paper/figures/
cd paper && latexmk -pdf siflow_aaai.tex
```
Tables 2–4 and the figures fill in automatically (the `.tex` `\input`s `tables_auto.tex` and
`\includegraphics` the figure PDFs). With no results present it still compiles, showing `--`
placeholders and figure-placeholder boxes.

---

## 8. Troubleshooting

- **CUDA OOM** → lower `train.micro_batch` (and `--gen-batch` in eval). DiffusionGemma needs the full
  80 GB; make sure no other notebook is using the GPU.
- **Session keeps disconnecting** → that's expected past ~12 h; just resume (§5). Checkpoints are every 1k steps.
- **`401 / gated repo` on DiffusionGemma** → accept the license on its HF page and run the `login()`
  cell (§2).
- **Dream/Gemma: "no mask token" or "no logits"** → the model card may name a different mask token or
  HF class. Set them in the config, e.g.
  `--set teacher.mask_token="<mask>" teacher.auto_class=AutoModelForMaskedLM`.
- **SDTT baseline fails to install** → it's optional; `run_3` skips it gracefully and the rest of the
  table is unaffected (SDTT is then a *reported* baseline).
- **Drive quota** → skip the optional cache cells in `run_5/7` (train live), or reduce `N_TRAIN`.
- **Want to start clean** → delete `MyDrive/siflow/` (or just the relevant `ckpt/<run_id>/`).
