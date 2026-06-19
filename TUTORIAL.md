# SIFLOW — Running the notebooks on Colab (step by step)

Run the whole pipeline on a **single Colab A100-80GB**, one notebook per part. Each part is
self-contained: it **downloads a `.zip` of its output** at the end, and the next part **uploads that
zip** to continue. No Google Drive needed (there's a one-line switch if you prefer Drive).

Every part is designed to finish in **well under one Colab session**. Training parts also have an
**11-hour guard**: if they get close to the limit they checkpoint and stop cleanly — you just re-run
and they resume.

---

## 0. One-time prerequisites

1. **Colab with an A100-80GB** (Pro / Pro+ / Enterprise). In each notebook:
   `Runtime → Change runtime type → A100 GPU` (+ High-RAM if offered). Check with `!nvidia-smi`.
2. **A Hugging Face token** (free) — needed for the **gated DiffusionGemma** weights (open its model
   page once and click *Agree and access*), and recommended for Dream. Token:
   <https://huggingface.co/settings/tokens> (read scope). The run_5–8 notebooks have a `login()` cell.
3. A place on your computer to keep the downloaded `.zip` files between parts. They're small
   (data ~100 MB, head checkpoints a few MB, results/figures tiny).

---

## 1. Open a notebook in Colab

Click (or swap the filename in the URL):

- run_0 — <https://colab.research.google.com/github/kagtgi/siflow/blob/main/notebooks/run_0_smoke.ipynb>
- run_1 … run_8 — same URL with `run_1_mdlm_data_cache.ipynb`, `run_2_mdlm_train.ipynb`, … ,
  `run_8_gemma_train_eval.ipynb`.

(Or browse `notebooks/` on GitHub and use **“Open in Colab”**.)

---

## 2. Every notebook has the same shape

1. **Cell 1 — Clone + install** (~2 min).
2. **Cell 2 — Where artifacts live.** Defaults to a local folder `BASE=/content/artifacts` with the
   zip flow. To use Drive instead, set `USE_DRIVE = True` here — then the import/download steps below
   become no-ops and everything persists in `MyDrive/siflow/`.
3. *(run_5–8 only)* **Hugging Face `login()`** — paste your token.
4. **“Import the previous part(s)”** — a markdown box lists exactly which zip(s) to upload, then a
   cell opens a file picker. Select all the listed zips at once.
5. **Work cells** — run top to bottom.
6. **Last cell — Save + auto-download** — zips this part's output and downloads it to your browser.

So the loop is always: **run all cells → a zip downloads → upload it into the next notebook.**

---

## 3. The parts, in order

| # | Notebook | ~Time | Upload at top | Downloads at end |
|---|----------|-------|---------------|------------------|
| 0 | `run_0_smoke` | 15 min | — | — |
| 1 | `run_1_mdlm_data_cache` | 1–2 h | — | **`run_1_data.zip`** |
| 2 | `run_2_mdlm_train` | 3–4 h | `run_1_data.zip` | **`run_2_mdlm_ckpt.zip`** |
| 3 | `run_3_mdlm_eval_figures` | 4–6 h | `run_1_data.zip`, `run_2_mdlm_ckpt.zip` | **`run_3_results.zip`** |
| 4 | `run_4_mdlm_ablations` | 5–7 h | `run_1_data.zip` (+`run_3_results.zip`) | **`run_4_ablations.zip`** |
| 5 | `run_5_dream_cache` | ~1 h | — | **`run_5_dream_data.zip`** |
| 6 | `run_6_dream_train_eval` | 6–8 h | `run_5_dream_data.zip` | **`run_6_dream.zip`** |
| 7 | `run_7_gemma_cache` | ~1.5 h | — | **`run_7_gemma_data.zip`** |
| 8 | `run_8_gemma_train_eval` | ~7 h | `run_7_gemma_data.zip` + `run_3_results.zip` + `run_4_ablations.zip` + `run_6_dream.zip` | **`run_8_final_paper_artifacts.zip`** |

**Minimum path to the primary result** (AAAI Table 2 + figures): **0 → 1 → 2 → 3**, then **4** for the
ablation table. Parts **5–8** add the large-teacher SIFLOW-D / SIFLOW-G rows and can be done later.

> Why run_8 imports several zips: it regenerates the **final** tables/figures from *all* the results
> you collected, so feed it every results zip you've made. Each part's tables only show what it has
> seen so far — that's fine; run_8 produces the complete version.

---

## 4. Resuming after a timeout (the important part)

Nothing is lost. If a part stops early (Colab limit or the 11h guard):

1. The **partial output zip still downloads** (it includes the latest checkpoint).
2. Reopen the **same** notebook in a fresh session, run cells 1–2, and in the **import** cell upload
   that part's **own** zip (e.g. re-upload `run_2_mdlm_ckpt.zip`) — plus the usual inputs.
3. Re-run the work cell. Training detects the checkpoint and **resumes from where it stopped**.

(With `USE_DRIVE=True` you skip the upload entirely — it resumes straight from Drive.)

---

## 5. Do a tiny end-to-end smoke first (recommended)

Validate the whole flow on your account before the long runs:

- **run_1:** set `N_TRAIN = 20_000` (instead of 200_000).
- **run_2:** edit the train command to `train.total_steps=300 train.batch_size=16 train.micro_batch=16`.
- **run_3:** add `--n-samples 64 --k-list 1 4 --no-mauve` to the SIFLOW eval (and skip the teacher 1024
  step if you want it faster).

If green, rerun with the defaults. Handy `--set` knobs for `scripts/train.py`:

| Knob | Meaning |
|------|---------|
| `train.total_steps` | training length |
| `train.max_hours` | wall-clock guard (default 11.0) |
| `train.micro_batch` | **lower if you hit OOM** (16 → 8 → 4) |
| `ablation.no_avg_velocity=true` / `ablation.hard_label=true` / `head.space=prob` / `head.mid_blocks=1` | ablations |

---

## 6. Build the final paper

`run_8_final_paper_artifacts.zip` contains `tables_auto.tex` and `figures/*.pdf`. Unzip it into the
repo's `paper/` folder and recompile:

```bash
unzip run_8_final_paper_artifacts.zip -d paper/      # drops tables_auto.tex + figures/ into paper/
cd paper && latexmk -pdf siflow_aaai.tex
```
Tables 2–4 and the figures fill in automatically. (The paper also compiles with no results — it shows
`--` placeholders and figure-placeholder boxes.)

---

## 7. Troubleshooting

- **CUDA OOM** → lower `train.micro_batch` (and `--gen-batch` in eval). DiffusionGemma needs the full 80 GB.
- **Stops early** → expected near the limit; resume per §4. Checkpoints are every 1k steps.
- **`401 / gated repo` (DiffusionGemma)** → accept the license on its HF page, run the `login()` cell.
- **Dream/Gemma “no mask token” / “no logits”** → set them in the config, e.g.
  `--set teacher.mask_token="<mask>" teacher.auto_class=AutoModelForMaskedLM`.
- **SDTT install fails (run_3)** → it's optional and skips itself; the rest of the table is unaffected.
- **Lost a zip** → just re-run the part that produced it (they're deterministic given the data).
- **Prefer Drive over zips** → set `USE_DRIVE = True` in cell 2 of every notebook; import/download
  steps become no-ops and everything persists under `MyDrive/siflow/`.
