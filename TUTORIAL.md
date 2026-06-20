# SIFLOW — Running the 2 notebooks on Colab (step by step)

Every result and figure in the paper comes from **two notebooks**, each on a **single Colab
A100-40GB** in **under 12 hours**, with **no quantization**:

| # | Notebook | Upload at top | Downloads at end | Produces |
|---|----------|---------------|------------------|----------|
| 1 | `nb1_mdlm` | — | `nb1_mdlm_outputs.zip` | MDLM rows of Table 2 + all of Table 3 + figures |
| 2 | `nb2_large_teachers` | `nb1_mdlm_outputs.zip` | `nb2_final_paper_artifacts.zip` | Dream-7B (-D) & LLaDA-8B (-L) rows + **final combined** tables/figures |

**The whole flow is two steps:** run NB1 → it downloads a zip → open NB2, upload that zip, run all.

---

## 0. One-time prerequisites

1. **Colab with an A100** (Pro / Pro+). `Runtime → Change runtime type → A100 GPU`. The free 40GB
   A100 is enough — all three teachers (MDLM-170M, Dream-7B ~14GB, LLaDA-8B ~16GB) fit in fp16.
   Check with `!nvidia-smi`.
2. **A Hugging Face token** (free, read scope) — recommended for the Dream-7B / LLaDA-8B downloads.
   <https://huggingface.co/settings/tokens>. NB2 has a `login()` cell.
3. Somewhere on your computer to keep the one zip between the two notebooks (it's small: the head
   checkpoints are a few MB, results/figures tiny).

---

## 1. Open a notebook in Colab

- NB1 — <https://colab.research.google.com/github/kagtgi/siflow/blob/main/notebooks/nb1_mdlm.ipynb>
- NB2 — <https://colab.research.google.com/github/kagtgi/siflow/blob/main/notebooks/nb2_large_teachers.ipynb>

(Or browse `notebooks/` on GitHub and click **Open in Colab**.)

---

## 2. Notebook 1 — `nb1_mdlm` (no upload needed)

Just **Runtime → Run all**. In order it:

1. Clones + installs (cell 1) and sets `BASE` (cell 2).
2. Smoke: unit tests + an MDLM load + SUBS sanity check.
3. Tokenizes OpenWebText (train + a disjoint val split for MAUVE).
4. Trains the velocity head (frozen MDLM backbone), ~15k steps, resumable.
5. Evaluates SIFLOW (k = 1, 2, 4, 8), the MDLM teacher step-curve (8/32/64/1024), AR GPT-2, and
   optionally SDTT@8; builds the figures.
6. Runs the 6-variant ablation suite (Table 3).
7. **Auto-downloads `nb1_mdlm_outputs.zip`** (results + figures + tables + the MDLM head).

Keep that zip. That's the only thing NB2 needs from NB1.

---

## 3. Notebook 2 — `nb2_large_teachers` (upload NB1's zip, run all)

1. **Runtime → Run all.** When the **“Import the previous part”** cell runs, a file picker opens —
   select **`nb1_mdlm_outputs.zip`**.
2. It then trains + evaluates **Dream-7B** (`-D`), frees the GPU, trains + evaluates **LLaDA-8B**
   (`-L`) — each fits 40GB on its own — and regenerates the **final combined** tables/figures from
   *all* results (the MDLM rows you uploaded plus the new -D / -L rows).
3. **Auto-downloads `nb2_final_paper_artifacts.zip`** — the complete `tables_auto.tex` + `figures/`.

That's it. Two notebooks, one upload.

---

## 4. Resuming after a timeout (nothing is lost)

Every stage is guarded by an existence check and training resumes from its checkpoint (there's an
11-hour wall-clock guard inside training too). If a session ends early:

1. The partial output zip still downloads (it includes the latest checkpoints + whatever finished).
2. Reopen the **same** notebook, run cells 1–2, and at the import cell upload that notebook's **own**
   latest zip (NB2 also re-upload `nb1_mdlm_outputs.zip`).
3. Run all again — finished stages print “skip (done)” and training picks up where it stopped.

Prefer no zips at all? Set `USE_DRIVE = True` in cell 2 of both notebooks; everything persists under
`MyDrive/siflow/` and the import/download steps become no-ops.

---

## 5. Quick end-to-end smoke first (recommended)

Validate the whole flow cheaply before the long runs. In **NB1** set `N_TRAIN = 20_000` and
`STEPS = 300` (and `ABL_STEPS = 200`); in **NB2** set `DREAM_STEPS = LLADA_STEPS = 300` and the
`N_*_TOK` to `5_000`. If it's green end-to-end, reset to the defaults and run for real.

---

## 6. Build the final paper

`nb2_final_paper_artifacts.zip` holds `tables_auto.tex` and `figures/*.pdf`:

```bash
unzip nb2_final_paper_artifacts.zip -d paper/      # drops tables_auto.tex + figures/ into paper/
cd paper && latexmk -pdf siflow_aaai.tex
```

Tables 2–3 (MDLM, Dream-7B `-D`, LLaDA-8B `-L`) and the figures fill in automatically. (The paper
also compiles with no results — it shows `--` placeholders and figure-placeholder boxes.)

---

## 7. Troubleshooting

- **CUDA OOM** → lower `train.micro_batch` (Dream/LLaDA default 4 → 2) via `--set train.micro_batch=2`
  on the relevant `!python scripts/train.py` line; lower `--gen-batch` in eval.
- **Stops early** → expected near the limit; just re-run (see §4). Stages skip / training resumes.
- **`401 / gated repo`** → run the `login()` cell with your HF token.
- **LLaDA mask token** → it's the fixed constant `126336` (already set in `siflow/config/llada.yaml`;
  no tokenizer mask id exists).
- **Dream “no mask token” / “no logits”** → set them in `siflow/config/dream.yaml`, e.g.
  `--set teacher.mask_token="<mask>" teacher.auto_class=AutoModelForMaskedLM`.
- **SDTT install fails (NB1)** → it's optional and skips itself; the rest of the table is unaffected.
- **DiffusionGemma-26B** → ~50GB, does **not** fit one 40GB card; it's deferred to future multi-GPU
  work (`gemma.yaml` / `gemma.py` are kept for that).
