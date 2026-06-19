# SIFLOW

**One- and few-step diffusion language models via average-velocity distillation on the probability simplex.**

SIFLOW distills a *pretrained* masked diffusion language model (DLM) into a one- or few-step
generator by learning an **average unmasking velocity** on the teacher's predictive simplex.
A lightweight velocity head (~1–3M params) is the only thing trained; the teacher backbone stays
frozen. The secant target `(μ_t − μ_s)/(t − s)` is the constant-velocity field of the **straight
(flat-metric) path** between two simplex points — the discrete-simplex analog of the straightened
conditional paths in Flow Matching / OT-CFM, which is exactly when one-step integration is accurate.

This repo is the full, runnable implementation behind the AAAI paper in [`paper/`](paper/), built to
run on a **single A100-80GB in <12h sessions** via the step-by-step notebooks in [`notebooks/`](notebooks/).

---

## What's here

```
siflow/            # the library
  teacher/         # frozen teachers: MDLM (reimplements SUBS!), Dream-7B, DiffusionGemma
  schedule.py      # noise schedule (loglinear/linear/cosine)
  masking.py       # nested masking (path consistency) + entropy-injected prior
  head.py          # VelocityHead (logit-space; reuses the frozen embedding as un-embed)
  student.py       # frozen teacher + head; one-/few-step self-conditioned generation
  losses.py        # SATD (annealed-temperature KD) + secant MSE + MDM regulariser
  support.py       # reduced top-m support for large-vocab teachers (Dream/Gemma)
  sampling.py      # teacher ancestral sampler (step-curve baseline)
  cache/           # optional offline simplex cache (build + dataset)
  eval/            # Gen-PPL (gpt2-large), MAUVE, LAMBADA, diversity, throughput/NFE
  analysis/        # simplex-trajectory straightness, per-position entropy, training curves
  train.py         # distillation loop (live MDLM / live-reduced D/G / cached D/G)
scripts/           # train, build_cache, evaluate, make_tables, make_figures (CLI)
notebooks/         # run_0 .. run_8 — clone→data→train→eval→figures, resumable on Drive
paper/             # the AAAI LaTeX source; tables/figures auto-fill from results/
tests/             # pytest: SUBS parity, nested-mask nesting, head budget, support mass, losses
```

## Install

```bash
pip install -e .                      # core deps (torch comes from your CUDA build / Colab)
pip install -r requirements-colab.txt # eval extras: mauve-text, sacrebleu, datasets, ...
pytest tests/ -q                      # SUBS, nested masking, head, reduced-support, losses
```

## The notebook pipeline (Colab A100-80GB)

> **New here? Read [`TUTORIAL.md`](TUTORIAL.md)** — a step-by-step guide to running the notebooks
> sequentially (Drive output saving, resuming after timeouts, a quick smoke pass, troubleshooting).

Each notebook is one session; it clones the repo, mounts Drive, does its part, and saves artifacts to
`MyDrive/siflow/` so the next notebook resumes. Long runs checkpoint to Drive and survive timeouts.

| Notebook | Does | Fills |
|---|---|---|
| `run_0_smoke` | unit tests + MDLM load + 1-step generate | — |
| `run_1_mdlm_data_cache` | tokenize OpenWebText (train + disjoint val) | — |
| `run_2_mdlm_train` | train the MDLM velocity head (20k steps, resumable) | — |
| `run_3_mdlm_eval_figures` | SIFLOW k-sweep + teacher curve + AR + SDTT; figures | **Table 2** |
| `run_4_mdlm_ablations` | retrain/eval ablation variants | **Table 3** |
| `run_5_dream_cache` | Dream-7B setup + tokenize (optional cache) | — |
| `run_6_dream_train_eval` | SIFLOW-D head-only train + eval | Table 2 (-D) |
| `run_7_gemma_cache` | DiffusionGemma setup + tokenize (optional cache) | — |
| `run_8_gemma_train_eval` | SIFLOW-G train + eval + **regenerate all tables/figures** | Table 2 (-G) |

After the runs, drop `paper/tables_auto.tex` and `paper/figures/*.pdf` into the paper tree and
recompile — the tables and figures populate from `results/*.json`.

## Teachers (all real, all downloadable)

| Variant | Teacher | Vocab | Notes |
|---|---|---|---|
| SIFLOW | [`kuleshov-group/mdlm-owt`](https://huggingface.co/kuleshov-group/mdlm-owt) (~170M) | GPT-2 | primary; teacher runs live, exact full-vocab loss. **HF forward returns raw logits — we re-apply SUBS ourselves** (`siflow/teacher/mdlm.py`). |
| SIFLOW-D | [`Dream-org`](https://huggingface.co/Dream-org) Dream-7B | Qwen | head-only on Dream's backbone; reduced top-m loss |
| SIFLOW-G | `google/diffusiongemma-26B-A4B-it` (MoE) | Gemma | head-only; ~50GB fp16 fits A100-80GB |

> **Head-only, not cross-tokenizer.** The three teachers use three different tokenizers, so a single
> "170M student distilled from Dream-7B" (regressing one vocabulary's simplex into another's) is
> ill-posed. SIFLOW-D/-G therefore train a head on **each teacher's own backbone + tokenizer** —
> "architecture-agnostic" is validated across three architectures, not via cross-vocab regression.

## Honesty notes

- **Baselines.** Measured here (single GPU, identical settings): MDLM teacher step-curve, AR GPT-2,
  our Di[M]O-style ablation, and SDTT@8 when its checkpoint is available. T3D / IMDM / FMLM / DLM-One
  use **reported** numbers from their papers (daggered in the tables) — not reproduced here.
- **Theory.** The paper states the secant = the constant-velocity geodesic of the **flat/Euclidean**
  simplex metric (= straight Flow-Matching conditional path), *not* a `W₂`-OT geodesic; a genuine `W₂`
  statement needs a token ground metric and is left as future work.
- **Reproducibility.** Every reported number/figure is regenerated from `results/*.json` by
  `scripts/make_tables.py` / `scripts/make_figures.py` — no hand-entered values.

## License

MIT (this code). Teacher checkpoints retain their own licenses.
