# SIFLOW

**One- and few-step diffusion language models via average-velocity distillation on the probability simplex.**

SIFLOW distills a *pretrained* masked diffusion language model (DLM) into a one- or few-step
generator by learning an **average unmasking velocity** on the teacher's predictive simplex.
A lightweight velocity head (~1–3M params) is the only thing trained; the teacher backbone stays
frozen. The secant target `(μ_t − μ_s)/(t − s)` is the constant-velocity field of the **straight
(flat-metric) path** between two simplex points — the discrete-simplex analog of the straightened
conditional paths in Flow Matching / OT-CFM, which is exactly when one-step integration is accurate.

This repo is the full, runnable implementation behind the AAAI paper in [`paper/`](paper/), built to
run on a **single A100-40GB in <12h sessions** via the two step-by-step notebooks in [`notebooks/`](notebooks/).

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
notebooks/         # nb1_mdlm + nb2_large_teachers — 2 notebooks, A100-40GB, <12h each
paper/             # the AAAI LaTeX source; tables/figures auto-fill from results/
tests/             # pytest: SUBS parity, nested-mask nesting, head budget, support mass, losses
```

## Install

```bash
pip install -e .                      # core deps (torch comes from your CUDA build / Colab)
pip install -r requirements-colab.txt # eval extras: mauve-text, sacrebleu, datasets, ...
pytest tests/ -q                      # SUBS, nested masking, head, reduced-support, losses
```

## The notebook pipeline (Colab A100-40GB — just 2 notebooks)

> **New here? Read [`TUTORIAL.md`](TUTORIAL.md)** — a step-by-step guide (the one zip handoff,
> resuming after a timeout, a quick smoke pass, troubleshooting).

All paper results come from **two notebooks**, each fitting **one A100-40GB session in <12h** (no
quantization). **NB1 needs no upload; for NB2 you just upload NB1's output zip and run all.** Every
stage is guarded by an existence check and training auto-resumes from its checkpoint (11h wall-clock
guard), so a session that ends early loses no work. Set `USE_DRIVE=True` to persist on Drive instead
of the zip handoff.

| Notebook | Upload at top | Does | Downloads | Fills |
|---|---|---|---|---|
| `nb1_mdlm` | — | tokenize OWT → train MDLM head → eval (k-sweep + teacher curve + AR + SDTT) → figures → 6 ablations | `nb1_mdlm_outputs.zip` | **Table 2 (MDLM) + Table 3** |
| `nb2_large_teachers` | `nb1_mdlm_outputs.zip` | Dream-7B (-D) then LLaDA-8B (-L), head-only train+eval (teacher freed between), then **regenerate the final combined tables/figures** | `nb2_final_paper_artifacts.zip` | **Table 2 (-D / -L)** |

Then unzip `nb2_final_paper_artifacts.zip` into `paper/` (drops `tables_auto.tex` + `figures/*.pdf`)
and recompile — Tables 2–3 and the figures populate from `results/*.json`.

## Teachers (all real, all downloadable)

| Variant | Teacher | Vocab | Notes |
|---|---|---|---|
| SIFLOW | [`kuleshov-group/mdlm-owt`](https://huggingface.co/kuleshov-group/mdlm-owt) (~170M) | GPT-2 | primary; teacher runs live, exact full-vocab loss. **HF forward returns raw logits — we re-apply SUBS ourselves** (`siflow/teacher/mdlm.py`). |
| SIFLOW-D | [`Dream-org/Dream-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-v0-Base-7B) | Qwen | head-only on Dream's backbone; reduced top-m loss; ~14GB fp16 |
| SIFLOW-L | [`GSAI-ML/LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) | LLaMA | head-only on LLaDA's backbone; reduced top-m loss; ~16GB fp16 (mask id 126336) |

> All three teachers fit a **single A100-40GB in fp16** (no quantization). DiffusionGemma-26B (~50GB)
> needs more than one 40GB card and is deferred to future multi-GPU work (`siflow/config/gemma.yaml`,
> `siflow/teacher/gemma.py` are retained for that).

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
