"""SIFLOW distillation training loop.

Handles three data paths from one code path:

* **MDLM, live, full-vocab** -- teacher runs each step; exact full-V loss.
* **Dream / Gemma, live, reduced-support** -- teacher runs each step; loss on the
  top-``m`` union support + rest bucket (``data.reduced_m > 0``).
* **Dream / Gemma, cached** -- targets + hidden states streamed from a shard cache
  built by ``siflow.cache.build_cache`` (``data.source == "cache"``); the teacher
  backbone is never loaded, only its embedding ``E``.

Ablation switches (``cfg.ablation``): ``no_avg_velocity`` (Di[M]O-style direct
0->1 map, no interval averaging), ``hard_label`` (beta=1, no SATD anneal). The
entropy prior, identity target, and head depth are controlled by their config
fields, so every ablation row is reproducible from a single yaml override.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .head import VelocityHead
from .losses import beta_schedule, satd_kl, secant_mse, mdm_ce
from .masking import nested_masks, entropy_inject, sample_st
from .schedule import NoiseSchedule
from .student import Student
from .support import reduce_to_support
from .utils import ckpt, drive, set_seed, EMA, JsonlLogger, log


# --------------------------------------------------------------------------- #
# construction helpers
# --------------------------------------------------------------------------- #
def _parts(**kw):
    """Detach loss components to plain floats for logging (avoids grad-scalar warning)."""
    out = {}
    for k, v in kw.items():
        out[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
    return out


def _embedding_for_cache(cache_dir: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(os.path.join(cache_dir, "embedding.safetensors"))["E"]


def build_student(cfg, device, teacher=None) -> Student:
    """Build a Student. For cache-mode training pass ``teacher=None`` and the
    embedding is read from the cache; otherwise the teacher provides it."""
    sched = NoiseSchedule(kind=cfg.schedule.kind, eps=float(cfg.schedule.eps))
    if teacher is None and cfg.data.source == "cache":
        E = _embedding_for_cache(cfg.data.cache_dir).to(device)
        hidden_dim = int(E.shape[1])

        class _EmbOnly:  # minimal teacher stub for cache-mode (no backbone)
            pass

        head = VelocityHead(hidden_dim, E, bottleneck=int(cfg.head.bottleneck),
                            time_dim=int(cfg.head.time_dim), space=str(cfg.head.space),
                            mid_blocks=int(getattr(cfg.head, "mid_blocks", 0))).to(device)
        stub = _EmbOnly()
        stub.device = device
        student = Student(stub, head, sched)  # teacher unused in cache-mode predict
        return student

    assert teacher is not None
    head = VelocityHead(teacher.hidden_dim, teacher.embedding_matrix, bottleneck=int(cfg.head.bottleneck),
                        time_dim=int(cfg.head.time_dim), space=str(cfg.head.space),
                        mid_blocks=int(getattr(cfg.head, "mid_blocks", 0))).to(device)
    return Student(teacher, head, sched)


def _lr_lambda(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return fn


# --------------------------------------------------------------------------- #
# one micro-step loss
# --------------------------------------------------------------------------- #
def _identity_term(student, z_s, h_s, s, t, mu_s, mu_t, support_idx, loss_mask, delta=2e-2):
    """MeanFlow-style identity regulariser (optional ablation, prob space).

    Supervises the student prob-velocity ``U = (mu_hat - mu_s)/(t-s)`` toward
    ``v_inst - (t-s) dU/dt`` (stop-grad), with ``v_inst`` proxied by the secant
    and ``dU/dt`` by a finite difference in ``t``.
    """
    dt = (t - s).clamp_min(1e-6).view(-1, 1, 1)
    mu_hat = student.predict(z_s, h_s, s, t, support_idx=support_idx).mu_hat
    U = (mu_hat - mu_s) / dt
    t2 = (t + delta).clamp(max=1.0)
    dt2 = (t2 - s).clamp_min(1e-6).view(-1, 1, 1)
    mu_hat2 = student.predict(z_s, h_s, s, t2, support_idx=support_idx).mu_hat
    U2 = (mu_hat2 - mu_s) / dt2
    with torch.no_grad():
        dUdt = (U2 - U) / delta
        v_inst = (mu_t - mu_s) / dt
        target = v_inst - dt * dUdt
    sq = (U - target).pow(2).sum(-1)  # [B, L]
    if loss_mask is None:
        return sq.mean()
    mask = loss_mask.to(sq.dtype)
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)


def train(cfg) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    sched = NoiseSchedule(kind=cfg.schedule.kind, eps=float(cfg.schedule.eps))
    reduced = int(getattr(cfg.data, "reduced_m", 0)) > 0
    m = int(getattr(cfg.data, "reduced_m", 0))
    abl = getattr(cfg, "ablation", {})
    no_avg = bool(getattr(abl, "no_avg_velocity", False))
    hard_label = bool(getattr(abl, "hard_label", False))

    # ---- teacher / student / data ----
    if cfg.data.source == "cache":
        from .cache import SparseSimplexDataset

        ds = SparseSimplexDataset(cfg.data.cache_dir)
        student = build_student(cfg, device, teacher=None)
        teacher = None
        vocab_size, mask_index = ds.vocab_size, ds.mask_index
        data_iter = ds.batches(int(cfg.train.micro_batch), seed=int(cfg.seed), device=device)
    else:
        from .teacher import build_teacher
        from .data import TokenChunkDataset, infinite_batches

        teacher = build_teacher(cfg, device=device)
        student = build_student(cfg, device, teacher=teacher)
        vocab_size, mask_index = teacher.vocab_size, teacher.mask_index
        tok_ds = TokenChunkDataset(cfg.data.tokens_path)
        data_iter = infinite_batches(tok_ds, int(cfg.train.micro_batch), seed=int(cfg.seed), device=device)

    head = student.head
    log(f"trainable head params: {head.num_trainable()/1e6:.2f}M  (reduced={reduced}, m={m})")

    # ---- optim ----
    total = int(cfg.train.total_steps)
    opt = AdamW(head.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.wd),
                betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(int(cfg.train.warmup), total))
    ema = EMA(head, float(cfg.train.ema_decay))

    out_dir = cfg.out_dir or drive.path("ckpt", str(cfg.run_id))
    start = ckpt.resume(out_dir, head, ema, opt, scheduler, map_location=device)
    logger = JsonlLogger(os.path.join(out_dir, "train_log.jsonl"), echo_every=int(cfg.train.log_every))
    accum = max(1, int(cfg.train.batch_size) // int(cfg.train.micro_batch))
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1234)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for step in range(start, total):
        beta = 1.0 if hard_label else beta_schedule(step, total, float(cfg.train.beta_max),
                                                     float(cfg.train.anneal_frac))
        opt.zero_grad(set_to_none=True)
        logs = {"step": step, "beta": beta, "lr": scheduler.get_last_lr()[0]}
        acc = {"loss": 0.0, "satd": 0.0, "vel": 0.0, "mdm": 0.0, "id": 0.0}

        for _ in range(accum):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                loss, parts, mu_pred = _micro_step(
                    cfg, student, teacher, sched, data_iter, reduced, m, no_avg, beta,
                    vocab_size, mask_index, gen, device)
            (loss / accum).backward()
            for k in acc:
                acc[k] += float(parts.get(k, 0.0)) / accum

        torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg.train.grad_clip))
        opt.step()
        scheduler.step()
        ema.update(head)

        if step % int(cfg.train.log_every) == 0:
            logs.update(acc)
            logger.write(logs, echo_keys=["step", "loss", "satd", "vel", "beta", "lr"])
        if int(cfg.train.ckpt_every) > 0 and (step + 1) % int(cfg.train.ckpt_every) == 0:
            ckpt.save(out_dir, step, head, ema, opt, scheduler, cfg=cfg)

    ckpt.save(out_dir, total - 1, head, ema, opt, scheduler, cfg=cfg)
    log(f"training done -> {out_dir}")
    return out_dir


def _micro_step(cfg, student, teacher, sched, data_iter, reduced, m, no_avg, beta,
                vocab_size, mask_index, gen, device):
    lam_ent = float(cfg.train.lam_ent)
    w_vel = float(cfg.train.w_vel)
    lam_reg = float(cfg.train.lam_reg)
    w_id = float(getattr(cfg.train, "w_id", 0.0))

    if cfg.data.source == "cache":
        batch = next(data_iter)
        x0, s, t = batch["x0"], batch["s"], batch["t"]
        loss_mask = batch["loss_mask"]
        z_s_red, z_t_red, h_s = batch["z_s_red"], batch["z_t_red"], batch["h_s"]
        support = batch["support_idx"]
        if no_avg:  # force the direct 0->1 map even with cached intervals
            s = torch.zeros_like(s)
            t = torch.ones_like(t)
        pred = student.predict(z_s_red, h_s, s, t, support_idx=support)
        mu_t = torch.softmax(z_t_red, dim=-1)
        L_satd = satd_kl(z_t_red, pred.log_mu_hat, beta, loss_mask)
        L_vel = secant_mse(pred.mu_hat, mu_t, loss_mask) if w_vel > 0 else pred.mu_hat.new_zeros(())
        L_mdm = pred.mu_hat.new_zeros(())  # MDM reg is full-vocab only
        L_id = pred.mu_hat.new_zeros(())
        if w_id > 0:
            mu_s = torch.softmax(z_s_red, dim=-1)
            L_id = _identity_term(student, z_s_red, h_s, s, t, mu_s, mu_t, support, loss_mask)
        loss = L_satd + w_vel * L_vel + w_id * L_id
        return loss, _parts(loss=loss, satd=L_satd, vel=L_vel, id=L_id), pred.mu_hat

    # ---- live path (teacher loaded) ----
    x0 = next(data_iter)
    B = x0.shape[0]
    if no_avg:
        s = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
    else:
        s, t = sample_st(B, device, p0=float(cfg.train.p0), p1=float(cfg.train.p1), generator=gen)
    x_s, x_t, keep_s, keep_t, _ = nested_masks(x0, s, t, mask_index, sched, generator=gen)
    x_s = entropy_inject(x_s, keep_s, mask_index, vocab_size, lam_ent, generator=gen)
    loss_mask = ~keep_s

    with torch.no_grad():
        z_s, h_s = teacher.logits_and_hidden(x_s)
        z_t, _ = teacher.logits_and_hidden(x_t)

    if reduced:
        support, z_s_red, z_t_red = reduce_to_support(z_s, z_t, m)
        pred = student.predict(z_s_red, h_s, s, t, support_idx=support)
        mu_t = torch.softmax(z_t_red, dim=-1)
        z_for_kl = z_t_red
    else:
        support = None
        pred = student.predict(z_s, h_s, s, t)
        mu_t = torch.softmax(z_t, dim=-1)
        z_for_kl = z_t

    L_satd = satd_kl(z_for_kl, pred.log_mu_hat, beta, loss_mask)
    L_vel = secant_mse(pred.mu_hat, mu_t, loss_mask) if w_vel > 0 else pred.mu_hat.new_zeros(())

    L_mdm = pred.mu_hat.new_zeros(())
    if lam_reg > 0 and not reduced:
        r = torch.rand(B, device=device, generator=gen)
        ones = torch.ones(B, device=device)
        x_r, x_one, keep_r, _, _ = nested_masks(x0, r, ones, mask_index, sched, generator=gen)
        with torch.no_grad():
            z_r, h_r = teacher.logits_and_hidden(x_r)
        pred_r = student.predict(z_r, h_r, r, ones)
        L_mdm = mdm_ce(pred_r.log_mu_hat, x0, ~keep_r)

    L_id = pred.mu_hat.new_zeros(())
    if w_id > 0:
        mu_s = torch.softmax(z_s_red if reduced else z_s, dim=-1)
        L_id = _identity_term(student, z_s_red if reduced else z_s, h_s, s, t, mu_s, mu_t, support, loss_mask)

    loss = L_satd + w_vel * L_vel + lam_reg * L_mdm + w_id * L_id
    parts = _parts(loss=loss, satd=L_satd, vel=L_vel, mdm=L_mdm, id=L_id)
    return loss, parts, pred.mu_hat
