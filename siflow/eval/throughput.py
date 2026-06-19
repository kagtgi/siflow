"""Throughput (tokens/s) and NFE accounting.

NFE counts student forward passes. One SIFLOW step = one backbone pass + one
velocity-head pass; we report NFE == k (the head pass is included in the
wall-clock tok/s, addressing the reviewer note that the head must be counted).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def student_throughput(student, length: int, k: int, batch_size: int = 16,
                       n_iters: int = 5, warmup: int = 1) -> Dict[str, float]:
    for _ in range(warmup):
        student.generate(batch_size, length, k=k)
    _sync()
    t0 = time.time()
    for _ in range(n_iters):
        student.generate(batch_size, length, k=k)
    _sync()
    dt = (time.time() - t0) / n_iters
    toks = batch_size * length
    return {"tok_per_s": toks / dt, "latency_ms": dt * 1e3, "nfe": k}


@torch.no_grad()
def teacher_throughput(teacher, sampler, length: int, num_steps: int, batch_size: int = 16,
                       n_iters: int = 3, warmup: int = 1, schedule=None) -> Dict[str, float]:
    for _ in range(warmup):
        sampler(teacher, batch_size, length, num_steps, schedule=schedule)
    _sync()
    t0 = time.time()
    for _ in range(n_iters):
        sampler(teacher, batch_size, length, num_steps, schedule=schedule)
    _sync()
    dt = (time.time() - t0) / n_iters
    toks = batch_size * length
    return {"tok_per_s": toks / dt, "latency_ms": dt * 1e3, "nfe": num_steps}
