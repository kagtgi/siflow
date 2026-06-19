"""Simplex-trajectory straightness (Figure F1; validates the flat-metric claim).

Along one nested-mask reverse path of a clean sequence, the teacher's predictive
point ``mu_tau`` moves from the all-mask prediction (tau=0) to the one-hot data
point (tau=1; SUBS pins revealed positions). The path-length ratio

    R = (sum_i || mu_{tau_{i+1}} - mu_{tau_i} ||) / || mu_1 - mu_0 ||   (>= 1)

is ~1 for a straight (flat-geodesic) path, which is exactly when one-step
integration is accurate. A large R means the teacher trajectory is curved and
the one-step gap will be large -- the single most important diagnostic for the
method's premise.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..masking import reveal_ranks
from ..schedule import NoiseSchedule


@torch.no_grad()
def path_length_ratio(
    teacher,
    token_dataset,
    schedule: NoiseSchedule = None,
    n_examples: int = 256,
    n_grid: int = 16,
    batch_size: int = 16,
    seed: int = 0,
    device=None,
    return_pca: bool = True,
) -> Dict[str, object]:
    schedule = schedule or NoiseSchedule()
    device = device or teacher.device
    L = token_dataset.seq_len
    rng = np.random.default_rng(seed)
    idx_all = rng.permutation(len(token_dataset))[:n_examples]
    taus = torch.linspace(0.0, 1.0, n_grid + 1)

    ratios = []
    pca_traj = None
    for b in range(0, len(idx_all), batch_size):
        idx = idx_all[b: b + batch_size]
        x0 = token_dataset.batch(idx, device=device)
        B = x0.shape[0]
        g = torch.Generator(device=device).manual_seed(seed + b)
        rank = reveal_ranks(B, L, device, generator=g)  # one shared reveal order per example
        mus = []
        for tau in taus.tolist():
            keep_n = schedule.n_keep(tau, L)
            keep = rank < keep_n
            x_tau = torch.where(keep, x0, torch.full_like(x0, teacher.mask_index))
            mus.append(teacher.mu(x_tau).float())        # [B, L, V]
        mus = torch.stack(mus, 0)                         # [G+1, B, L, V]
        diffs = (mus[1:] - mus[:-1]).norm(dim=-1)         # [G, B, L]
        path = diffs.sum(0)                               # [B, L]
        chord = (mus[-1] - mus[0]).norm(dim=-1).clamp_min(1e-6)  # [B, L]
        ratios.append((path / chord).flatten().cpu().numpy())
        if return_pca and pca_traj is None:
            # 2D PCA of one example's first masked position's trajectory
            traj = mus[:, 0, 0, :].cpu().numpy()          # [G+1, V]
            traj = traj - traj.mean(0, keepdims=True)
            try:
                u, s_, vt = np.linalg.svd(traj, full_matrices=False)
                pca_traj = (u[:, :2] * s_[:2]).tolist()
            except Exception:  # noqa: BLE001
                pca_traj = None

    r = np.concatenate(ratios) if ratios else np.array([1.0])
    return {
        "ratio_mean": float(r.mean()),
        "ratio_std": float(r.std()),
        "ratio_median": float(np.median(r)),
        "ratios": r.tolist(),
        "pca_trajectory": pca_traj,
        "teacher": teacher.name,
    }
