"""Checkpoint save/resume + EMA — the Colab cross-session resume guarantee."""
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from siflow.utils import ckpt as ckpt_io
from siflow.utils.ema import EMA


def test_ckpt_resume_roundtrip(tmp_path):
    head = nn.Linear(8, 8)
    ema = EMA(head, 0.9)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    # one optimisation step so optimizer/ema state is non-trivial
    loss = head(torch.randn(4, 8)).pow(2).mean()
    loss.backward()
    opt.step()
    sched.step()
    ema.update(head)
    ckpt_io.save(str(tmp_path), step=41, head=head, ema=ema, optimizer=opt, scheduler=sched)

    head2 = nn.Linear(8, 8)
    ema2 = EMA(head2, 0.9)
    opt2 = torch.optim.AdamW(head2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 1.0)
    start = ckpt_io.resume(str(tmp_path), head2, ema2, opt2, sched2)

    assert start == 42  # resumes at saved step + 1
    for p, q in zip(head.parameters(), head2.parameters()):
        assert torch.allclose(p, q)
    for k in ema.shadow:
        assert torch.allclose(ema.shadow[k], ema2.shadow[k])


def test_resume_no_ckpt(tmp_path):
    head = nn.Linear(4, 4)
    assert ckpt_io.resume(str(tmp_path), head) == 0  # fresh start
