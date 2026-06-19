"""Velocity head: param budget and zero-init identity start."""
import pytest

torch = pytest.importorskip("torch")

from siflow.head import VelocityHead
from siflow.student import Student


class _Stub:
    device = torch.device("cpu")


def test_param_budget():
    E = torch.randn(2000, 256)
    head = VelocityHead(256, E, bottleneck=1024)
    n = head.num_trainable()
    assert 0.3e6 < n < 6e6, f"head should be ~1-5M params, got {n/1e6:.2f}M"
    # frozen embedding is not trainable
    assert not head.E.requires_grad


def test_zero_init_identity():
    E = torch.randn(2000, 256)
    head = VelocityHead(256, E, bottleneck=512)
    h = torch.randn(2, 8, 256)
    s = torch.rand(2)
    t = s + 0.3
    U = head(h, s, t)
    assert U.shape == (2, 8, 2000)
    assert torch.allclose(U, torch.zeros_like(U), atol=1e-6), "zero-init -> U=0 at start"

    student = Student(_Stub(), head)
    z_s = torch.randn(2, 8, 2000)
    pred = student.predict(z_s, h, s, t)
    # identity start: mu_hat == softmax(z_s)
    assert torch.allclose(pred.mu_hat, torch.softmax(z_s, -1), atol=1e-5)


def test_reduced_support_shapes():
    E = torch.randn(500, 64)
    head = VelocityHead(64, E, bottleneck=256)
    h = torch.randn(2, 5, 64)
    s = torch.zeros(2)
    t = torch.ones(2)
    support = torch.randint(0, 500, (2, 5, 32))
    U = head(h, s, t, support_idx=support)
    assert U.shape == (2, 5, 32)
