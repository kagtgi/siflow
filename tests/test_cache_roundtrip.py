"""Reduced-support representation must conserve probability mass."""
import pytest

torch = pytest.importorskip("torch")

from siflow.support import reduce_to_support


def test_roundtrip_mass():
    z_s = torch.randn(2, 4, 500)
    z_t = torch.randn(2, 4, 500)
    support, z_s_red, z_t_red = reduce_to_support(z_s, z_t, m=32)
    assert support.shape == (2, 4, 32)
    assert z_s_red.shape == (2, 4, 33)  # m + rest bucket

    mu_full = torch.softmax(z_s, -1)
    mu_red = torch.softmax(z_s_red, -1)
    # reduced distribution sums to 1
    assert torch.allclose(mu_red.sum(-1), torch.ones(2, 4), atol=1e-4)
    # mass on the support tokens matches the full distribution's mass there
    mass_full = mu_full.gather(-1, support).sum(-1)
    mass_red = mu_red[..., :32].sum(-1)
    assert torch.allclose(mass_full, mass_red, atol=1e-4)
    # the rest bucket carries the remaining tail mass
    assert torch.allclose(mu_red[..., -1], 1.0 - mass_full, atol=1e-4)
