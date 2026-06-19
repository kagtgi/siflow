"""Loss sanity: non-negativity, shapes, and SATD beta schedule endpoints."""
import pytest

torch = pytest.importorskip("torch")

from siflow.losses import satd_kl, secant_mse, mdm_ce, beta_schedule, masked_mean


def test_satd_nonneg_and_zero_at_match():
    z_t = torch.randn(2, 4, 50)
    log_mu = torch.log_softmax(z_t, -1)  # student exactly matches teacher
    assert satd_kl(z_t, log_mu, beta=1.0) < 1e-4  # KL ~ 0 at a perfect match
    other = torch.log_softmax(torch.randn(2, 4, 50), -1)
    assert satd_kl(z_t, other, beta=1.0) >= 0


def test_secant_and_mdm():
    mu_hat = torch.softmax(torch.randn(2, 4, 50), -1)
    mu_t = torch.softmax(torch.randn(2, 4, 50), -1)
    assert secant_mse(mu_hat, mu_t) >= 0
    assert secant_mse(mu_t, mu_t) < 1e-6
    log_mu = torch.log_softmax(torch.randn(2, 4, 50), -1)
    x0 = torch.randint(0, 50, (2, 4))
    assert mdm_ce(log_mu, x0) >= 0


def test_beta_schedule():
    assert abs(beta_schedule(0, 100, 2.0, 0.5) - 2.0) < 1e-6
    assert abs(beta_schedule(100, 100, 2.0, 0.5) - 1.0) < 1e-6
    assert beta_schedule(0, 100, 1.0) == 1.0  # no anneal when beta_max==1


def test_masked_mean():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    m = torch.tensor([[True, False, True]])
    assert abs(float(masked_mean(x, m)) - 2.0) < 1e-6
