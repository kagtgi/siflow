"""End-to-end integration on a tiny fake teacher (no model download).

Exercises the real Student.predict (full + reduced), one-/few-step generation,
conditional completion, the teacher samplers, and a real training micro-step
(loss assembly + backward into the head). Runs on CPU.
"""
import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace

from siflow.teacher.base import Teacher
from siflow.head import VelocityHead
from siflow.student import Student
from siflow.schedule import NoiseSchedule
from siflow.support import reduce_to_support
from siflow.sampling import teacher_ancestral_sample, teacher_complete
from siflow import train as T

V, H, MASK = 48, 16, 47


class FakeTeacher(Teacher):
    def __init__(self):
        self.name = "fake"
        self.vocab_size = V
        self.hidden_dim = H
        self.mask_index = MASK
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        g = torch.Generator().manual_seed(0)
        self._E = torch.randn(V, H, generator=g)
        self._W = torch.randn(H, V, generator=g)

    @property
    def embedding_matrix(self):
        return self._E

    @torch.no_grad()
    def logits_and_hidden(self, input_ids):
        input_ids = input_ids.to(self.device)
        hidden = torch.nn.functional.embedding(input_ids, self._E)  # [B,L,H]
        raw = hidden @ self._W                                       # [B,L,V]
        return self.subs(raw, input_ids), hidden


def _student(space="logit"):
    t = FakeTeacher()
    head = VelocityHead(H, t.embedding_matrix, bottleneck=64, space=space)
    return Student(t, head, NoiseSchedule()), t, head


def test_predict_full_and_reduced():
    student, t, _ = _student()
    x = torch.randint(0, V - 1, (3, 12))
    z, h = t.logits_and_hidden(x)
    s = torch.zeros(3)
    tt = torch.ones(3)
    pred = student.predict(z, h, s, tt)
    assert pred.mu_hat.shape == (3, 12, V)
    assert torch.allclose(pred.mu_hat.sum(-1), torch.ones(3, 12), atol=1e-4)
    # reduced support (+ rest bucket)
    z2, _ = t.logits_and_hidden(torch.randint(0, V - 1, (3, 12)))
    support, z_s_red, z_t_red = reduce_to_support(z, z2, m=8)
    predr = student.predict(z_s_red, h, s, tt, support_idx=support)
    assert predr.mu_hat.shape == (3, 12, 9)
    assert torch.allclose(predr.mu_hat.sum(-1), torch.ones(3, 12), atol=1e-4)


def test_generate_and_complete():
    student, t, _ = _student()
    for k in (1, 4):
        out = student.generate(2, 16, k=k)
        assert out.shape == (2, 16)
        assert (out != MASK).all(), "no position should remain masked"
    ids = torch.randint(0, V - 1, (2, 16))
    fill = torch.zeros(2, 16, dtype=torch.bool)
    fill[:, -2:] = True
    comp = student.complete(ids, fill, k=2)
    assert (comp[~fill] == ids[~fill]).all(), "context must be preserved"
    assert (comp != MASK).all()


def test_teacher_samplers():
    t = FakeTeacher()
    out = teacher_ancestral_sample(t, 2, 16, num_steps=8)
    assert out.shape == (2, 16) and (out != MASK).all()
    ids = torch.randint(0, V - 1, (2, 16))
    fill = torch.zeros(2, 16, dtype=torch.bool)
    fill[:, 5:7] = True
    comp = teacher_complete(t, ids, fill, num_steps=4)
    assert (comp[~fill] == ids[~fill]).all()


def _cfg(reduced_m=0):
    return SimpleNamespace(
        data=SimpleNamespace(source="live", reduced_m=reduced_m),
        train=SimpleNamespace(lam_ent=0.1, w_vel=0.1, lam_reg=0.05, w_id=0.0, p0=0.25, p1=0.25),
    )


def _micro(reduced_m):
    student, t, head = _student()
    sched = NoiseSchedule()
    gen = torch.Generator().manual_seed(1)
    data = iter(lambda: torch.randint(0, V - 1, (4, 16)), None)  # endless x0 batches
    loss, parts, mu = T._micro_step(_cfg(reduced_m), student, t, sched, data,
                                    reduced=reduced_m > 0, m=reduced_m, no_avg=False,
                                    beta=1.5, vocab_size=V, mask_index=MASK,
                                    gen=gen, device=torch.device("cpu"))
    assert torch.isfinite(loss), parts
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), "head must receive gradient"


def test_micro_step_full():
    _micro(reduced_m=0)


def test_micro_step_reduced():
    _micro(reduced_m=8)
