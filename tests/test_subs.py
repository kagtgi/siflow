"""SUBS parameterization: the most error-prone MDLM integration point."""
import pytest

torch = pytest.importorskip("torch")

from siflow.teacher.base import Teacher


class _Dummy(Teacher):
    def __init__(self):
        self.name = "dummy"
        self.mask_index = 5
        self.vocab_size = 6
        self.hidden_dim = 4

    def logits_and_hidden(self, input_ids):  # pragma: no cover - unused
        raise NotImplementedError

    @property
    def embedding_matrix(self):  # pragma: no cover - unused
        raise NotImplementedError


def test_subs_structure():
    d = _Dummy()
    raw = torch.randn(1, 3, 6)
    ids = torch.tensor([[2, 5, 5]])  # pos0 = revealed token 2; pos1,2 = masked
    out = d.subs(raw, ids)
    neg = torch.finfo(raw.dtype).min

    # mask column is killed everywhere
    assert (out[..., 5] == neg).all()
    # revealed position is one-hot at the observed token
    assert out[0, 0, 2] == 0.0
    others = [i for i in range(6) if i != 2]
    assert (out[0, 0, others] == neg).all()
    # masked positions keep their raw (non-mask) logits
    assert torch.allclose(out[0, 1, :5], raw[0, 1, :5])
    # probabilities never put mass on the mask token
    p = torch.softmax(out.float(), dim=-1)
    assert p[..., 5].max() < 1e-6
