"""Token-id packing must use a uint wide enough for the tokenizer, or large-vocab
teachers (Dream ~152k, LLaDA ~126k) silently overflow uint16 and corrupt training."""
import pytest

np = pytest.importorskip("numpy")

from siflow.data import _id_dtype  # noqa: E402


class _Tok:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


def test_id_dtype_picks_width_by_vocab():
    assert _id_dtype(_Tok(50257)) == np.uint16     # GPT-2 (MDLM)
    assert _id_dtype(_Tok(65536)) == np.uint16     # boundary still fits
    assert _id_dtype(_Tok(126464)) == np.uint32    # LLaDA
    assert _id_dtype(_Tok(151936)) == np.uint32    # Qwen / Dream


def test_id_dtype_unknown_is_safe():
    class _NoLen:
        vocab_size = 0
    assert _id_dtype(_NoLen()) == np.uint32        # unknown -> widest (never overflows)
