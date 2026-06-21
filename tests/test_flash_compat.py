"""The flash_attn shim must be numerically faithful: SDPA = exact attention, and
the rotary must be the standard rotate-half RoPE (flash_attn's interleaved=False).
A wrong shim would silently corrupt MDLM's distillation targets, so we check both
against independent references."""
import math

import pytest

torch = pytest.importorskip("torch")

from siflow.flash_compat import (  # noqa: E402
    ensure_flash_attn,
    flash_attn_varlen_qkvpacked_func,
    apply_rotary_emb_qkv_,
)


def test_ensure_registers_importable_shim():
    kind = ensure_flash_attn()
    assert kind in ("real", "shim", "already-shim")
    import flash_attn  # noqa: F401
    import flash_attn.layers.rotary  # noqa: F401
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as f  # noqa: F401


def _ref_attention(qkv, B, L):
    # qkv: [B*L, 3, H, D]; full bidirectional softmax attention per (batch, head).
    H, D = qkv.shape[2], qkv.shape[3]
    x = qkv.view(B, L, 3, H, D)
    q, k, v = x[:, :, 0], x[:, :, 1], x[:, :, 2]            # [B, L, H, D]
    scores = torch.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(D)
    attn = scores.softmax(dim=-1)
    out = torch.einsum("bhlm,bmhd->blhd", attn, v)          # [B, L, H, D]
    return out.reshape(B * L, H, D)


def test_varlen_qkvpacked_matches_exact_attention():
    torch.manual_seed(0)
    B, L, H, D = 3, 7, 4, 16
    qkv = torch.randn(B * L, 3, H, D, dtype=torch.float64)
    cu = torch.arange(0, (B + 1) * L, L, dtype=torch.int32)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, L, causal=False)
    ref = _ref_attention(qkv, B, L)
    assert out.shape == (B * L, H, D)
    assert torch.allclose(out, ref, atol=1e-8), (out - ref).abs().max()


def _ref_rope(x, cos, sin):
    # independent complex-rotation reference; x: [B, L, H, D], cos/sin: [L, D/2]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def test_rotary_matches_rotate_half_and_leaves_v():
    torch.manual_seed(0)
    B, L, H, D = 2, 5, 3, 8
    qkv = torch.randn(B, L, 3, H, D, dtype=torch.float64)
    orig = qkv.clone()
    pos = torch.arange(L, dtype=torch.float64)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    ang = torch.outer(pos, inv_freq)                        # [L, D/2]
    cos, sin = ang.cos(), ang.sin()
    apply_rotary_emb_qkv_(qkv, cos, sin)                    # in-place on q, k
    assert torch.allclose(qkv[:, :, 0], _ref_rope(orig[:, :, 0], cos, sin), atol=1e-10)
    assert torch.allclose(qkv[:, :, 1], _ref_rope(orig[:, :, 1], cos, sin), atol=1e-10)
    assert torch.allclose(qkv[:, :, 2], orig[:, :, 2]), "rotary must leave v untouched"
    # zero angle -> identity
    qkv2 = orig.clone()
    apply_rotary_emb_qkv_(qkv2, torch.ones_like(cos), torch.zeros_like(sin))
    assert torch.allclose(qkv2[:, :, :2], orig[:, :, :2], atol=1e-10)
