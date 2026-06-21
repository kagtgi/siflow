"""Pure-PyTorch fallback for ``flash_attn`` (so notebooks "just run all").

Several teacher backbones load remote modeling code that does a **bare top-level**
``import flash_attn`` with no fallback (notably ``kuleshov-group/mdlm-owt``, whose
DiT calls ``flash_attn_varlen_qkvpacked_func`` and ``layers.rotary``). On Colab
``flash_attn`` is not installed and building it is slow/fragile, which breaks a
plain *Run all*.

FlashAttention computes **exact** attention, and so does ``torch`` SDPA (identical
``1/sqrt(d)`` scale); MDLM's rotary is the standard rotate-half RoPE. So a small
SDPA + rotate-half shim is **numerically faithful** -- the distillation targets are
the same as with the real kernels -- while needing no CUDA build.

:func:`ensure_flash_attn` registers the shim in ``sys.modules`` *only when the real
package is unavailable*. Teachers call it before ``from_pretrained`` so the remote
code's ``import flash_attn`` (and ``transformers``' static ``check_imports``) resolve
to the shim. If a real ``flash_attn`` is importable, it is used unchanged.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import sys
import types

import torch
import torch.nn.functional as F

__all__ = ["ensure_flash_attn"]


# --------------------------------------------------------------------------- #
# core ops (exact attention via SDPA; standard rotate-half RoPE)
# --------------------------------------------------------------------------- #
def _sdpa(q, k, v, causal, scale):
    """q,k,v: [B, H, L, D]; exact scaled-dot-product attention."""
    try:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal, scale=scale)
    except TypeError:  # very old torch without the `scale` kwarg
        d = q.shape[-1]
        return F.scaled_dot_product_attention(
            q * (scale * math.sqrt(d)), k, v, dropout_p=0.0, is_causal=causal)


def _attn_packed(qkv, causal, softmax_scale):
    """qkv: [B, L, 3, H, D] -> out [B, L, H, D] (exact full/causal attention)."""
    d = qkv.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # each [B, H, L, D]
    out = _sdpa(q, k, v, causal, scale)                          # [B, H, L, D]
    return out.transpose(1, 2).contiguous()                      # [B, L, H, D]


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None, causal=False, **_):
    """qkv: [total_tokens, 3, H, D]; segments given by ``cu_seqlens`` (int tensor).
    Returns [total_tokens, H, D]. Exact attention via SDPA."""
    total, _three, h, d = qkv.shape
    cu = [int(x) for x in (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens)]
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    if lengths and len(set(lengths)) == 1:           # equal-length: one batched SDPA
        L = lengths[0]
        out = _attn_packed(qkv.view(len(lengths), L, 3, h, d), causal, softmax_scale)
        return out.reshape(total, h, d)
    parts = []                                       # ragged: per-segment
    for i in range(len(lengths)):
        seg = qkv[cu[i]:cu[i + 1]].unsqueeze(0)      # [1, Li, 3, H, D]
        parts.append(_attn_packed(seg, causal, softmax_scale).squeeze(0))  # [Li, H, D]
    return torch.cat(parts, 0) if parts else qkv[:, 0]


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """qkv: [B, L, 3, H, D] -> [B, L, H, D]."""
    return _attn_packed(qkv, causal, softmax_scale)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """q,k,v: [B, L, H, D] -> [B, L, H, D]."""
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    out = _sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal, scale)
    return out.transpose(1, 2).contiguous()


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """Variable-length q,k,v: [total, H, D] each. Equal-length fast path + ragged fallback."""
    d = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
    cq = [int(x) for x in (cu_seqlens_q.tolist() if torch.is_tensor(cu_seqlens_q) else cu_seqlens_q)]
    ck = [int(x) for x in (cu_seqlens_k.tolist() if torch.is_tensor(cu_seqlens_k) else cu_seqlens_k)]
    parts = []
    for i in range(len(cq) - 1):
        qs = q[cq[i]:cq[i + 1]].transpose(0, 1).unsqueeze(0)   # [1, H, Lq, D]
        ks = k[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)
        vs = v[ck[i]:ck[i + 1]].transpose(0, 1).unsqueeze(0)
        o = _sdpa(qs, ks, vs, causal, scale).squeeze(0).transpose(0, 1)  # [Lq, H, D]
        parts.append(o)
    return torch.cat(parts, 0) if parts else q


# -- rotary (rotate-half, the flash_attn default `interleaved=False`) --------- #
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    """x: [B, L, H, D]; cos/sin: [L, D/2] -> rotated x (rotate-half)."""
    cos_f = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(x.dtype)  # [1, L, 1, D]
    sin_f = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(x.dtype)
    return x * cos_f + _rotate_half(x) * sin_f


def apply_rotary_emb_qkv_(qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False, **_):
    """In-place rotary on q (idx 0) and k (idx 1) of qkv [B, L, 3, H, D]; v untouched.
    cos/sin: [L, D/2]. Matches ``flash_attn.layers.rotary.apply_rotary_emb_qkv_``."""
    if interleaved:
        raise NotImplementedError("flash_attn shim supports rotate-half (interleaved=False) only")
    cos_k = cos if cos_k is None else cos_k
    sin_k = sin if sin_k is None else sin_k
    qkv[:, :, 0] = _rope(qkv[:, :, 0], cos, sin)
    qkv[:, :, 1] = _rope(qkv[:, :, 1], cos_k, sin_k)
    return qkv


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, **_):
    """x: [B, L, H, D]; cos/sin: [L, D/2]."""
    if interleaved:
        raise NotImplementedError("flash_attn shim supports rotate-half (interleaved=False) only")
    return _rope(x, cos, sin)


# -- bert_padding helpers (some Qwen/LLaMA flash paths import these) ----------- #
def _index_first_axis(x, indices):
    return x[indices.long()]


def _unpad_input(hidden_states, attention_mask):
    """hidden_states: [B, L, ...]; attention_mask: [B, L] (1=keep)."""
    mask = attention_mask.bool()
    seqlens = mask.sum(dim=-1, dtype=torch.int32)              # [B]
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    cu = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    return flat[indices], indices, cu, int(seqlens.max().item())


def _pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    out[indices.long()] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])


# --------------------------------------------------------------------------- #
def _make_mod(name: str) -> types.ModuleType:
    """Create a ModuleType whose __spec__ is set so importlib.util.find_spec()
    doesn't raise ValueError: <name>.__spec__ is None (Python 3.12+)."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


def _build_shim() -> types.ModuleType:
    fa = _make_mod("flash_attn")
    fa.__siflow_shim__ = True
    fa.__version__ = "2.6.3"  # plausible version for any packaging.version checks
    fa.flash_attn_func = flash_attn_func
    fa.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func

    fai = _make_mod("flash_attn.flash_attn_interface")
    for fn in (flash_attn_func, flash_attn_qkvpacked_func,
               flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func):
        setattr(fai, fn.__name__, fn)
    fa.flash_attn_interface = fai

    layers = _make_mod("flash_attn.layers")
    rotary = _make_mod("flash_attn.layers.rotary")
    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    rotary.apply_rotary_emb = apply_rotary_emb
    rotary.apply_rotary_emb_func = apply_rotary_emb
    layers.rotary = rotary
    fa.layers = layers

    bert = _make_mod("flash_attn.bert_padding")
    bert.unpad_input = _unpad_input
    bert.pad_input = _pad_input
    bert.index_first_axis = _index_first_axis
    fa.bert_padding = bert

    sys.modules.update({
        "flash_attn": fa,
        "flash_attn.flash_attn_interface": fai,
        "flash_attn.layers": layers,
        "flash_attn.layers.rotary": rotary,
        "flash_attn.bert_padding": bert,
    })
    return fa


def ensure_flash_attn() -> str:
    """Make ``import flash_attn`` work. Returns 'real', 'shim', or 'already-shim'.

    Tries the real package first; only installs the SDPA shim if it is missing or
    broken. Safe to call repeatedly. Call BEFORE loading any teacher whose remote
    code imports flash_attn (the teachers do this for you)."""
    existing = sys.modules.get("flash_attn")
    if existing is not None:
        return "already-shim" if getattr(existing, "__siflow_shim__", False) else "real"
    try:
        import flash_attn  # noqa: F401  (real package present and importable)
        return "real"
    except Exception:  # noqa: BLE001 - ImportError, or CUDA/loader errors on a broken install
        _build_shim()
        return "shim"
