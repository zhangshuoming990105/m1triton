"""Reusable M1Triton FlashAttention-M1 tiled scaled dot-product attention kernel (KB L1 #97)."""

from __future__ import annotations

import math

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    CAUSAL_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SG: tl.constexpr,
    FAST_ONLY: tl.constexpr = 0,
    D_STATIC: tl.constexpr = 0,
    LKV_STATIC: tl.constexpr = 0,
):
    tlx.flash_attention_tile(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
        causal_offset=CAUSAL_OFFSET,
    )


_STATIC_HEAD_DIMS = frozenset({64, 128, 256, 512, 1024})
_STATIC_LKV_SIZES = frozenset({64, 128, 256, 512})


def _select_tile_schedule(
    l_q: int,
    l_kv: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[int, int, int, int, int, int, int]:
    """Select M1 Max 32 KB SRAM tile schedule (BM, BN, BD, NUM_SG, FAST_ONLY, D_STATIC, LKV_STATIC)."""
    if dtype == torch.float32 and l_q > 0 and 0 < head_dim <= 1024 and (head_dim % 64 == 0) and l_kv > 0 and (l_kv % 32 == 0):
        d_static = head_dim if head_dim in _STATIC_HEAD_DIMS else 0
        lkv_static = l_kv if l_kv in _STATIC_LKV_SIZES else 0
        if l_kv <= 512:
            if l_q % 32 == 0:
                return 32, 512, 64, 16, 1, d_static, lkv_static
            if l_q % 16 == 0:
                return 16, 512, 16, 16, 1, d_static, lkv_static
        elif l_q % 16 == 0:
            return 16, 256, 16, 16, 1, d_static, 0
    return 16, 256, 16, 16, 0, 0, 0


def m1_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    causal_offset: int = 0,
    wait: bool = True,
) -> torch.Tensor:
    """Compute scaled dot-product attention via a single fused M1Triton FlashAttention kernel."""
    if query.dim() < 3 or key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError(
            f"Expected matching >=3D (..., L, D) tensors, got Q={tuple(query.shape)}, "
            f"K={tuple(key.shape)}, V={tuple(value.shape)}"
        )
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("Batch/head leading dimensions of Q, K, V must match")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2] or query.shape[-1] != value.shape[-1]:
        raise ValueError("Q, K, V sequence/head dimensions are incompatible")

    q_c = query if query.is_contiguous() else query.contiguous()
    k_c = key if key.is_contiguous() else key.contiguous()
    v_c = value if value.is_contiguous() else value.contiguous()

    lead_shape = tuple(int(s) for s in q_c.shape[:-2])
    bh = 1
    for s in lead_shape:
        bh *= s
    l_q = int(q_c.shape[-2])
    l_kv = int(k_c.shape[-2])
    d = int(q_c.shape[-1])

    out = torch.empty((*lead_shape, l_q, d), dtype=q_c.dtype, device=q_c.device)
    if out.numel() == 0:
        return out

    eff_scale = float(scale) if scale is not None else (1.0 / math.sqrt(float(d)))
    bm, bn, bd, num_sg, fast_only, d_static, lkv_static = _select_tile_schedule(l_q, l_kv, d, q_c.dtype)
    grid = (m1triton.cdiv(l_q, bm), bh, 1)

    flash_attention_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        bh,
        l_q,
        l_kv,
        d,
        SCALE=eff_scale,
        CAUSAL=bool(is_causal),
        CAUSAL_OFFSET=int(causal_offset),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_D=bd,
        NUM_SG=num_sg,
        FAST_ONLY=fast_only,
        D_STATIC=d_static,
        LKV_STATIC=lkv_static,
        wait=wait,
    )
    return out
