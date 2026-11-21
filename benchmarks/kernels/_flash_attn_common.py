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


def _select_tile_schedule(l_kv: int, head_dim: int, dtype: torch.dtype) -> tuple[int, int, int, int]:
    """Select M1 Max 32 KB SRAM tile schedule (BM, BN, BD, NUM_SG)."""
    if dtype == torch.float32 and l_kv <= 512 and (l_kv % 32 == 0) and 0 < head_dim <= 1024 and (head_dim % 64 == 0):
        return 16, 512, 16, 16
    return 16, 256, 16, 16


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

    q_c = query.contiguous()
    k_c = key.contiguous()
    v_c = value.contiguous()

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
    bm, bn, bd, num_sg = _select_tile_schedule(l_kv, d, q_c.dtype)
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
        wait=wait,
    )
    return out
