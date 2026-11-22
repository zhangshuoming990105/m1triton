"""L3-04 Candidate Kernel: Autotuned Sliding-Window + ALiBi + Logit-Softcap Causal FlashAttention."""

from __future__ import annotations

import math

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "NUM_SG": 4}, num_warps=4),
    ],
    key=["BH", "L_q", "L_kv", "D", "WINDOW_SIZE", "SOFTCAP", "ALIBI_HEADS"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def sliding_window_alibi_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    alibi_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SOFTCAP: tl.constexpr,
    ALIBI_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
        causal=True,
        window_size=WINDOW_SIZE,
        softcap=SOFTCAP,
        alibi_ptr=alibi_ptr,
    )


def candidate_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,
    *,
    window_size: int = 128,
    softcap: float = 50.0,
) -> torch.Tensor:
    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()
    slopes_c = alibi_slopes if alibi_slopes.is_contiguous() else alibi_slopes.contiguous()

    b, h, l_q, d = q_c.shape
    l_kv = k_c.shape[2]
    bh = b * h
    scale = 1.0 / math.sqrt(float(d))

    out = torch.empty((b, h, l_q, d), device=q_c.device, dtype=q_c.dtype)
    grid = lambda meta: (m1triton.cdiv(l_q, meta["BLOCK_M"]), bh, 1)
    sliding_window_alibi_attn_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        slopes_c,
        bh,
        l_q,
        l_kv,
        d,
        SCALE=scale,
        WINDOW_SIZE=int(window_size),
        SOFTCAP=float(softcap),
        ALIBI_HEADS=int(h),
        wait=True,
        sync_mps=True,
    )
    return out
