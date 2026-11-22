"""L3-01 Candidate Kernel: Autotuned Causal GQA FlashAttention-2 Forward with zero KV expansion."""

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
    key=["BH", "L_q", "L_kv", "D", "KV_GROUP_SIZE", "CAUSAL"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def flash_attn2_gqa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    BH,
    L_q,
    L_kv,
    D,
    H_q,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
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
        causal=CAUSAL,
        kv_group_size=KV_GROUP_SIZE,
        num_q_heads=H_q,
    )


def candidate_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()

    b, h_q, l_q, d = q_c.shape
    h_kv = k_c.shape[1]
    l_kv = k_c.shape[2]
    bh = b * h_q
    kv_group_size = h_q // h_kv
    scale = 1.0 / math.sqrt(float(d))

    out = torch.empty((b, h_q, l_q, d), device=q_c.device, dtype=q_c.dtype)
    grid = lambda meta: (m1triton.cdiv(l_q, meta["BLOCK_M"]), bh, 1)
    flash_attn2_gqa_fwd_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        bh,
        l_q,
        l_kv,
        d,
        h_q,
        SCALE=scale,
        CAUSAL=bool(is_causal),
        KV_GROUP_SIZE=int(kv_group_size),
        wait=True,
        sync_mps=True,
    )
    return out
