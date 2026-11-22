"""L3-03 Candidate Kernel: Autotuned 2-Stage Split-KV PagedAttention Decode in a single Metal command batch."""

from __future__ import annotations

import math

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.tlx as tlx

_MAX_SPLITS = 16


@m1triton.autotune(
    configs=[
        m1triton.Config({"NUM_SPLITS": 2, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"NUM_SPLITS": 4, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"NUM_SPLITS": 8, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"NUM_SPLITS": 4, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"NUM_SPLITS": 8, "NUM_SG": 2}, num_warps=2),
    ],
    key=["B", "H_q", "H_kv", "MaxBlocks", "D", "BLOCK_SIZE"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def paged_attn_decode_stage1_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    partial_o_ptr,
    partial_lse_ptr,
    B,
    H_q,
    H_kv,
    MaxBlocks,
    D,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.paged_attention_decode(
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        partial_o_ptr,
        partial_lse_ptr,
        B,
        H_q,
        H_kv,
        MaxBlocks,
        D,
        scale=SCALE,
    )


@m1triton.jit
def split_kv_merge_kernel(
    partial_o_ptr,
    partial_lse_ptr,
    out_ptr,
    D,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0).to(tl.int64)
    d_i64 = D.to(tl.int64)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    o_base = bh * (NUM_SPLITS * d_i64)
    lse_base = bh * (NUM_SPLITS * 2)

    m_max = -1e30
    for s in tl.static_range(0, NUM_SPLITS):
        m_s = tl.load(partial_lse_ptr + lse_base + s * 2)
        l_s = tl.load(partial_lse_ptr + lse_base + s * 2 + 1)
        m_eff = tl.where(l_s > 0.0, m_s, -1e30)
        m_max = tl.maximum(m_max, m_eff)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    denom = 0.0
    for s in tl.static_range(0, NUM_SPLITS):
        m_s = tl.load(partial_lse_ptr + lse_base + s * 2)
        l_s = tl.load(partial_lse_ptr + lse_base + s * 2 + 1)
        w_s = tl.where(l_s > 0.0, l_s * tl.exp(m_s - m_max), 0.0)
        po = tl.load(partial_o_ptr + o_base + s * d_i64 + d_offs.to(tl.int64), mask=d_mask, other=0.0)
        acc = acc + po * w_s
        denom = denom + w_s

    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    tl.store(out_ptr + bh * d_i64 + d_offs.to(tl.int64), acc * inv_denom, mask=d_mask)


def candidate_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    q_c = q if q.is_contiguous() else q.contiguous()
    kc_c = k_cache if k_cache.is_contiguous() else k_cache.contiguous()
    vc_c = v_cache if v_cache.is_contiguous() else v_cache.contiguous()
    bt_c = block_tables if block_tables.is_contiguous() else block_tables.contiguous()
    cl_c = context_lens if context_lens.is_contiguous() else context_lens.contiguous()

    b, h_q, d = q_c.shape
    h_kv = kc_c.shape[1]
    max_blocks = bt_c.shape[1]
    bh = b * h_q
    scale = 1.0 / math.sqrt(float(d))

    partial_o = torch.empty((bh, _MAX_SPLITS, d), device=q_c.device, dtype=torch.float32)
    partial_lse = torch.empty((bh, _MAX_SPLITS, 2), device=q_c.device, dtype=torch.float32)
    out = torch.empty((b, h_q, d), device=q_c.device, dtype=q_c.dtype)

    grid_s1 = lambda meta: (meta["NUM_SPLITS"], bh, 1)
    s1_args = (q_c, kc_c, vc_c, bt_c, cl_c, partial_o, partial_lse, b, h_q, h_kv, max_blocks, d)
    s1_kw = {"SCALE": scale, "BLOCK_SIZE": int(block_size)}

    chosen_cfg = paged_attn_decode_stage1_kernel._select_config(grid_s1, s1_args, dict(s1_kw))
    num_splits = int(chosen_cfg.kwargs["NUM_SPLITS"])
    block_d = m1triton.next_power_of_2(max(d, 32))

    with rt.command_batch(wait=True, sync_mps=True):
        paged_attn_decode_stage1_kernel[grid_s1](*s1_args, **s1_kw, wait=False, sync_mps=False)
        split_kv_merge_kernel[(bh, 1, 1)](
            partial_o,
            partial_lse,
            out,
            d,
            NUM_SPLITS=num_splits,
            BLOCK_D=block_d,
            wait=False,
            sync_mps=False,
        )

    return out
