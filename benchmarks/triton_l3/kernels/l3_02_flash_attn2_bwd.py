"""L3-02 Candidate Kernel: Autotuned Causal FlashAttention-2 Forward + Backward via torch.autograd.Function."""

from __future__ import annotations

import math

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.tlx as tlx


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "NUM_SG": 4}, num_warps=4),
    ],
    key=["BH", "L_q", "L_kv", "D", "CAUSAL"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def flash_attn2_fwd_lse_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
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
        lse_ptr=lse_ptr,
    )


@m1triton.jit
def flash_attn2_delta_kernel(
    do_ptr,
    o_ptr,
    delta_ptr,
    D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d_i64 = D.to(tl.int64)
    row_base = row * d_i64
    ar = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(0, D, BLOCK_D):
        k_offs = k + ar
        mask = k_offs < D
        do_v = tl.load(do_ptr + row_base + k_offs.to(tl.int64), mask=mask, other=0.0)
        o_v = tl.load(o_ptr + row_base + k_offs.to(tl.int64), mask=mask, other=0.0)
        acc = acc + do_v * o_v
    tl.store(delta_ptr + row, tl.sum(acc, axis=0))


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "NUM_SG": 8}, num_warps=8),
        m1triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "NUM_SG": 8}, num_warps=8),
    ],
    key=["BH", "L_q", "L_kv", "D", "CAUSAL"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def flash_attn2_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.flash_attention_bwd_dkdv(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
    )


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "NUM_SG": 8}, num_warps=8),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "NUM_SG": 8}, num_warps=8),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "NUM_SG": 4}, num_warps=4),
    ],
    key=["BH", "L_q", "L_kv", "D", "CAUSAL"],
    warmup=4,
    rep=10,
)
@m1triton.jit
def flash_attn2_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.flash_attention_bwd_dq(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        delta_ptr,
        dq_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
    )


class _FlashAttention2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()

        b, h, l_q, d = q_c.shape
        l_kv = k_c.shape[2]
        bh = b * h
        scale = 1.0 / math.sqrt(float(d))

        out = torch.empty((b, h, l_q, d), device=q_c.device, dtype=q_c.dtype)
        lse = torch.empty((bh, l_q), device=q_c.device, dtype=torch.float32)

        grid = lambda meta: (m1triton.cdiv(l_q, meta["BLOCK_M"]), bh, 1)
        need_bwd = any(ctx.needs_input_grad[:3])
        flash_attn2_fwd_lse_kernel[grid](
            q_c,
            k_c,
            v_c,
            out,
            lse,
            bh,
            l_q,
            l_kv,
            d,
            SCALE=scale,
            CAUSAL=bool(is_causal),
            wait=not need_bwd,
            sync_mps=True,
        )

        ctx.save_for_backward(q_c, k_c, v_c, out, lse)
        ctx.scale = scale
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_c, k_c, v_c, out, lse = ctx.saved_tensors
        do_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        b, h, l_q, d = q_c.shape
        l_kv = k_c.shape[2]
        bh = b * h
        scale = ctx.scale
        is_causal = ctx.is_causal

        delta = torch.empty((bh, l_q), device=q_c.device, dtype=torch.float32)
        dq = torch.empty((b, h, l_q, d), device=q_c.device, dtype=q_c.dtype)
        dk = torch.empty((b, h, l_kv, d), device=k_c.device, dtype=k_c.dtype)
        dv = torch.empty((b, h, l_kv, d), device=v_c.device, dtype=v_c.dtype)

        block_d = 64 if d <= 64 else 128
        grid_delta = (bh * l_q, 1, 1)
        grid_dkdv = lambda meta: (m1triton.cdiv(l_kv, meta["BLOCK_N"]), bh, 1)
        grid_dq = lambda meta: (m1triton.cdiv(l_q, meta["BLOCK_M"]), bh, 1)

        dkdv_args = (q_c, k_c, v_c, do_c, lse, delta, dk, dv, bh, l_q, l_kv, d)
        dq_args = (q_c, k_c, v_c, do_c, lse, delta, dq, bh, l_q, l_kv, d)
        bwd_kw = {"SCALE": scale, "CAUSAL": is_causal}

        if flash_attn2_bwd_dkdv_kernel.best_config is None or flash_attn2_bwd_dq_kernel.best_config is None:
            flash_attn2_delta_kernel[grid_delta](do_c, out, delta, d, BLOCK_D=block_d, wait=True, sync_mps=True)
            flash_attn2_bwd_dkdv_kernel._select_config(grid_dkdv, dkdv_args, dict(bwd_kw))
            flash_attn2_bwd_dq_kernel._select_config(grid_dq, dq_args, dict(bwd_kw))

        with rt.command_batch(wait=True, sync_mps=True):
            flash_attn2_delta_kernel[grid_delta](do_c, out, delta, d, BLOCK_D=block_d, wait=False, sync_mps=False)
            flash_attn2_bwd_dkdv_kernel[grid_dkdv](*dkdv_args, **bwd_kw, wait=False, sync_mps=False)
            flash_attn2_bwd_dq_kernel[grid_dq](*dq_args, **bwd_kw, wait=False, sync_mps=False)

        return dq, dk, dv, None


def candidate_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    return _FlashAttention2Function.apply(q, k, v, bool(is_causal))
