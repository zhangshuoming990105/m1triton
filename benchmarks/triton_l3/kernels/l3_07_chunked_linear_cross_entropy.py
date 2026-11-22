"""Level 3 Kernel L3-07: Chunked Fused Linear + Cross-Entropy Forward & Backward.

Liger-style chunked LM-head fusion on Apple M1 Max:
  - Never materializes a full `[BT, V]` logits tensor; reuses a single `[CHUNK_T, V]` scratch tile.
  - Fuses forward loss and unit input/weight gradients in 3 SGMM passes total (zero recomputation).
  - Applies exact upstream scalar `grad_out` in backward via vectorized `float4` M1Triton kernels.
  - Autotunes both SGMM / CE tile schedules (`@m1triton.autotune`) and `CHUNK_T` with persistent disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import time
from typing import Any

import torch

import m1triton
from m1triton.autotune import get_autotune_cache_dir
import m1triton.language as tl
import m1triton.runtime as m1_runtime
import m1triton.tlx as tlx

_CHUNK_T_CANDIDATES: tuple[int, ...] = (128, 256, 512)
_CHUNK_T_CACHE: dict[tuple[Any, ...], int] = {}
_CHUNK_CACHE_VERSION = 1


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _sgmm_fwd_configs() -> list[m1triton.Config]:
    return [
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "SWIZZLE_W": 8}),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 8}),
        m1triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 4}),
        m1triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 8}),
        m1triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 8}),
    ]


def _sgmm_bwd_configs() -> list[m1triton.Config]:
    return [
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "SWIZZLE_W": 8}),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 8}),
        m1triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 8, "SWIZZLE_W": 8}),
    ]


def _dlogits_configs() -> list[m1triton.Config]:
    return [
        m1triton.Config({"BLOCK_V": 512}),
        m1triton.Config({"BLOCK_V": 1024}),
    ]


@m1triton.autotune(configs=_sgmm_fwd_configs(), key=["M", "N", "K", "ALIGNED_MNK"], warmup=2, rep=6, use_disk_cache=True)
@m1triton.jit
def _chunk_linear_fwd_sgmm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    ALIGNED_MNK: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 16,
    NUM_SG: tl.constexpr = 8,
    SWIZZLE_W: tl.constexpr = 8,
):
    tlx.tile_spec(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_simdgroups=NUM_SG,
        double_buffer=False,
        split_k=1,
        swizzle_w=SWIZZLE_W,
    )
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :]
        b_ptrs = b_ptr + k_idx[:, None] + offs_n[None, :] * stride_bn
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@m1triton.autotune(configs=_sgmm_bwd_configs(), key=["M", "N", "K", "ALIGNED_MNK"], warmup=2, rep=6, use_disk_cache=True)
@m1triton.jit
def _chunk_dx_sgmm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,
    ALIGNED_MNK: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 16,
    NUM_SG: tl.constexpr = 4,
    SWIZZLE_W: tl.constexpr = 8,
):
    tlx.tile_spec(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_simdgroups=NUM_SG,
        double_buffer=False,
        split_k=1,
        swizzle_w=SWIZZLE_W,
    )
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :]
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :]
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@m1triton.autotune(
    configs=_sgmm_bwd_configs(),
    key=["M", "N", "K", "ALIGNED_MNK", "ADD_RES"],
    warmup=2,
    rep=6,
    restore_value=["c_ptr"],
    use_disk_cache=True,
)
@m1triton.jit
def _chunk_dw_sgmm(
    a_ptr,
    b_ptr,
    c_ptr,
    res_ptr,
    M,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_cm,
    ALIGNED_MNK: tl.constexpr,
    ADD_RES: tl.constexpr = 0,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 16,
    NUM_SG: tl.constexpr = 8,
    SWIZZLE_W: tl.constexpr = 8,
):
    tlx.tile_spec(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_simdgroups=NUM_SG,
        double_buffer=False,
        split_k=1,
        swizzle_w=SWIZZLE_W,
    )
    if ADD_RES == 1:
        tlx.epilogue((("add_residual", "res_ptr"),))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :]
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@m1triton.jit
def _online_log_softmax_inplace_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    ROW_LEN: tl.constexpr,
):
    tlx.online_softmax(x_ptr, out_ptr, num_rows, ROW_LEN, axis=1, log_softmax=True)


@m1triton.jit
def _ce_infer_stage1_kernel(
    pred_ptr,
    tgt_ptr,
    partial_ptr,
    batch_size,
    NUM_CLASSES: tl.constexpr,
):
    tlx.fused_loss("cross_entropy", pred_ptr, tgt_ptr, partial_ptr, batch_size, NUM_CLASSES, stage=1)


@m1triton.jit
def _ce_reduce_mean_kernel(
    partial_ptr,
    out_ptr,
    num_partials,
    denom,
):
    tlx.fused_loss("cross_entropy", partial_ptr, out_ptr, num_partials, denom, stage=2)


@m1triton.jit
def _gather_neg_logprob_kernel(
    logp_ptr,
    targets_ptr,
    loss_ptr,
    C_size,
    V_size,
    BLOCK_C: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < C_size
    o64 = offs.to(tl.int64)
    tgt = tl.load(targets_ptr + o64, mask=mask, other=0)
    lp = tl.load(logp_ptr + o64 * V_size.to(tl.int64) + tgt, mask=mask, other=0.0)
    tl.store(loss_ptr + o64, -lp, mask=mask)


@m1triton.autotune(
    configs=_dlogits_configs(),
    key=["V_size"],
    warmup=2,
    rep=6,
    restore_value=["buf_ptr"],
    use_disk_cache=True,
)
@m1triton.jit
def _logp_to_dlogits_vec4_kernel(
    buf_ptr,
    targets_ptr,
    V_size,
    inv_bt,
    BLOCK_V: tl.constexpr = 1024,
):
    tlx.vec_width(4)
    pid_v = tl.program_id(0)
    row = tl.program_id(1)
    base_v = pid_v * (4 * BLOCK_V) + tl.arange(0, BLOCK_V)
    r64 = row.to(tl.int64)
    row_off = r64 * V_size.to(tl.int64)
    tgt = tl.load(targets_ptr + r64).to(tl.int32)

    o0 = base_v
    o1 = base_v + BLOCK_V
    o2 = base_v + 2 * BLOCK_V
    o3 = base_v + 3 * BLOCK_V
    m0 = o0 < V_size
    m1 = o1 < V_size
    m2 = o2 < V_size
    m3 = o3 < V_size

    p0 = tl.exp(tl.load(buf_ptr + row_off + o0.to(tl.int64), mask=m0, other=0.0))
    p1 = tl.exp(tl.load(buf_ptr + row_off + o1.to(tl.int64), mask=m1, other=0.0))
    p2 = tl.exp(tl.load(buf_ptr + row_off + o2.to(tl.int64), mask=m2, other=0.0))
    p3 = tl.exp(tl.load(buf_ptr + row_off + o3.to(tl.int64), mask=m3, other=0.0))

    tl.store(buf_ptr + row_off + o0.to(tl.int64), tl.where(o0 == tgt, p0 - 1.0, p0) * inv_bt, mask=m0)
    tl.store(buf_ptr + row_off + o1.to(tl.int64), tl.where(o1 == tgt, p1 - 1.0, p1) * inv_bt, mask=m1)
    tl.store(buf_ptr + row_off + o2.to(tl.int64), tl.where(o2 == tgt, p2 - 1.0, p2) * inv_bt, mask=m2)
    tl.store(buf_ptr + row_off + o3.to(tl.int64), tl.where(o3 == tgt, p3 - 1.0, p3) * inv_bt, mask=m3)


@m1triton.jit
def _scale_by_scalar_ptr_vec4_kernel(
    in_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid = tl.program_id(0).to(tl.int64)
    base = pid * (4 * BLOCK) + tl.arange(0, BLOCK).to(tl.int64)
    n64 = n_elements.to(tl.int64)
    s = tl.load(scale_ptr)

    o0 = base
    o1 = base + BLOCK
    o2 = base + 2 * BLOCK
    o3 = base + 3 * BLOCK
    m0 = o0 < n64
    m1 = o1 < n64
    m2 = o2 < n64
    m3 = o3 < n64

    tl.store(out_ptr + o0, tl.load(in_ptr + o0, mask=m0, other=0.0) * s, mask=m0)
    tl.store(out_ptr + o1, tl.load(in_ptr + o1, mask=m1, other=0.0) * s, mask=m1)
    tl.store(out_ptr + o2, tl.load(in_ptr + o2, mask=m2, other=0.0) * s, mask=m2)
    tl.store(out_ptr + o3, tl.load(in_ptr + o3, mask=m3, other=0.0) * s, mask=m3)


def _is_aligned_mnk(m: int, n: int, k: int) -> int:
    return 1 if (m % 128 == 0 and n % 128 == 0 and k % 16 == 0 and k > 0) else 0


def _ensure_chunk_kernels_tuned(
    x_c: torch.Tensor,
    weight: torch.Tensor,
    targets_c: torch.Tensor,
    logits_c: torch.Tensor,
    dx_c: torch.Tensor | None,
    dw_unit: torch.Tensor | None,
    *,
    bt: int,
    has_multi_chunks: bool,
) -> None:
    """Resolve autotune schedules outside any active Metal command batch."""
    cur = int(x_c.shape[0])
    h = int(x_c.shape[1])
    v = int(weight.shape[0])
    inv_bt = 1.0 / float(bt)

    al_fwd = _is_aligned_mnk(cur, v, h)
    _chunk_linear_fwd_sgmm._select_config(
        lambda meta: (m1triton.cdiv(cur, meta["BLOCK_M"]), m1triton.cdiv(v, meta["BLOCK_N"]), 1),
        (x_c, weight, logits_c, cur, v, h, int(x_c.stride(0)), int(weight.stride(0)), int(logits_c.stride(0))),
        {"ALIGNED_MNK": al_fwd, "fast_math": True, "sync_mps": False, "wait": True},
    )

    if dx_c is not None or dw_unit is not None:
        _logp_to_dlogits_vec4_kernel._select_config(
            lambda meta: (m1triton.cdiv(v, 4 * meta["BLOCK_V"]), cur, 1),
            (logits_c, targets_c, v, inv_bt),
            {"fast_math": True, "sync_mps": False, "wait": True},
        )

    if dx_c is not None:
        al_dx = _is_aligned_mnk(cur, h, v)
        _chunk_dx_sgmm._select_config(
            lambda meta: (m1triton.cdiv(cur, meta["BLOCK_M"]), m1triton.cdiv(h, meta["BLOCK_N"]), 1),
            (logits_c, weight, dx_c, cur, h, v, int(logits_c.stride(0)), int(weight.stride(0)), int(dx_c.stride(0))),
            {"ALIGNED_MNK": al_dx, "fast_math": True, "sync_mps": False, "wait": True},
        )

    if dw_unit is not None:
        al_dw = _is_aligned_mnk(v, h, cur)
        _chunk_dw_sgmm._select_config(
            lambda meta: (m1triton.cdiv(v, meta["BLOCK_M"]), m1triton.cdiv(h, meta["BLOCK_N"]), 1),
            (logits_c, x_c, dw_unit, dw_unit, v, h, cur, int(logits_c.stride(0)), int(x_c.stride(0)), int(dw_unit.stride(0))),
            {"ALIGNED_MNK": al_dw, "ADD_RES": 0, "fast_math": True, "sync_mps": False, "wait": True},
        )
        if has_multi_chunks:
            _chunk_dw_sgmm._select_config(
                lambda meta: (m1triton.cdiv(v, meta["BLOCK_M"]), m1triton.cdiv(h, meta["BLOCK_N"]), 1),
                (logits_c, x_c, dw_unit, dw_unit, v, h, cur, int(logits_c.stride(0)), int(x_c.stride(0)), int(dw_unit.stride(0))),
                {"ALIGNED_MNK": al_dw, "ADD_RES": 1, "fast_math": True, "sync_mps": False, "wait": True},
            )


def _execute_chunked_linear_ce(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_t: int,
    need_dx: bool,
    need_dw: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run chunked forward + unit-gradient pass using a single `[chunk_t, V]` scratch tile."""
    bt, h = int(x.shape[0]), int(x.shape[1])
    v = int(weight.shape[0])
    eff_chunk = max(1, min(int(chunk_t), bt))
    inv_bt = 1.0 / float(bt)
    need_grad = need_dx or need_dw

    dx_unit = torch.empty((bt, h), dtype=torch.float32, device=x.device) if need_dx else None
    dw_unit = torch.empty((v, h), dtype=torch.float32, device=x.device) if need_dw else None
    logits_buf = torch.empty((eff_chunk, v), dtype=torch.float32, device=x.device)
    loss_partials = torch.empty((bt,), dtype=torch.float32, device=x.device)
    loss_out = torch.empty((1,), dtype=torch.float32, device=x.device)

    has_multi_chunks = eff_chunk < bt
    first_cur = min(eff_chunk, bt)
    _ensure_chunk_kernels_tuned(
        x[:first_cur],
        weight,
        targets[:first_cur],
        logits_buf[:first_cur],
        dx_unit[:first_cur] if dx_unit is not None else None,
        dw_unit,
        bt=bt,
        has_multi_chunks=has_multi_chunks,
    )
    tail_rem = bt % eff_chunk
    if tail_rem > 0:
        _ensure_chunk_kernels_tuned(
            x[bt - tail_rem : bt],
            weight,
            targets[bt - tail_rem : bt],
            logits_buf[:tail_rem],
            dx_unit[bt - tail_rem : bt] if dx_unit is not None else None,
            dw_unit,
            bt=bt,
            has_multi_chunks=True,
        )

    with m1_runtime.command_batch(wait=True, sync_mps=True):
        for t0 in range(0, bt, eff_chunk):
            t1 = min(t0 + eff_chunk, bt)
            cur = t1 - t0
            x_c = x[t0:t1]
            tgt_c = targets[t0:t1]
            logits_c = logits_buf[:cur]
            lp_c = loss_partials[t0:t1]

            al_fwd = _is_aligned_mnk(cur, v, h)
            _chunk_linear_fwd_sgmm[
                lambda meta, m=cur, n=v: (m1triton.cdiv(m, meta["BLOCK_M"]), m1triton.cdiv(n, meta["BLOCK_N"]), 1)
            ](
                x_c,
                weight,
                logits_c,
                cur,
                v,
                h,
                int(x_c.stride(0)),
                int(weight.stride(0)),
                int(logits_c.stride(0)),
                ALIGNED_MNK=al_fwd,
                fast_math=True,
                sync_mps=False,
                wait=False,
            )

            if need_grad:
                _online_log_softmax_inplace_kernel[(cur, 1, 1)](
                    logits_c,
                    logits_c,
                    cur,
                    ROW_LEN=v,
                    fast_math=True,
                    sync_mps=False,
                    wait=False,
                )
                _gather_neg_logprob_kernel[(m1triton.cdiv(cur, 256),)](
                    logits_c,
                    tgt_c,
                    lp_c,
                    cur,
                    v,
                    BLOCK_C=256,
                    sync_mps=False,
                    wait=False,
                )
                _logp_to_dlogits_vec4_kernel[
                    lambda meta, n=v, m=cur: (m1triton.cdiv(n, 4 * meta["BLOCK_V"]), m, 1)
                ](
                    logits_c,
                    tgt_c,
                    v,
                    inv_bt,
                    fast_math=True,
                    sync_mps=False,
                    wait=False,
                )

                if dx_unit is not None:
                    dx_c = dx_unit[t0:t1]
                    al_dx = _is_aligned_mnk(cur, h, v)
                    _chunk_dx_sgmm[
                        lambda meta, m=cur, n=h: (m1triton.cdiv(m, meta["BLOCK_M"]), m1triton.cdiv(n, meta["BLOCK_N"]), 1)
                    ](
                        logits_c,
                        weight,
                        dx_c,
                        cur,
                        h,
                        v,
                        int(logits_c.stride(0)),
                        int(weight.stride(0)),
                        int(dx_c.stride(0)),
                        ALIGNED_MNK=al_dx,
                        fast_math=True,
                        sync_mps=False,
                        wait=False,
                    )

                if dw_unit is not None:
                    al_dw = _is_aligned_mnk(v, h, cur)
                    add_res = 0 if t0 == 0 else 1
                    _chunk_dw_sgmm[
                        lambda meta, m=v, n=h: (m1triton.cdiv(m, meta["BLOCK_M"]), m1triton.cdiv(n, meta["BLOCK_N"]), 1)
                    ](
                        logits_c,
                        x_c,
                        dw_unit,
                        dw_unit,
                        v,
                        h,
                        cur,
                        int(logits_c.stride(0)),
                        int(x_c.stride(0)),
                        int(dw_unit.stride(0)),
                        ALIGNED_MNK=al_dw,
                        ADD_RES=add_res,
                        fast_math=True,
                        sync_mps=False,
                        wait=False,
                    )
            else:
                _ce_infer_stage1_kernel[(cur, 1, 1)](
                    logits_c,
                    tgt_c,
                    lp_c,
                    cur,
                    NUM_CLASSES=v,
                    sync_mps=False,
                    wait=False,
                )

        _ce_reduce_mean_kernel[(1, 1, 1)](
            loss_partials,
            loss_out,
            bt,
            float(bt),
            sync_mps=False,
            wait=False,
        )

    del logits_buf, loss_partials
    return loss_out.reshape(()), dx_unit, dw_unit


def _chunk_disk_cache_key(bt: int, h: int, v: int, need_grad: bool, candidates: tuple[int, ...]) -> tuple[str, dict[str, Any]]:
    gpu_name = "unknown"
    try:
        info = m1_runtime.device_info()
        if getattr(info, "name", ""):
            gpu_name = str(info.name)
    except Exception:
        pass
    payload = {
        "version": _CHUNK_CACHE_VERSION,
        "op": "l3_07_chunked_linear_ce_chunk_t",
        "gpu_name": gpu_name,
        "bt": int(bt),
        "h": int(h),
        "v": int(v),
        "need_grad": bool(need_grad),
        "candidates": list(candidates),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), payload


def _select_chunk_t(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    need_dx: bool,
    need_dw: bool,
) -> int:
    """Autotune `CHUNK_T` over `(128, 256, 512)` (strictly `< BT`) with persistent disk caching."""
    bt, h = int(x.shape[0]), int(x.shape[1])
    v = int(weight.shape[0])
    need_grad = need_dx or need_dw

    candidates = tuple(c for c in _CHUNK_T_CANDIDATES if c < bt)
    if not candidates:
        return max(1, bt // 2) if bt >= 2 else 1

    mem_key = (bt, h, v, need_grad, candidates)
    cached = _CHUNK_T_CACHE.get(mem_key)
    if cached is not None:
        return cached

    disk_disabled = os.environ.get("M1TRITON_DISABLE_AUTOTUNE_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}
    cache_hash, payload = _chunk_disk_cache_key(bt, h, v, need_grad, candidates)
    cache_dir = get_autotune_cache_dir()
    cache_path = cache_dir / f"chunk_t_{cache_hash}.json"

    if not disk_disabled and cache_path.is_file():
        try:
            rec = json.loads(cache_path.read_text(encoding="utf-8"))
            val = int(rec.get("best_chunk_t", 0))
            if val in candidates:
                _CHUNK_T_CACHE[mem_key] = val
                return val
        except (OSError, ValueError, TypeError):
            pass

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        best_c = candidates[-1]
        best_ms = float("inf")
        for c in candidates:
            _execute_chunked_linear_ce(x, weight, targets, chunk_t=c, need_dx=need_dx, need_dw=need_dw)
            times_ms: list[float] = []
            for _ in range(2):
                m1_runtime.synchronize()
                t0 = time.perf_counter()
                _execute_chunked_linear_ce(x, weight, targets, chunk_t=c, need_dx=need_dx, need_dw=need_dw)
                m1_runtime.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            med = float(statistics.median(times_ms))
            if med < best_ms:
                best_ms = med
                best_c = c
        chosen = best_c

    _CHUNK_T_CACHE[mem_key] = chosen
    if not disk_disabled:
        tmp_path: str | None = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".chunk_t_", suffix=".tmp", dir=str(cache_dir))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"version": _CHUNK_CACHE_VERSION, "key": payload, "best_chunk_t": chosen}, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, cache_path)
        except OSError:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    return chosen


class _ChunkedLinearCEFn(torch.autograd.Function):
    """Autograd wrapper for chunked fused linear + cross-entropy with zero `[BT, V]` storage."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        orig_x_shape = tuple(x.shape)
        x_2d = _contig(x.reshape(-1, int(x.shape[-1])) if x.ndim > 2 else x)
        w_2d = _contig(weight)
        tgt_1d = _contig(targets.reshape(-1))
        if tgt_1d.dtype != torch.int64:
            tgt_1d = tgt_1d.to(dtype=torch.int64)

        need_dx = bool(ctx.needs_input_grad[0])
        need_dw = bool(ctx.needs_input_grad[1])
        chunk_t = _select_chunk_t(x_2d, w_2d, tgt_1d, need_dx=need_dx, need_dw=need_dw)

        loss, dx_unit, dw_unit = _execute_chunked_linear_ce(
            x_2d,
            w_2d,
            tgt_1d,
            chunk_t=chunk_t,
            need_dx=need_dx,
            need_dw=need_dw,
        )
        ctx.orig_x_shape = orig_x_shape
        ctx.has_dx = dx_unit is not None
        ctx.has_dw = dw_unit is not None
        saved: list[torch.Tensor] = []
        if dx_unit is not None:
            saved.append(dx_unit)
        if dw_unit is not None:
            saved.append(dw_unit)
        ctx.save_for_backward(*saved)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        saved = ctx.saved_tensors
        idx = 0
        dx_unit = saved[idx] if ctx.has_dx else None
        if ctx.has_dx:
            idx += 1
        dw_unit = saved[idx] if ctx.has_dw else None

        g_scalar = _contig(grad_out)
        if g_scalar.dtype != torch.float32:
            g_scalar = g_scalar.to(dtype=torch.float32)

        dx_out = torch.empty_like(dx_unit) if dx_unit is not None else None
        dw_out = torch.empty_like(dw_unit) if dw_unit is not None else None

        block = 1024
        with m1_runtime.command_batch(wait=True, sync_mps=True):
            if dx_unit is not None and dx_out is not None:
                n_dx = int(dx_unit.numel())
                _scale_by_scalar_ptr_vec4_kernel[(m1triton.cdiv(n_dx, 4 * block),)](
                    dx_unit,
                    g_scalar,
                    dx_out,
                    n_dx,
                    BLOCK=block,
                    sync_mps=False,
                    wait=False,
                )
            if dw_unit is not None and dw_out is not None:
                n_dw = int(dw_unit.numel())
                _scale_by_scalar_ptr_vec4_kernel[(m1triton.cdiv(n_dw, 4 * block),)](
                    dw_unit,
                    g_scalar,
                    dw_out,
                    n_dw,
                    BLOCK=block,
                    sync_mps=False,
                    wait=False,
                )

        if dx_out is not None and tuple(dx_out.shape) != ctx.orig_x_shape:
            dx_out = dx_out.reshape(ctx.orig_x_shape)
        return dx_out, dw_out, None


def candidate_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Chunked fused linear + cross-entropy returning exact scalar mean loss with custom backward."""
    return _ChunkedLinearCEFn.apply(x, weight, targets)
