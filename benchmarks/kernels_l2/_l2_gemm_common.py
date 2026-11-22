"""Reusable M1Triton fused SGMM + EpilogueDAG + FusedTail + GPU Norm/Reduce helpers for KernelBench Level 2."""

from __future__ import annotations

from typing import Any, Sequence

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as m1_runtime
import m1triton.tlx as tlx


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def copy_tensor_(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.detach().to(device=dst.device, dtype=dst.dtype))


def _select_fused_sgmm_schedule(
    m: int,
    n: int,
    k: int,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    tail_kind: str = "none",
    tail_k: int = 1,
) -> tuple[int, int, int, int, int, int]:
    if trans_b and 96 <= m <= 256 and n >= 4096 and (tail_k <= 1 or 64 % tail_k == 0):
        return 128, 64, 16, 8, 8, 0
    if tail_kind in ("pool1d_max", "pool1d_avg") and tail_k > 1:
        if m >= 64 and n >= 128 and (128 % tail_k == 0):
            return 64, 128, 16, 8, 8, 0
        if 64 % tail_k == 0:
            return 64, 64, 16, 4, 8, 0
    if k <= 64 and m >= 64 and n >= 64 and tail_kind == "none":
        return 64, 64, 16, 4, 1, 1
    if (trans_a or trans_b) and m >= 64 and n >= 128:
        return 64, 128, 16, 8, 8, 0
    return 64, 64, 16, 4, 8, 0


@m1triton.jit
def sgmm_fused_epilogue_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    ep0_ptr,
    ep1_ptr,
    ep2_ptr,
    ep3_ptr,
    ep_s0,
    ep_s1,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr = 4,
    SWIZZLE_W: tl.constexpr = 8,
    TRANS_A: tl.constexpr = 0,
    TRANS_B: tl.constexpr = 1,
    N_MAJOR: tl.constexpr = 0,
    ALIGNED_MNK: tl.constexpr = 0,
    EPILOGUE_SPEC: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_K: tl.constexpr = 1,
    TAIL_S: tl.constexpr = 1,
    TAIL_SCALE: tl.constexpr = 1.0,
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
    if EPILOGUE_SPEC:
        tlx.epilogue(EPILOGUE_SPEC)
    if TAIL_KIND != "none":
        tlx.fused_tail(TAIL_KIND, kernel_size=TAIL_K, stride=TAIL_S, post_scale=TAIL_SCALE)

    if N_MAJOR == 0:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    else:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :]
        else:
            a_ptrs = a_ptr + offs_m[:, None] + k_idx[None, :] * stride_ak
        if TRANS_B == 0:
            b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :]
        else:
            b_ptrs = b_ptr + k_idx[:, None] + offs_n[None, :] * stride_bn
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


_PTR_EPILOGUE_OPS = frozenset(
    {
        "add_bias",
        "sub_bias",
        "mul_bias",
        "div_bias",
        "min_bias",
        "max_bias",
        "add_residual",
        "sub_residual",
        "mul_residual",
    }
)
_SCALAR_EPILOGUE_OPS = frozenset({"add_scalar", "sub_scalar", "mul_scalar", "div_scalar"})
_INF_SENSITIVE_TAILS = frozenset({"row_max", "row_min", "row_logsumexp", "pool1d_max"})


def _normalize_epilogue_spec(
    bias: torch.Tensor | None,
    epilogue_steps: Sequence[Any],
    extra_bufs: Sequence[torch.Tensor],
    extra_scalars: Sequence[float],
    fallback_buf: torch.Tensor,
) -> tuple[tuple[Any, ...], list[torch.Tensor], list[float]]:
    bufs: list[torch.Tensor] = []
    alias_map: dict[str, str] = {}
    steps_out: list[Any] = []

    if bias is not None:
        bufs.append(_contig(bias))
        alias_map["bias"] = "ep0_ptr"
        steps_out.append(("add_bias", "ep0_ptr"))

    for i, t in enumerate(extra_bufs):
        slot = len(bufs)
        if slot >= 4:
            raise ValueError("sgmm_fused_epilogue_kernel supports at most 4 epilogue buffer arguments")
        bufs.append(_contig(t))
        alias_map[f"buf{i}"] = f"ep{slot}_ptr"

    while len(bufs) < 4:
        bufs.append(fallback_buf)

    scalars = [float(s) for s in extra_scalars]
    if len(scalars) > 2:
        raise ValueError("sgmm_fused_epilogue_kernel supports at most 2 epilogue scalar arguments")
    while len(scalars) < 2:
        scalars.append(0.0)

    scalar_alias = {"s0": "ep_s0", "s1": "ep_s1"}

    for raw in epilogue_steps:
        if isinstance(raw, str):
            steps_out.append((raw,))
            continue
        op = str(raw[0])
        key = op.strip().lower().replace("-", "_")
        if key in _PTR_EPILOGUE_OPS:
            ref = raw[1]
            if isinstance(ref, str) and ref in alias_map:
                ref = alias_map[ref]
            elif isinstance(ref, int):
                offset = 1 if bias is not None else 0
                ref = f"ep{ref + offset}_ptr"
            steps_out.append((op, ref))
        elif key in _SCALAR_EPILOGUE_OPS:
            ref = raw[1]
            if isinstance(ref, str) and ref in scalar_alias:
                ref = scalar_alias[ref]
            elif isinstance(ref, int):
                ref = f"ep_s{ref}"
            steps_out.append((op, ref))
        else:
            steps_out.append(tuple(raw))

    return tuple(steps_out), bufs, scalars


def fused_linear_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    epilogue_steps: Sequence[Any] = (),
    extra_bufs: Sequence[torch.Tensor] = (),
    extra_scalars: Sequence[float] = (),
    tail: dict[str, Any] | None = None,
    trans_b: bool = True,
    out: torch.Tensor | None = None,
    fast_math: bool | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> tuple[torch.Tensor, int]:
    """Compile & launch 1 fused SGMM kernel with arbitrary EpilogueDAG and optional FusedTail.

    Returns `(output_tensor, grid_n)`. For row-reduction tails (`row_sum`, `row_mean`, `row_max`, `row_min`),
    `output_tensor` has shape `(M, grid_n)`; for `row_logsumexp`, shape `(M, grid_n, 2)`; for `pool1d_*`,
    shape `(M, N // kernel_size)`; otherwise shape `(M, N)`.
    """
    x_c = _contig(x)
    w_c = _contig(weight)
    m, k = int(x_c.shape[0]), int(x_c.shape[1])
    if trans_b:
        n, k_w = int(w_c.shape[0]), int(w_c.shape[1])
        stride_bk = int(w_c.stride(1))
        stride_bn = int(w_c.stride(0))
    else:
        k_w, n = int(w_c.shape[0]), int(w_c.shape[1])
        stride_bk = int(w_c.stride(0))
        stride_bn = int(w_c.stride(1))
    if k != k_w:
        raise ValueError(f"Incompatible Linear K dimensions: {k} vs {k_w}")

    tail_kind = "none"
    tail_k = 1
    tail_s = 1
    tail_scale = 1.0
    if tail is not None:
        tail_kind = str(tail.get("kind", "none")).lower()
        tail_k = int(tail.get("kernel_size", 1))
        tail_s = int(tail.get("stride", tail_k))
        tail_scale = float(tail.get("post_scale", 1.0))

    bm, bn, bk, num_sg, swizzle_w, n_maj = _select_fused_sgmm_schedule(
        m, n, k, trans_a=False, trans_b=trans_b, tail_kind=tail_kind, tail_k=tail_k
    )
    grid_m = m1triton.cdiv(m, bm)
    grid_n = m1triton.cdiv(n, bn)

    if out is None:
        if tail_kind in ("row_sum", "row_mean", "row_max", "row_min"):
            out = torch.empty((m, grid_n), dtype=x_c.dtype, device=x_c.device)
        elif tail_kind == "row_logsumexp":
            out = torch.empty((m, grid_n, 2), dtype=x_c.dtype, device=x_c.device)
        elif tail_kind in ("pool1d_max", "pool1d_avg"):
            out = torch.empty((m, n // tail_k), dtype=x_c.dtype, device=x_c.device)
        else:
            out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)

    stride_cm = int(out.stride(0))
    stride_cn = 1

    ep_spec, ep_bufs, ep_scalars = _normalize_epilogue_spec(bias, epilogue_steps, extra_bufs, extra_scalars, x_c)
    aligned = 1 if (m % bm == 0 and n % bn == 0 and k % bk == 0 and k > 0) else 0
    if fast_math is None:
        use_fast_math = tail_kind not in _INF_SENSITIVE_TAILS
    else:
        use_fast_math = bool(fast_math)

    grid = (grid_n, grid_m, 1) if n_maj else (grid_m, grid_n, 1)
    sgmm_fused_epilogue_kernel[grid](
        x_c,
        w_c,
        out,
        ep_bufs[0],
        ep_bufs[1],
        ep_bufs[2],
        ep_bufs[3],
        ep_scalars[0],
        ep_scalars[1],
        m,
        n,
        k,
        int(x_c.stride(0)),
        int(x_c.stride(1)),
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=num_sg,
        SWIZZLE_W=swizzle_w,
        TRANS_A=0,
        TRANS_B=1 if trans_b else 0,
        N_MAJOR=n_maj,
        ALIGNED_MNK=aligned,
        EPILOGUE_SPEC=ep_spec,
        TAIL_KIND=tail_kind,
        TAIL_K=tail_k,
        TAIL_S=tail_s,
        TAIL_SCALE=tail_scale,
        fast_math=use_fast_math,
        sync_mps=sync_mps,
        wait=wait,
    )
    return out, grid_n


@m1triton.jit
def bn_fold_affine_kernel(
    w_ptr,
    b_ptr,
    rm_ptr,
    rv_ptr,
    pre_s_ptr,
    post_s_ptr,
    post_b_ptr,
    out_s_ptr,
    out_b_ptr,
    C,
    eps,
    div_val,
    HAS_AFFINE: tl.constexpr,
    HAS_PRE_SCALE: tl.constexpr,
    POST_SCALE_MODE: tl.constexpr,
    POST_BIAS_MODE: tl.constexpr,
    HAS_DIV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    rm = tl.load(rm_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(rv_ptr + offs, mask=mask, other=1.0)
    inv_std = 1.0 / tl.sqrt(rv + eps)
    if HAS_AFFINE:
        w = tl.load(w_ptr + offs, mask=mask, other=1.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        bn_s = w * inv_std
        shift = b - rm * bn_s
    else:
        bn_s = inv_std
        shift = -rm * bn_s

    if HAS_PRE_SCALE:
        pre_s = tl.load(pre_s_ptr + offs, mask=mask, other=1.0)
        scale = pre_s * bn_s
    else:
        scale = bn_s

    if POST_SCALE_MODE == 1:
        ps = tl.load(post_s_ptr)
        scale = scale * ps
        shift = shift * ps
    elif POST_SCALE_MODE == 2:
        ps = tl.load(post_s_ptr + offs, mask=mask, other=1.0)
        scale = scale * ps
        shift = shift * ps

    if POST_BIAS_MODE == 1:
        pb = tl.load(post_b_ptr)
        shift = shift + pb
    elif POST_BIAS_MODE == 2:
        pb = tl.load(post_b_ptr + offs, mask=mask, other=0.0)
        shift = shift + pb

    if HAS_DIV:
        inv_d = 1.0 / div_val
        scale = scale * inv_d
        shift = shift * inv_d

    tl.store(out_s_ptr + offs, scale, mask=mask)
    tl.store(out_b_ptr + offs, shift, mask=mask)


def bn_fold_affine_gpu(
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor | None,
    bn_bias: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    post_bias: torch.Tensor | None = None,
    divide_value: float | None = None,
    out_scale: torch.Tensor | None = None,
    out_shift: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute live inference BatchNorm1d affine `(scale, shift)` on GPU (zero host caching)."""
    rm = _contig(running_mean)
    rv = _contig(running_var)
    c = int(rm.numel())
    if out_scale is None:
        out_scale = torch.empty((c,), dtype=rm.dtype, device=rm.device)
    if out_shift is None:
        out_shift = torch.empty((c,), dtype=rm.dtype, device=rm.device)

    has_affine = 1 if (bn_weight is not None and bn_bias is not None) else 0
    w_t = _contig(bn_weight) if has_affine else rm
    b_t = _contig(bn_bias) if has_affine else rm
    has_pre = 1 if pre_scale is not None else 0
    pre_t = _contig(pre_scale) if has_pre else rm

    if post_scale is None:
        ps_mode = 0
        ps_t = rm
    else:
        ps_t = _contig(post_scale)
        ps_mode = 1 if ps_t.numel() == 1 else 2

    if post_bias is None:
        pb_mode = 0
        pb_t = rm
    else:
        pb_t = _contig(post_bias)
        pb_mode = 1 if pb_t.numel() == 1 else 2

    has_div = 1 if (divide_value is not None and float(divide_value) != 1.0) else 0
    div_v = float(divide_value) if divide_value is not None else 1.0

    block = 256
    bn_fold_affine_kernel[(m1triton.cdiv(c, block),)](
        w_t,
        b_t,
        rm,
        rv,
        pre_t,
        ps_t,
        pb_t,
        out_scale,
        out_shift,
        c,
        float(eps),
        div_v,
        HAS_AFFINE=has_affine,
        HAS_PRE_SCALE=has_pre,
        POST_SCALE_MODE=ps_mode,
        POST_BIAS_MODE=pb_mode,
        HAS_DIV=has_div,
        BLOCK=block,
        sync_mps=sync_mps,
        wait=wait,
    )
    return out_scale, out_shift


@m1triton.jit
def finalize_row_sum_kernel(
    in_ptr,
    out_ptr,
    P,
    scale,
    BLOCK_P: tl.constexpr,
    APPLY_SCALE: tl.constexpr = 0,
    POST_OP: tl.constexpr = 0,
):
    row = tl.program_id(0)
    row_base = in_ptr + row.to(tl.int64) * P.to(tl.int64)
    ar = tl.arange(0, BLOCK_P)
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for p in range(0, P, BLOCK_P):
        offs = p + ar
        mask = offs < P
        acc = acc + tl.load(row_base + offs.to(tl.int64), mask=mask, other=0.0)
    v = tl.sum(acc, axis=0)
    if APPLY_SCALE:
        v = v * scale
    if POST_OP == 1:
        v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    tl.store(out_ptr + row.to(tl.int64), v)


def finalize_row_sum(
    partials: torch.Tensor,
    *,
    scale: float = 1.0,
    keepdim: bool = True,
    post_op: int = 0,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    p_c = _contig(partials)
    m, p = int(p_c.shape[0]), int(p_c.shape[1])
    out_shape = (m, 1) if keepdim else (m,)
    if out is None:
        out = torch.empty(out_shape, dtype=p_c.dtype, device=p_c.device)
    block_p = min(1024, max(32, _next_pow2(p)))
    apply_scale = 1 if float(scale) != 1.0 else 0
    finalize_row_sum_kernel[(m,)](
        p_c,
        out,
        p,
        float(scale),
        BLOCK_P=block_p,
        APPLY_SCALE=apply_scale,
        POST_OP=int(post_op),
        sync_mps=sync_mps,
        wait=wait,
    )
    return out


@m1triton.jit
def finalize_row_lse_kernel(
    part_ptr,
    out_ptr,
    P,
    BLOCK_P: tl.constexpr,
    POST_OP: tl.constexpr = 0,
):
    row = tl.program_id(0)
    row_base = part_ptr + row.to(tl.int64) * (P.to(tl.int64) * 2)
    ar = tl.arange(0, BLOCK_P)
    neg_inf = -float("inf")
    max_acc = tl.full((BLOCK_P,), neg_inf, dtype=tl.float32)
    for p in range(0, P, BLOCK_P):
        offs = p + ar
        mask = offs < P
        m_vals = tl.load(row_base + (offs.to(tl.int64) * 2), mask=mask, other=neg_inf)
        max_acc = tl.maximum(max_acc, m_vals)
    row_max = tl.max(max_acc, axis=0)

    sum_acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for p in range(0, P, BLOCK_P):
        offs = p + ar
        mask = offs < P
        m_vals = tl.load(row_base + (offs.to(tl.int64) * 2), mask=mask, other=neg_inf)
        s_vals = tl.load(row_base + (offs.to(tl.int64) * 2 + 1), mask=mask, other=0.0)
        valid = mask & (s_vals > 0.0)
        term = tl.where(valid, s_vals * tl.exp(tl.where(valid, m_vals - row_max, 0.0)), 0.0)
        sum_acc = sum_acc + term
    v = row_max + tl.log(tl.sum(sum_acc, axis=0))

    if POST_OP == 1:
        sp = tl.where(v > 20.0, v, tl.log(1.0 + tl.exp(tl.where(v > 20.0, 0.0, v))))
        v = v * (v * tl.tanh(sp))
    elif POST_OP == 2:
        v = tl.where(v > 0.0, v, v * 0.01)
        v = tl.where(v > 0.0, v, v * 0.01)
        v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
        v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))

    tl.store(out_ptr + row.to(tl.int64), v)


def finalize_row_lse(
    partials: torch.Tensor,
    *,
    keepdim: bool = True,
    post_op: int = 0,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    p_c = _contig(partials)
    m, p = int(p_c.shape[0]), int(p_c.shape[1])
    out_shape = (m, 1) if keepdim else (m,)
    if out is None:
        out = torch.empty(out_shape, dtype=p_c.dtype, device=p_c.device)
    block_p = min(64, max(32, _next_pow2(p)))
    finalize_row_lse_kernel[(m,)](
        p_c,
        out,
        p,
        BLOCK_P=block_p,
        POST_OP=int(post_op),
        sync_mps=sync_mps,
        wait=wait,
    )
    return out


@m1triton.jit
def finalize_row_max_kernel(
    in_ptr,
    out_ptr,
    P,
    scale,
    BLOCK_P: tl.constexpr,
    PRE_OP: tl.constexpr = 0,
    POST_OP: tl.constexpr = 0,
):
    row = tl.program_id(0)
    row_base = in_ptr + row.to(tl.int64) * P.to(tl.int64)
    ar = tl.arange(0, BLOCK_P)
    neg_inf = -float("inf")
    acc = tl.full((BLOCK_P,), neg_inf, dtype=tl.float32)
    for p in range(0, P, BLOCK_P):
        offs = p + ar
        mask = offs < P
        vals = tl.load(row_base + offs.to(tl.int64), mask=mask, other=neg_inf)
        if PRE_OP == 1:
            g = 0.5 * vals * (1.0 + tl.erf(vals * 0.7071067811865476)) * scale
            vals = tl.where(mask, g, neg_inf)
        acc = tl.maximum(acc, vals)
    v = tl.max(acc, axis=0)
    if POST_OP == 1:
        v = v - v
        v = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    tl.store(out_ptr + row.to(tl.int64), v)


def finalize_row_max(
    partials: torch.Tensor,
    *,
    scale: float = 1.0,
    keepdim: bool = True,
    pre_op: int = 0,
    post_op: int = 0,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    p_c = _contig(partials)
    m, p = int(p_c.shape[0]), int(p_c.shape[1])
    out_shape = (m, 1) if keepdim else (m,)
    if out is None:
        out = torch.empty(out_shape, dtype=p_c.dtype, device=p_c.device)
    block_p = min(64, max(32, _next_pow2(p)))
    finalize_row_max_kernel[(m,)](
        p_c,
        out,
        p,
        float(scale),
        BLOCK_P=block_p,
        PRE_OP=int(pre_op),
        POST_OP=int(post_op),
        sync_mps=sync_mps,
        wait=wait,
    )
    return out


@m1triton.jit
def broadcast_row_sub_gelu_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    tlx.vec_width(4)
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    mu = tl.load(mean_ptr + row)
    idx = row.to(tl.int64) * N.to(tl.int64) + offs.to(tl.int64)
    v = tl.load(x_ptr + idx, mask=mask, other=0.0) - mu
    g = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    tl.store(out_ptr + idx, g, mask=mask)


def row_mean_sub_gelu(
    x: torch.Tensor,
    *,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    x_c = _contig(x)
    m, n = int(x_c.shape[0]), int(x_c.shape[1])
    means = finalize_row_sum(x_c, scale=1.0 / float(n), keepdim=True, sync_mps=sync_mps, wait=False)
    out = torch.empty_like(x_c)
    block_n = 1024
    broadcast_row_sub_gelu_kernel[(m1triton.cdiv(n, block_n), m)](
        x_c, means, out, n, BLOCK_N=block_n, sync_mps=False, wait=wait
    )
    return out


@m1triton.jit
def online_softmax_rows_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
    LOG_SOFTMAX: tl.constexpr,
):
    tlx.online_softmax(x_ptr, out_ptr, NUM_ROWS, ROW_LEN, axis=1, log_softmax=LOG_SOFTMAX)


def softmax_rows(
    x: torch.Tensor,
    *,
    log_softmax: bool = False,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    x_c = _contig(x)
    rows, row_len = int(x_c.shape[0]), int(x_c.shape[-1])
    if out is None:
        out = torch.empty_like(x_c)
    online_softmax_rows_kernel[(rows, 1, 1)](
        x_c, out, rows, row_len, bool(log_softmax), sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def groupnorm_2d_small_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    extra_ptr,
    out_ptr,
    C,
    G,
    GS,
    eps,
    p0,
    p1,
    HAS_AFFINE: tl.constexpr,
    POST_OP: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid - n * G
    ar = tl.arange(0, BLOCK_GS)
    mask = ar < GS
    c_offs = g * GS + ar
    base = n.to(tl.int64) * C.to(tl.int64) + c_offs.to(tl.int64)

    xv = tl.load(x_ptr + base, mask=mask, other=0.0)
    inv_gs = 1.0 / GS.to(tl.float32)
    mu = tl.sum(xv, axis=0) * inv_gs
    dx = tl.where(mask, xv - mu, 0.0)
    var = tl.sum(dx * dx, axis=0) * inv_gs
    yv = dx * (1.0 / tl.sqrt(var + eps))

    if HAS_AFFINE:
        wv = tl.load(w_ptr + c_offs, mask=mask, other=1.0)
        bv = tl.load(b_ptr + c_offs, mask=mask, other=0.0)
        yv = yv * wv + bv

    if POST_OP == 1:
        yv = tl.where(yv < p0, p0, tl.where(yv > p1, p1, yv))
    elif POST_OP == 2:
        yv = tl.where(yv > 0.0, yv, yv * p0)
        yv = yv + yv
    elif POST_OP == 3:
        mw = tl.load(extra_ptr + c_offs, mask=mask, other=1.0)
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))
        yv = yv * mw
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))

    tl.store(out_ptr + base, yv, mask=mask)


@m1triton.jit
def gn_post_op_1d_kernel(
    x_ptr,
    extra_ptr,
    out_ptr,
    n_elements,
    C,
    p0,
    p1,
    POST_OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    yv = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if POST_OP == 1:
        yv = tl.where(yv < p0, p0, tl.where(yv > p1, p1, yv))
    elif POST_OP == 2:
        yv = tl.where(yv > 0.0, yv, yv * p0)
        yv = yv + yv
    elif POST_OP == 3:
        c_offs = offs % C.to(tl.int64)
        mw = tl.load(extra_ptr + c_offs, mask=mask, other=1.0)
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))
        yv = yv * mw
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))
    tl.store(out_ptr + offs, yv, mask=mask)


@m1triton.jit
def welford_instance_norm_rows_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm("instance_norm", x_ptr, out_ptr, NUM_ROWS, ROW_LEN, eps=EPS, axis=1)


@m1triton.jit
def group_affine_1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    extra_ptr,
    out_ptr,
    n_elements,
    C,
    p0,
    p1,
    HAS_AFFINE: tl.constexpr,
    POST_OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    yv = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_offs = offs % C.to(tl.int64)
    if HAS_AFFINE:
        wv = tl.load(w_ptr + c_offs, mask=mask, other=1.0)
        bv = tl.load(b_ptr + c_offs, mask=mask, other=0.0)
        yv = yv * wv + bv
    if POST_OP == 1:
        yv = tl.where(yv < p0, p0, tl.where(yv > p1, p1, yv))
    elif POST_OP == 2:
        yv = tl.where(yv > 0.0, yv, yv * p0)
        yv = yv + yv
    elif POST_OP == 3:
        mw = tl.load(extra_ptr + c_offs, mask=mask, other=1.0)
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))
        yv = yv * mw
        yv = yv * (1.0 / (1.0 + tl.exp(-yv)))
    tl.store(out_ptr + offs, yv, mask=mask)


def groupnorm_2d_fused(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    num_groups: int,
    eps: float = 1e-5,
    post_op: int = 0,
    p0: float = 0.0,
    p1: float = 0.0,
    extra: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    """Single-dispatch 2D `(N, C)` GroupNorm with optional fused post-activation (fallback via Welford for GS>1024)."""
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    g = int(num_groups)
    gs = c // g
    if out is None:
        out = torch.empty_like(x_c)
    has_affine = 1 if (weight is not None and bias is not None) else 0
    w_t = _contig(weight) if has_affine else x_c
    b_t = _contig(bias) if has_affine else x_c
    e_t = _contig(extra) if extra is not None else x_c

    if gs <= 64:
        block_gs = max(16, _next_pow2(gs))
        groupnorm_2d_small_fused_kernel[(n * g,)](
            x_c,
            w_t,
            b_t,
            e_t,
            out,
            c,
            g,
            gs,
            float(eps),
            float(p0),
            float(p1),
            HAS_AFFINE=has_affine,
            POST_OP=int(post_op),
            BLOCK_GS=block_gs,
            sync_mps=sync_mps,
            wait=wait,
        )
    else:
        normed = torch.empty_like(x_c)
        welford_instance_norm_rows_kernel[(n * g, 1, 1)](
            x_c, normed, n * g, gs, float(eps), sync_mps=sync_mps, wait=False
        )
        if has_affine or post_op != 0:
            total = n * c
            block = 1024
            group_affine_1d_kernel[(m1triton.cdiv(total, block),)](
                normed,
                w_t,
                b_t,
                e_t,
                out,
                total,
                c,
                float(p0),
                float(p1),
                HAS_AFFINE=has_affine,
                POST_OP=int(post_op),
                BLOCK=block,
                sync_mps=False,
                wait=wait,
            )
        else:
            out = normed
    return out


@m1triton.jit
def groupnorm_2d_row_min_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    G,
    GS,
    eps,
    HAS_AFFINE: tl.constexpr,
    BLOCK_GS: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * C.to(tl.int64)
    ar = tl.arange(0, BLOCK_GS)
    mask = ar < GS
    inv_gs = 1.0 / GS.to(tl.float32)
    pos_inf = float("inf")
    row_min = pos_inf
    for g in range(0, G):
        c_offs = g * GS + ar
        xv = tl.load(row_base + c_offs.to(tl.int64), mask=mask, other=0.0)
        mu = tl.sum(xv, axis=0) * inv_gs
        dx = tl.where(mask, xv - mu, 0.0)
        var = tl.sum(dx * dx, axis=0) * inv_gs
        yv = dx * (1.0 / tl.sqrt(var + eps))
        if HAS_AFFINE:
            wv = tl.load(w_ptr + c_offs, mask=mask, other=1.0)
            bv = tl.load(b_ptr + c_offs, mask=mask, other=0.0)
            yv = yv * wv + bv
        g_min = tl.min(tl.where(mask, yv, pos_inf), axis=0)
        row_min = tl.minimum(row_min, g_min)
    tl.store(out_ptr + row.to(tl.int64), row_min)


def groupnorm_2d_row_min(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    num_groups: int,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    """Single-dispatch 2D `(N, C)` GroupNorm fused with `min(dim=1, keepdim=True)`."""
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    g = int(num_groups)
    gs = c // g
    if out is None:
        out = torch.empty((n, 1), dtype=x_c.dtype, device=x_c.device)
    has_affine = 1 if (weight is not None and bias is not None) else 0
    w_t = _contig(weight) if has_affine else x_c
    b_t = _contig(bias) if has_affine else x_c
    block_gs = max(16, _next_pow2(gs))
    groupnorm_2d_row_min_kernel[(n,)](
        x_c,
        w_t,
        b_t,
        out,
        c,
        g,
        gs,
        float(eps),
        HAS_AFFINE=has_affine,
        BLOCK_GS=block_gs,
        sync_mps=sync_mps,
        wait=wait,
    )
    return out


@m1triton.jit
def add_mul_y_vec4_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0)
    yv = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (xv + yv) * yv, mask=mask)


def row_instancenorm_add_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-5,
    norm_buf: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    """Vectorized Welford per-row InstanceNorm followed by `(norm_x + y) * y`."""
    x_c = _contig(x)
    y_c = _contig(y)
    m, n = int(x_c.shape[0]), int(x_c.shape[1])
    if norm_buf is None:
        norm_buf = torch.empty_like(x_c)
    if out is None:
        out = torch.empty_like(x_c)
    welford_instance_norm_rows_kernel[(m, 1, 1)](
        x_c, norm_buf, m, n, float(eps), sync_mps=sync_mps, wait=False
    )
    total = m * n
    block = 1024
    add_mul_y_vec4_kernel[(m1triton.cdiv(total, block),)](
        norm_buf, y_c, out, total, BLOCK=block, sync_mps=False, wait=wait
    )
    return out


@m1triton.jit
def broadcast_row_add_2d_kernel(
    row_ptr,
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    tlx.vec_width(4)
    pid_c = tl.program_id(0)
    row = tl.program_id(1)
    offs = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    rv = tl.load(row_ptr + row)
    idx = row.to(tl.int64) * N.to(tl.int64) + offs.to(tl.int64)
    xv = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, xv + rv, mask=mask)


def broadcast_row_add_2d(
    row_vec: torch.Tensor,
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    """Compute `row_vec (M, 1) + x (M, N) -> (M, N)` in 1 vectorized kernel."""
    r_c = _contig(row_vec)
    x_c = _contig(x)
    m, n = int(x_c.shape[0]), int(x_c.shape[1])
    if out is None:
        out = torch.empty_like(x_c)
    block_n = 1024
    broadcast_row_add_2d_kernel[(m1triton.cdiv(n, block_n), m)](
        r_c, x_c, out, n, BLOCK_N=block_n, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def broadcast_outer_add_kernel(
    row_ptr,
    bias_ptr,
    out_ptr,
    M,
    BLOCK_M: tl.constexpr,
):
    tlx.vec_width(4)
    pid_m = tl.program_id(0)
    c = tl.program_id(1)
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M
    bv = tl.load(bias_ptr + c)
    rv = tl.load(row_ptr + offs, mask=mask, other=0.0)
    idx = c.to(tl.int64) * M.to(tl.int64) + offs.to(tl.int64)
    tl.store(out_ptr + idx, rv + bv, mask=mask)


def broadcast_outer_add_4d(
    row_min: torch.Tensor,
    bias: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    """Compute `row_min (M, 1) + bias (1, C, 1, 1) -> (1, C, M, 1)` in 1 vectorized kernel."""
    r_c = _contig(row_min)
    b_c = _contig(bias)
    m = int(r_c.shape[0])
    c = int(b_c.shape[1])
    if out is None:
        out = torch.empty((1, c, m, 1), dtype=r_c.dtype, device=r_c.device)
    block_m = 1024
    broadcast_outer_add_kernel[(m1triton.cdiv(m, block_m), c)](
        r_c, b_c, out, m, BLOCK_M=block_m, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def axis_max_dim0_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offs < N.to(tl.int64)
    p = x_ptr + offs
    acc = tl.load(p, mask=mask, other=-float("inf"))
    p = p + N.to(tl.int64)
    for _ in range(1, M):
        v = tl.load(p, mask=mask, other=-float("inf"))
        acc = tl.maximum(acc, v)
        p = p + N.to(tl.int64)
    tl.store(out_ptr + offs, acc, mask=mask)


@m1triton.jit
def pool1d_fallback_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    W_in,
    W_out,
    KIND: tl.constexpr,
    K_W: tl.constexpr,
    S_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tlx.pool(
        x_ptr,
        out_ptr,
        N,
        C,
        W_in,
        W_out,
        kind=KIND,
        ndim=1,
        kernel_size=(K_W,),
        stride=(S_W,),
        padding=(0,),
        dilation=(1,),
        count_include_pad=True,
        ceil_mode=False,
    )


def pool1d_2d_fallback(
    x: torch.Tensor,
    *,
    kind: str,
    kernel_size: int,
    sync_mps: bool = True,
    wait: bool = False,
) -> torch.Tensor:
    x_c = _contig(x)
    m, w_in = int(x_c.shape[0]), int(x_c.shape[1])
    k = int(kernel_size)
    w_out = (w_in - k) // k + 1
    out = torch.empty((m, w_out), dtype=x_c.dtype, device=x_c.device)
    block_w = 256
    pool1d_fallback_kernel[(m1triton.cdiv(w_out, block_w), 1, m)](
        x_c,
        out,
        m,
        1,
        w_in,
        w_out,
        KIND=kind,
        K_W=k,
        S_W=k,
        BLOCK_W=block_w,
        sync_mps=sync_mps,
        wait=wait,
    )
    return out


__all__ = [
    "axis_max_dim0_kernel",
    "bn_fold_affine_gpu",
    "broadcast_outer_add_4d",
    "broadcast_row_add_2d",
    "copy_tensor_",
    "finalize_row_lse",
    "finalize_row_max",
    "finalize_row_sum",
    "fused_linear_epilogue",
    "groupnorm_2d_fused",
    "groupnorm_2d_row_min",
    "m1_runtime",
    "pool1d_2d_fallback",
    "row_instancenorm_add_mul",
    "row_mean_sub_gelu",
    "sgmm_fused_epilogue_kernel",
    "softmax_rows",
]
