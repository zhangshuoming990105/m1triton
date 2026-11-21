"""Reusable M1Triton coalesced strided and last-axis reduction kernels."""

from __future__ import annotations

import math

import torch

import m1triton
import m1triton.language as tl


@m1triton.jit
def axis_sum_strided_kernel(
    x_ptr,
    out_ptr,
    R,
    inner,
    scale,
    BLOCK_INNER: tl.constexpr,
    APPLY_SCALE: tl.constexpr = 0,
):
    pid_c = tl.program_id(0)
    pid_o = tl.program_id(1)
    r_i64 = R.to(tl.int64)
    inner_i64 = inner.to(tl.int64)
    base_o = pid_o.to(tl.int64) * (r_i64 * inner_i64)
    out_base = pid_o.to(tl.int64) * inner_i64

    offs = pid_c.to(tl.int64) * BLOCK_INNER + tl.arange(0, BLOCK_INNER).to(tl.int64)
    mask = offs < inner_i64
    p = x_ptr + (base_o + offs)
    acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
    for _ in range(0, R):
        acc = acc + tl.load(p, mask=mask, other=0.0)
        p = p + inner_i64
    if APPLY_SCALE:
        acc = acc * scale
    tl.store(out_ptr + (out_base + offs), acc, mask=mask)


@m1triton.jit
def axis_extrema_strided_kernel(
    x_ptr,
    out_ptr,
    R,
    inner,
    BLOCK_INNER: tl.constexpr,
    IS_MAX: tl.constexpr = 1,
):
    pid_c = tl.program_id(0)
    pid_o = tl.program_id(1)
    r_i64 = R.to(tl.int64)
    inner_i64 = inner.to(tl.int64)
    base_o = pid_o.to(tl.int64) * (r_i64 * inner_i64)
    out_base = pid_o.to(tl.int64) * inner_i64

    offs = pid_c.to(tl.int64) * BLOCK_INNER + tl.arange(0, BLOCK_INNER).to(tl.int64)
    mask = offs < inner_i64
    p = x_ptr + (base_o + offs)
    acc = tl.load(p, mask=mask, other=0.0)
    p = p + inner_i64
    for _ in range(1, R):
        v = tl.load(p, mask=mask, other=0.0)
        if IS_MAX:
            acc = tl.maximum(acc, v)
        else:
            acc = tl.minimum(acc, v)
        p = p + inner_i64
    tl.store(out_ptr + (out_base + offs), acc, mask=mask)


@m1triton.jit
def axis_argextrema_strided_kernel(
    x_ptr,
    out_ptr,
    R,
    inner,
    BLOCK_INNER: tl.constexpr,
    IS_MAX: tl.constexpr = 1,
):
    pid_c = tl.program_id(0)
    pid_o = tl.program_id(1)
    r_i64 = R.to(tl.int64)
    inner_i64 = inner.to(tl.int64)
    base_o = pid_o.to(tl.int64) * (r_i64 * inner_i64)
    out_base = pid_o.to(tl.int64) * inner_i64

    offs = pid_c.to(tl.int64) * BLOCK_INNER + tl.arange(0, BLOCK_INNER).to(tl.int64)
    full_tile = (pid_c.to(tl.int64) + 1) * BLOCK_INNER <= inner_i64
    p = x_ptr + (base_o + offs)
    if full_tile:
        best_val = tl.load(p)
        best_idx = tl.zeros((BLOCK_INNER,), dtype=tl.int32)
        p = p + inner_i64
        for r in range(1, R):
            v = tl.load(p)
            if IS_MAX:
                bt = v > best_val
            else:
                bt = v < best_val
            best_val = tl.where(bt, v, best_val)
            best_idx = tl.where(bt, r, best_idx)
            p = p + inner_i64
        tl.store(out_ptr + (out_base + offs), best_idx.to(tl.int64))
    else:
        mask = offs < inner_i64
        best_val = tl.load(p, mask=mask, other=0.0)
        best_idx = tl.zeros((BLOCK_INNER,), dtype=tl.int32)
        p = p + inner_i64
        for r in range(1, R):
            v = tl.load(p, mask=mask, other=0.0)
            if IS_MAX:
                bt = v > best_val
            else:
                bt = v < best_val
            best_val = tl.where(bt, v, best_val)
            best_idx = tl.where(bt, r, best_idx)
            p = p + inner_i64
        tl.store(out_ptr + (out_base + offs), best_idx.to(tl.int64), mask=mask)


@m1triton.jit
def last_axis_sum_kernel(
    x_ptr,
    out_ptr,
    R,
    scale,
    BLOCK_K: tl.constexpr,
    APPLY_SCALE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * R.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, R, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < R
        vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=0.0)
        acc = acc + vals
    res = tl.sum(acc, axis=0)
    if APPLY_SCALE:
        res = res * scale
    tl.store(out_ptr + row.to(tl.int64), res)


@m1triton.jit
def last_axis_extrema_kernel(
    x_ptr,
    out_ptr,
    R,
    BLOCK_K: tl.constexpr,
    IS_MAX: tl.constexpr = 1,
):
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * R.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    if IS_MAX:
        init_v = -float("inf")
    else:
        init_v = float("inf")
    acc = tl.full((BLOCK_K,), init_v, dtype=tl.float32)
    for k in range(0, R, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < R
        vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=init_v)
        if IS_MAX:
            acc = tl.maximum(acc, vals)
        else:
            acc = tl.minimum(acc, vals)
    if IS_MAX:
        res = tl.max(acc, axis=0)
    else:
        res = tl.min(acc, axis=0)
    tl.store(out_ptr + row.to(tl.int64), res)


@m1triton.jit
def last_axis_argextrema_kernel(
    x_ptr,
    out_ptr,
    R,
    BLOCK_K: tl.constexpr,
    IS_MAX: tl.constexpr = 1,
):
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * R.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    if IS_MAX:
        init_v = -float("inf")
    else:
        init_v = float("inf")
    best_val = init_v
    best_idx = 0
    for k in range(0, R, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < R
        vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=init_v)
        if IS_MAX:
            tile_v = tl.max(vals, axis=0)
            tile_i = tl.argmax(vals, axis=0)
            better = tile_v > best_val
        else:
            tile_v = tl.min(vals, axis=0)
            tile_i = tl.argmin(vals, axis=0)
            better = tile_v < best_val
        best_val = tl.where(better, tile_v, best_val)
        best_idx = tl.where(better, k + tile_i, best_idx)
    tl.store(out_ptr + row.to(tl.int64), best_idx.to(tl.int64))


def _prepare_reduction(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    ndim = x.ndim
    if ndim == 0:
        raise ValueError("Cannot reduce a 0-D tensor along a dimension")
    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], got {dim})")

    x_c = x if x.is_contiguous() else x.contiguous()
    shape = tuple(int(s) for s in x_c.shape)
    outer = math.prod(shape[:d]) if d > 0 else 1
    reduce_len = shape[d]
    inner = math.prod(shape[d + 1 :]) if (d + 1) < ndim else 1
    if keepdim:
        out_shape = shape[:d] + (1,) + shape[d + 1 :]
    else:
        out_shape = shape[:d] + shape[d + 1 :]
    out = torch.empty(out_shape, dtype=out_dtype, device=x_c.device)
    return x_c, out, outer, reduce_len, inner


def _select_last_axis_block_k(reduce_len: int) -> int:
    return 256 if reduce_len <= 256 else 1024


def axis_sum(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=x.dtype)
    if outer == 0 or inner == 0:
        return out
    if reduce_len == 0:
        out.zero_()
        return out
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_sum_kernel[(outer,)](x_c, out, reduce_len, 1.0, BLOCK_K=block_k, APPLY_SCALE=0)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_sum_strided_kernel[grid](x_c, out, reduce_len, inner, 1.0, BLOCK_INNER=block_inner, APPLY_SCALE=0)
    return out


def axis_mean(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=x.dtype)
    if outer == 0 or inner == 0:
        return out
    if reduce_len == 0:
        out.fill_(float("nan"))
        return out
    scale = 1.0 / float(reduce_len)
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_sum_kernel[(outer,)](x_c, out, reduce_len, scale, BLOCK_K=block_k, APPLY_SCALE=1)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_sum_strided_kernel[grid](x_c, out, reduce_len, inner, scale, BLOCK_INNER=block_inner, APPLY_SCALE=1)
    return out


def axis_max(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=x.dtype)
    if reduce_len == 0:
        raise RuntimeError("axis_max: reduction dimension has size 0")
    if outer == 0 or inner == 0:
        return out
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_extrema_kernel[(outer,)](x_c, out, reduce_len, BLOCK_K=block_k, IS_MAX=1)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_extrema_strided_kernel[grid](x_c, out, reduce_len, inner, BLOCK_INNER=block_inner, IS_MAX=1)
    return out


def axis_min(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=x.dtype)
    if reduce_len == 0:
        raise RuntimeError("axis_min: reduction dimension has size 0")
    if outer == 0 or inner == 0:
        return out
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_extrema_kernel[(outer,)](x_c, out, reduce_len, BLOCK_K=block_k, IS_MAX=0)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_extrema_strided_kernel[grid](x_c, out, reduce_len, inner, BLOCK_INNER=block_inner, IS_MAX=0)
    return out


def axis_argmax(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=torch.int64)
    if reduce_len == 0:
        raise RuntimeError("axis_argmax: reduction dimension has size 0")
    if outer == 0 or inner == 0:
        return out
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_argextrema_kernel[(outer,)](x_c, out, reduce_len, BLOCK_K=block_k, IS_MAX=1)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_argextrema_strided_kernel[grid](x_c, out, reduce_len, inner, BLOCK_INNER=block_inner, IS_MAX=1)
    return out


def axis_argmin(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    block_inner: int = 128,
) -> torch.Tensor:
    x_c, out, outer, reduce_len, inner = _prepare_reduction(x, dim, keepdim=keepdim, out_dtype=torch.int64)
    if reduce_len == 0:
        raise RuntimeError("axis_argmin: reduction dimension has size 0")
    if outer == 0 or inner == 0:
        return out
    if inner == 1:
        block_k = _select_last_axis_block_k(reduce_len)
        last_axis_argextrema_kernel[(outer,)](x_c, out, reduce_len, BLOCK_K=block_k, IS_MAX=0)
    else:
        grid = (m1triton.cdiv(inner, block_inner), outer)
        axis_argextrema_strided_kernel[grid](x_c, out, reduce_len, inner, BLOCK_INNER=block_inner, IS_MAX=0)
    return out


__all__ = [
    "axis_argmax",
    "axis_argmin",
    "axis_argextrema_strided_kernel",
    "axis_extrema_strided_kernel",
    "axis_max",
    "axis_mean",
    "axis_min",
    "axis_sum",
    "axis_sum_strided_kernel",
    "last_axis_argextrema_kernel",
    "last_axis_extrema_kernel",
    "last_axis_sum_kernel",
]
