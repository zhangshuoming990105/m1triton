"""PyTorch 2.x `torch.compile(model, backend="m1triton")` FX graph fusion compiler.

Reuses PyTorch TorchDynamo + AOTAutograd to capture ATen FX graphs, greedily fuses
pointwise/broadcast DAGs and producer-epilogue patterns into cached `@m1triton.jit`
and `tlx.*` Metal kernels, and batches sequential M1Triton dispatches inside a single
`m1triton.runtime.command_batch()` with race-free MPS synchronization at partition
boundaries.
"""

from __future__ import annotations

import builtins
import hashlib
import linecache
import math
import operator
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch._dynamo as _dynamo
import torch.fx as fx
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils.weak import TensorWeakRef

import m1triton
from m1triton import language as tl
from m1triton import runtime as rt
from m1triton import tlx

aten = torch.ops.aten

_SUPPORTED_PW_DTYPES = frozenset(
    {
        torch.float32,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.bool,
    }
)
_SUPPORTED_FLOAT_DTYPES = frozenset({torch.float32, torch.float16})
_SUPPORTED_DEVICES = frozenset({"mps", "cpu"})
_INT32_FAST_LIMIT = 2_000_000_000


def _env_strict() -> bool:
    return os.environ.get("M1TRITON_STRICT_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _torch_dtype_to_tl_name(dt: torch.dtype) -> str:
    mapping = {
        torch.float32: "tl.float32",
        torch.float16: "tl.float16",
        torch.int32: "tl.int32",
        torch.int64: "tl.int64",
        torch.bool: "tl.bool_",
    }
    if dt not in mapping:
        raise NotImplementedError(f"Unsupported M1Triton dtype: {dt}")
    return mapping[dt]


def _float_literal(val: float | int | bool) -> str:
    if isinstance(val, bool):
        return "1.0" if val else "0.0"
    f = float(val)
    if math.isnan(f):
        return "float('nan')"
    if math.isinf(f):
        return "float('inf')" if f > 0 else "-float('inf')"
    return repr(f)


def _scalar_literal(val: float | int | bool) -> str:
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, int):
        return repr(int(val))
    return _float_literal(val)


def _compile_generated_jit(func_name: str, source: str, *, fast_math: bool = False) -> m1triton.JITFunction:
    filename = f"<m1triton_fx_{func_name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    ns: dict[str, Any] = {"m1triton": m1triton, "tl": tl, "tlx": tlx}
    code = builtins.compile(source, filename, "exec")
    exec(code, ns)
    fn_obj = ns[func_name]
    if fast_math and isinstance(fn_obj, m1triton.JITFunction):
        fn_obj.fast_math = True
    return fn_obj


# ============================================================================
# Core M1Triton Kernels (GEMM, Conv, Pool, Softmax, Norm, Reduction, Fill)
# ============================================================================


@m1triton.jit
def _sgmm_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr = 4,
    SWIZZLE_W: tl.constexpr = 8,
    TRANS_A: tl.constexpr = 0,
    TRANS_B: tl.constexpr = 0,
    N_MAJOR: tl.constexpr = 0,
    IS_BATCHED: tl.constexpr = 0,
    ALIGNED_MNK: tl.constexpr = 0,
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
    if N_MAJOR == 0:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    else:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)

    if IS_BATCHED == 0:
        a_base = a_ptr
        b_base = b_ptr
        c_base = c_ptr
    else:
        pid_b = tl.program_id(2)
        a_base = a_ptr + pid_b.to(tl.int64) * stride_ab
        b_base = b_ptr + pid_b.to(tl.int64) * stride_bb
        c_base = c_ptr + pid_b.to(tl.int64) * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        k_idx64 = k_idx.to(tl.int64)
        if TRANS_A == 0:
            a_ptrs = a_base + offs_m64[:, None] * stride_am + k_idx64[None, :]
        else:
            a_ptrs = a_base + offs_m64[:, None] + k_idx64[None, :] * stride_ak
        if TRANS_B == 0:
            b_ptrs = b_base + k_idx64[:, None] * stride_bk + offs_n64[None, :]
        else:
            b_ptrs = b_base + k_idx64[:, None] + offs_n64[None, :] * stride_bn
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_base + offs_m64[:, None] * stride_cm + offs_n64[None, :]
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


@m1triton.jit
def _conv_std_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    D_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    D_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_D: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_D: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    NDIM: tl.constexpr = 2,
    HAS_BIAS: tl.constexpr = 0,
):
    if NDIM == 1:
        if HAS_BIAS:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, W_in, W_out,
                ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,), groups=GROUPS, has_bias=True,
            )
        else:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, N, C_in, C_out, W_in, W_out,
                ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,), groups=GROUPS,
            )
    elif NDIM == 2:
        if HAS_BIAS:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
                ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
                groups=GROUPS, has_bias=True,
            )
        else:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
                ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
                groups=GROUPS,
            )
    else:
        if HAS_BIAS:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
                ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
                dilation=(D_D, D_H, D_W), groups=GROUPS, has_bias=True,
            )
        else:
            tlx.implicit_gemm_conv(
                x_ptr, w_ptr, out_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
                ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
                dilation=(D_D, D_H, D_W), groups=GROUPS,
            )


@m1triton.jit
def _conv_dw2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    N,
    C_in,
    C_out,
    H_in,
    W_in,
    H_out,
    W_out,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VEC_W: tl.constexpr = 1,
    ROWS_PER_THREAD: tl.constexpr = 0,
    HAS_BIAS: tl.constexpr = 0,
):
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, bias_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
            is_depthwise=True, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
            is_depthwise=True, has_bias=False,
        )


@m1triton.jit
def _wino2x3_filter_kernel(
    w_ptr,
    u_ptr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    C_IN_PAD: tl.constexpr,
    C_OUT_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    total = C_OUT_PAD * C_IN_PAD
    valid = idx < total
    co = idx // C_IN_PAD
    ci = idx - co * C_IN_PAD
    in_bounds = valid & (co < C_out) & (ci < C_in)
    base = (co * C_in + ci) * 9
    g00 = tl.load(w_ptr + (base + 0), mask=in_bounds, other=0.0)
    g01 = tl.load(w_ptr + (base + 1), mask=in_bounds, other=0.0)
    g02 = tl.load(w_ptr + (base + 2), mask=in_bounds, other=0.0)
    g10 = tl.load(w_ptr + (base + 3), mask=in_bounds, other=0.0)
    g11 = tl.load(w_ptr + (base + 4), mask=in_bounds, other=0.0)
    g12 = tl.load(w_ptr + (base + 5), mask=in_bounds, other=0.0)
    g20 = tl.load(w_ptr + (base + 6), mask=in_bounds, other=0.0)
    g21 = tl.load(w_ptr + (base + 7), mask=in_bounds, other=0.0)
    g22 = tl.load(w_ptr + (base + 8), mask=in_bounds, other=0.0)

    s0, s1, s2 = g00 + g20, g01 + g21, g02 + g22
    t00, t01, t02 = g00, g01, g02
    t10, t11, t12 = 0.5 * (s0 + g10), 0.5 * (s1 + g11), 0.5 * (s2 + g12)
    t20, t21, t22 = 0.5 * (s0 - g10), 0.5 * (s1 - g11), 0.5 * (s2 - g12)
    t30, t31, t32 = g20, g21, g22

    r0 = t00 + t02
    u0, u1, u2, u3 = t00, 0.5 * (r0 + t01), 0.5 * (r0 - t01), t02
    r1 = t10 + t12
    u4, u5, u6, u7 = t10, 0.5 * (r1 + t11), 0.5 * (r1 - t11), t12
    r2 = t20 + t22
    u8, u9, u10, u11 = t20, 0.5 * (r2 + t21), 0.5 * (r2 - t21), t22
    r3 = t30 + t32
    u12, u13, u14, u15 = t30, 0.5 * (r3 + t31), 0.5 * (r3 - t31), t32

    ci_blk = ci // 8
    ci_loc = ci - ci_blk * 8
    co_blk = co // 8
    co_loc = co - co_blk * 8
    c_out_blks = C_OUT_PAD // 8
    alpha_stride = (C_IN_PAD // 8) * c_out_blks * 64
    dst = (ci_blk * c_out_blks + co_blk) * 64 + co_loc * 8 + ci_loc

    tl.store(u_ptr + (0 * alpha_stride + dst), u0, mask=valid)
    tl.store(u_ptr + (1 * alpha_stride + dst), u1, mask=valid)
    tl.store(u_ptr + (2 * alpha_stride + dst), u2, mask=valid)
    tl.store(u_ptr + (3 * alpha_stride + dst), u3, mask=valid)
    tl.store(u_ptr + (4 * alpha_stride + dst), u4, mask=valid)
    tl.store(u_ptr + (5 * alpha_stride + dst), u5, mask=valid)
    tl.store(u_ptr + (6 * alpha_stride + dst), u6, mask=valid)
    tl.store(u_ptr + (7 * alpha_stride + dst), u7, mask=valid)
    tl.store(u_ptr + (8 * alpha_stride + dst), u8, mask=valid)
    tl.store(u_ptr + (9 * alpha_stride + dst), u9, mask=valid)
    tl.store(u_ptr + (10 * alpha_stride + dst), u10, mask=valid)
    tl.store(u_ptr + (11 * alpha_stride + dst), u11, mask=valid)
    tl.store(u_ptr + (12 * alpha_stride + dst), u12, mask=valid)
    tl.store(u_ptr + (13 * alpha_stride + dst), u13, mask=valid)
    tl.store(u_ptr + (14 * alpha_stride + dst), u14, mask=valid)
    tl.store(u_ptr + (15 * alpha_stride + dst), u15, mask=valid)


@m1triton.jit
def _conv2d_wino2x3_kernel(
    x_ptr,
    u_ptr,
    out_ptr,
    b_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_TH: tl.constexpr,
    TILE_TW: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
    WINOGRAD: tl.constexpr = 1,
    WINO_PAD: tl.constexpr = 1,
    HAS_BIAS: tl.constexpr = 0,
):
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, u_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1), groups=1, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, u_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1), groups=1,
        )


@m1triton.jit
def _pool_nd_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    KIND: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_D: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_D: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    CEIL_MODE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr = 1,
    VEC_W: tl.constexpr = 1,
    ROWS_PER_THREAD: tl.constexpr = 0,
    NDIM: tl.constexpr = 2,
):
    if NDIM == 1:
        tlx.pool(
            x_ptr, out_ptr, N, C, W_in, W_out,
            kind=KIND, ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,),
            count_include_pad=COUNT_INCLUDE_PAD, ceil_mode=CEIL_MODE,
        )
    elif NDIM == 2:
        tlx.pool(
            x_ptr, out_ptr, N, C, H_in, W_in, H_out, W_out,
            kind=KIND, ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
            count_include_pad=COUNT_INCLUDE_PAD, ceil_mode=CEIL_MODE,
        )
    else:
        tlx.pool(
            x_ptr, out_ptr, N, C, D_in, H_in, W_in, D_out, H_out, W_out,
            kind=KIND, ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
            dilation=(D_D, D_H, D_W), count_include_pad=COUNT_INCLUDE_PAD, ceil_mode=CEIL_MODE,
        )


@m1triton.jit
def _max_pool2d_with_indices_kernel(
    x_ptr,
    out_ptr,
    idx_ptr,
    NC,
    H_in,
    W_in,
    H_out,
    W_out,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    h_out_i64 = H_out.to(tl.int64)
    w_out_i64 = W_out.to(tl.int64)
    h_in_i64 = H_in.to(tl.int64)
    w_in_i64 = W_in.to(tl.int64)
    total = NC.to(tl.int64) * h_out_i64 * w_out_i64
    mask = offs < total
    ow = offs % w_out_i64
    tmp = offs // w_out_i64
    oh = tmp % h_out_i64
    nc = tmp // h_out_i64
    in_base = nc * (h_in_i64 * w_in_i64)

    best_val = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    has_best = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) != 0
    for kh in tl.static_range(0, K_H):
        ih = oh * S_H - P_H + kh * D_H
        valid_h = (ih >= 0) & (ih < h_in_i64)
        for kw in tl.static_range(0, K_W):
            iw = ow * S_W - P_W + kw * D_W
            valid = mask & valid_h & (iw >= 0) & (iw < w_in_i64)
            spatial_idx = ih * w_in_i64 + iw
            v = tl.load(x_ptr + (in_base + spatial_idx), mask=valid, other=-float("inf"))
            is_nan_v = v != v
            better = valid & ((~has_best) | (v > best_val) | (is_nan_v & (best_val == best_val)))
            best_val = tl.where(better, v, best_val)
            best_idx = tl.where(better, spatial_idx, best_idx)
            has_best = has_best | valid
    tl.store(out_ptr + offs, best_val, mask=mask)
    tl.store(idx_ptr + offs, best_idx, mask=mask)


@m1triton.jit
def _online_softmax_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
    LOG_SOFTMAX: tl.constexpr,
):
    tlx.online_softmax(x_ptr, out_ptr, NUM_ROWS, ROW_LEN, axis=1, log_softmax=LOG_SOFTMAX)


@m1triton.jit
def _strided_softmax_kernel(
    x_ptr,
    out_ptr,
    R,
    inner,
    BLOCK_INNER: tl.constexpr,
    LOG_SOFTMAX: tl.constexpr = 0,
):
    pid_c = tl.program_id(0)
    pid_o = tl.program_id(1)
    r_i64 = R.to(tl.int64)
    inner_i64 = inner.to(tl.int64)
    base_o = pid_o.to(tl.int64) * (r_i64 * inner_i64)
    offs = pid_c.to(tl.int64) * BLOCK_INNER + tl.arange(0, BLOCK_INNER).to(tl.int64)
    mask = offs < inner_i64

    p = x_ptr + (base_o + offs)
    max_v = tl.full((BLOCK_INNER,), -float("inf"), dtype=tl.float32)
    for _ in range(0, R):
        v = tl.load(p, mask=mask, other=-float("inf"))
        max_v = tl.maximum(max_v, v)
        p = p + inner_i64

    p = x_ptr + (base_o + offs)
    denom = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
    for _ in range(0, R):
        v = tl.load(p, mask=mask, other=-float("inf"))
        denom = denom + tl.exp(v - max_v)
        p = p + inner_i64

    if LOG_SOFTMAX:
        log_denom = tl.log(denom)
    else:
        inv_denom = 1.0 / denom

    p_in = x_ptr + (base_o + offs)
    p_out = out_ptr + (base_o + offs)
    for _ in range(0, R):
        v = tl.load(p_in, mask=mask, other=0.0)
        if LOG_SOFTMAX:
            res = (v - max_v) - log_denom
        else:
            res = tl.exp(v - max_v) * inv_denom
        tl.store(p_out, res, mask=mask)
        p_in = p_in + inner_i64
        p_out = p_out + inner_i64


@m1triton.jit
def _welford_norm_stats_kernel(
    x_ptr,
    partial_ptr,
    N: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    CHUNKS: tl.constexpr,
    KIND: tl.constexpr = 0,
    GROUPS: tl.constexpr = 1,
):
    if KIND == 0:
        tlx.welford_norm("layer_norm", x_ptr, partial_ptr, N, D1, CHUNKS, stage=1)
    elif KIND == 1:
        tlx.welford_norm("group_norm", x_ptr, partial_ptr, N, D1, D2, CHUNKS, num_groups=GROUPS, stage=1, axis=1)
    else:
        tlx.welford_norm("batch_norm", x_ptr, partial_ptr, N, D1, D2, CHUNKS, stage=1, axis=1)


@m1triton.jit
def _welford_norm_apply_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    N: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    CHUNKS: tl.constexpr,
    EPS: tl.constexpr,
    KIND: tl.constexpr = 0,
    HAS_WEIGHT: tl.constexpr = 0,
    HAS_BIAS: tl.constexpr = 0,
    GROUPS: tl.constexpr = 1,
):
    if KIND == 0:
        if HAS_WEIGHT and HAS_BIAS:
            tlx.welford_norm("layer_norm", x_ptr, out_ptr, weight_ptr, bias_ptr, partial_ptr, N, D1, CHUNKS, eps=EPS, stage=2, has_weight=True, has_bias=True)
        elif HAS_WEIGHT:
            tlx.welford_norm("layer_norm", x_ptr, out_ptr, weight_ptr, partial_ptr, N, D1, CHUNKS, eps=EPS, stage=2, has_weight=True, has_bias=False)
        elif HAS_BIAS:
            tlx.welford_norm("layer_norm", x_ptr, out_ptr, bias_ptr, partial_ptr, N, D1, CHUNKS, eps=EPS, stage=2, has_weight=False, has_bias=True)
        else:
            tlx.welford_norm("layer_norm", x_ptr, out_ptr, partial_ptr, N, D1, CHUNKS, eps=EPS, stage=2, has_weight=False, has_bias=False)
    elif KIND == 1:
        if HAS_WEIGHT and HAS_BIAS:
            tlx.welford_norm("group_norm", x_ptr, out_ptr, weight_ptr, bias_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, num_groups=GROUPS, stage=2, axis=1, has_weight=True, has_bias=True)
        elif HAS_WEIGHT:
            tlx.welford_norm("group_norm", x_ptr, out_ptr, weight_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, num_groups=GROUPS, stage=2, axis=1, has_weight=True, has_bias=False)
        elif HAS_BIAS:
            tlx.welford_norm("group_norm", x_ptr, out_ptr, bias_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, num_groups=GROUPS, stage=2, axis=1, has_weight=False, has_bias=True)
        else:
            tlx.welford_norm("group_norm", x_ptr, out_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, num_groups=GROUPS, stage=2, axis=1, has_weight=False, has_bias=False)
    else:
        if HAS_WEIGHT and HAS_BIAS:
            tlx.welford_norm("batch_norm", x_ptr, out_ptr, weight_ptr, bias_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, stage=2, axis=1, has_weight=True, has_bias=True)
        elif HAS_WEIGHT:
            tlx.welford_norm("batch_norm", x_ptr, out_ptr, weight_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, stage=2, axis=1, has_weight=True, has_bias=False)
        elif HAS_BIAS:
            tlx.welford_norm("batch_norm", x_ptr, out_ptr, bias_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, stage=2, axis=1, has_weight=False, has_bias=True)
        else:
            tlx.welford_norm("batch_norm", x_ptr, out_ptr, partial_ptr, N, D1, D2, CHUNKS, eps=EPS, stage=2, axis=1, has_weight=False, has_bias=False)


@m1triton.jit
def _batch_norm_eval_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    C,
    HW,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tlx.vec_width(4)
    pid_hw = tl.program_id(0)
    nc = tl.program_id(1)
    c_idx = nc % C
    hw_i64 = HW.to(tl.int64)
    offs = pid_hw.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < hw_i64
    base = nc.to(tl.int64) * hw_i64
    m = tl.load(mean_ptr + c_idx)
    v = tl.load(var_ptr + c_idx)
    inv_std = tl.rsqrt(v + eps)
    x = tl.load(x_ptr + (base + offs), mask=mask, other=0.0)
    y = (x - m) * inv_std
    if HAS_WEIGHT:
        w = tl.load(weight_ptr + c_idx)
        y = y * w
    if HAS_BIAS:
        b = tl.load(bias_ptr + c_idx)
        y = y + b
    tl.store(out_ptr + (base + offs), y, mask=mask)


@m1triton.jit
def _welford_partials_extract_kernel(
    partial_ptr,
    mean_ptr,
    rstd_ptr,
    num_groups,
    CHUNKS: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < num_groups.to(tl.int64)
    n_tot = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    mean_tot = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    m2_tot = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for ch in tl.static_range(0, CHUNKS):
        slot = (offs * CHUNKS + ch) * 3
        n_b = tl.load(partial_ptr + slot, mask=mask, other=0.0)
        m_b = tl.load(partial_ptr + (slot + 1), mask=mask, other=0.0)
        m2_b = tl.load(partial_ptr + (slot + 2), mask=mask, other=0.0)
        n_new = n_tot + n_b
        delta = m_b - mean_tot
        safe_n = tl.where(n_new > 0.0, n_new, 1.0)
        mean_new = mean_tot + delta * (n_b / safe_n)
        m2_new = m2_tot + m2_b + delta * delta * (n_tot * n_b / safe_n)
        n_tot = n_new
        mean_tot = tl.where(n_new > 0.0, mean_new, mean_tot)
        m2_tot = tl.where(n_new > 0.0, m2_new, m2_tot)
    safe_final = tl.where(n_tot > 0.0, n_tot, 1.0)
    rstd = tl.rsqrt(m2_tot / safe_final + eps)
    tl.store(mean_ptr + offs, mean_tot, mask=mask)
    tl.store(rstd_ptr + offs, rstd, mask=mask)


@m1triton.jit
def _axis_reduce_strided_kernel(
    x_ptr,
    out_ptr,
    R,
    inner,
    scale,
    BLOCK_INNER: tl.constexpr,
    OP_KIND: tl.constexpr = 0,
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
    if OP_KIND == 0:
        acc = tl.zeros((BLOCK_INNER,), dtype=tl.float32)
        for _ in range(0, R):
            acc = acc + tl.load(p, mask=mask, other=0.0)
            p = p + inner_i64
        if APPLY_SCALE:
            acc = acc * scale
    else:
        acc = tl.load(p, mask=mask, other=0.0)
        p = p + inner_i64
        for _ in range(1, R):
            v = tl.load(p, mask=mask, other=0.0)
            if OP_KIND == 1:
                acc = tl.maximum(acc, v)
            else:
                acc = tl.minimum(acc, v)
            p = p + inner_i64
    tl.store(out_ptr + (out_base + offs), acc, mask=mask)


@m1triton.jit
def _last_axis_reduce_kernel(
    x_ptr,
    out_ptr,
    R,
    scale,
    BLOCK_K: tl.constexpr,
    OP_KIND: tl.constexpr = 0,
    APPLY_SCALE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * R.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    if OP_KIND == 0:
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k in range(0, R, BLOCK_K):
            k_offs = k + ar
            k_mask = k_offs < R
            vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=0.0)
            acc = acc + vals
        res = tl.sum(acc, axis=0)
        if APPLY_SCALE:
            res = res * scale
    else:
        if OP_KIND == 1:
            init_v = -float("inf")
        else:
            init_v = float("inf")
        acc = tl.full((BLOCK_K,), init_v, dtype=tl.float32)
        for k in range(0, R, BLOCK_K):
            k_offs = k + ar
            k_mask = k_offs < R
            vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=init_v)
            if OP_KIND == 1:
                acc = tl.maximum(acc, vals)
            else:
                acc = tl.minimum(acc, vals)
        if OP_KIND == 1:
            res = tl.max(acc, axis=0)
        else:
            res = tl.min(acc, axis=0)
    tl.store(out_ptr + row.to(tl.int64), res)


@m1triton.jit
def _full_reduce_stage1_kernel(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    OP_KIND: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    if OP_KIND == 0:
        vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
        res = tl.sum(vals, axis=0)
    elif OP_KIND == 1:
        vals = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        res = tl.max(vals, axis=0)
    else:
        vals = tl.load(x_ptr + offs, mask=mask, other=float("inf"))
        res = tl.min(vals, axis=0)
    tl.store(partial_ptr + pid.to(tl.int64), res)


@m1triton.jit
def _fill_const_kernel(
    out_ptr,
    fill_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    tlx.vec_width(4)
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    tl.store(out_ptr + offs, fill_val, mask=mask)


@m1triton.jit
def _cast_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, x, mask=mask)


# ============================================================================
# Pointwise ATen Op Table & Dynamic Kernel Synthesis + Cache
# ============================================================================

_POINTWISE_TARGETS: frozenset[Any] = frozenset(
    {
        aten.add.Tensor,
        aten.add.Scalar,
        aten.sub.Tensor,
        aten.sub.Scalar,
        aten.rsub.Scalar,
        aten.rsub.Tensor,
        aten.mul.Tensor,
        aten.mul.Scalar,
        aten.div.Tensor,
        aten.div.Scalar,
        aten.div.Tensor_mode,
        aten.div.Scalar_mode,
        aten.pow.Tensor_Scalar,
        aten.pow.Tensor_Tensor,
        aten.pow.Scalar,
        aten.relu.default,
        aten.leaky_relu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.gelu.default,
        aten.silu.default,
        aten.mish.default,
        aten.hardswish.default,
        aten.hardsigmoid.default,
        aten.hardtanh.default,
        aten.clamp.default,
        aten.clamp.Tensor,
        aten.clamp_min.default,
        aten.clamp_min.Tensor,
        aten.clamp_max.default,
        aten.clamp_max.Tensor,
        aten.elu.default,
        aten.selu.default,
        aten.softplus.default,
        aten.exp.default,
        aten.log.default,
        aten.abs.default,
        aten.neg.default,
        aten.sqrt.default,
        aten.rsqrt.default,
        aten.reciprocal.default,
        aten.sin.default,
        aten.cos.default,
        aten.floor.default,
        aten.ceil.default,
        aten.maximum.default,
        aten.minimum.default,
        aten.where.self,
        aten.where.ScalarOther,
        aten.where.ScalarSelf,
        aten.gt.Tensor,
        aten.gt.Scalar,
        aten.ge.Tensor,
        aten.ge.Scalar,
        aten.lt.Tensor,
        aten.lt.Scalar,
        aten.le.Tensor,
        aten.le.Scalar,
        aten.eq.Tensor,
        aten.eq.Scalar,
        aten.ne.Tensor,
        aten.ne.Scalar,
        aten.logical_and.default,
        aten.logical_or.default,
        aten.logical_not.default,
        aten.bitwise_and.Tensor,
        aten.bitwise_or.Tensor,
        aten.bitwise_not.default,
        aten._to_copy.default,
        aten.clone.default,
    }
)

_METADATA_TARGETS: frozenset[Any] = frozenset(
    {
        operator.getitem,
        aten.view.default,
        aten._unsafe_view.default,
        aten.reshape.default,
        aten.unsqueeze.default,
        aten.squeeze.default,
        aten.squeeze.dim,
        aten.squeeze.dims,
        aten.expand.default,
        aten.permute.default,
        aten.transpose.int,
        aten.t.default,
        aten.alias.default,
        aten.detach.default,
        aten.slice.Tensor,
        aten.select.int,
        aten.narrow.default,
        aten.contiguous.default,
    }
)

_FACTORY_FILL_TARGETS: frozenset[Any] = frozenset(
    {
        aten.zeros.default,
        aten.ones.default,
        aten.full.default,
        aten.zeros_like.default,
        aten.ones_like.default,
        aten.full_like.default,
    }
)

_GEMM_TARGETS: frozenset[Any] = frozenset(
    {
        aten.mm.default,
        aten.addmm.default,
        aten.bmm.default,
        aten.linear.default,
        aten.matmul.default,
    }
)

_CONV_TARGETS: frozenset[Any] = frozenset({aten.convolution.default})

_POOL_TARGETS: frozenset[Any] = frozenset(
    {
        aten.avg_pool1d.default,
        aten.avg_pool2d.default,
        aten.avg_pool3d.default,
        aten.max_pool2d.default,
        aten.max_pool2d_with_indices.default,
        aten.max_pool1d_with_indices.default,
        aten.max_pool3d_with_indices.default,
    }
)

_SOFTMAX_TARGETS: frozenset[Any] = frozenset(
    {
        aten._softmax.default,
        aten._log_softmax.default,
    }
)

_NORM_TARGETS: frozenset[Any] = frozenset(
    {
        aten.native_layer_norm.default,
        aten.native_group_norm.default,
        aten.native_batch_norm.default,
        aten._native_batch_norm_legit.default,
        aten._native_batch_norm_legit.no_stats,
        aten._native_batch_norm_legit_no_training.default,
        aten._native_batch_norm_legit_functional.default,
    }
)

_REDUCE_TARGETS: frozenset[Any] = frozenset(
    {
        aten.sum.default,
        aten.sum.dim_IntList,
        aten.mean.default,
        aten.mean.dim,
        aten.amax.default,
        aten.amin.default,
    }
)


_POINTWISE_KERNEL_CACHE: dict[tuple[Any, ...], m1triton.JITFunction] = {}
_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


def get_cache_stats() -> dict[str, int]:
    """Return kernel synthesis cache statistics (`pointwise_kernels`, `hits`, `misses`)."""
    return {
        "pointwise_kernels": len(_POINTWISE_KERNEL_CACHE),
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
    }


def clear_kernel_cache() -> None:
    """Clear the synthesized fused pointwise kernel and transformed filter caches."""
    _POINTWISE_KERNEL_CACHE.clear()
    _WINO_FILTER_CACHE.clear()
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0


def _node_val(node: fx.Node) -> Any:
    val = node.meta.get("val")
    if val is not None:
        return val
    return node.meta.get("example_value")


def _node_tensor_meta(node: fx.Node) -> tuple[tuple[int, ...] | None, torch.dtype | None, str | None]:
    val = _node_val(node)
    if isinstance(val, torch.Tensor):
        shape: tuple[int, ...] | None = None
        try:
            shape = tuple(int(s) for s in val.shape)
        except Exception:
            shape = None
        return shape, val.dtype, val.device.type
    tm = node.meta.get("tensor_meta")
    if tm is not None and hasattr(tm, "shape") and hasattr(tm, "dtype"):
        try:
            shape = tuple(int(s) for s in tm.shape)
        except Exception:
            shape = None
        dev = getattr(tm, "device", None)
        dev_type = dev.type if dev is not None else None
        return shape, tm.dtype, dev_type
    return None, None, None


def _is_supported_pointwise_node(node: fx.Node) -> bool:
    if node.op != "call_function" or node.target not in _POINTWISE_TARGETS:
        return False
    if node.target in (aten.div.Tensor_mode, aten.div.Scalar_mode):
        mode = node.kwargs.get("rounding_mode")
        if mode is None and len(node.args) >= 3:
            mode = node.args[2]
        if mode not in (None, "floor"):
            return False
    shape, dtype, dev = _node_tensor_meta(node)
    if node.target == aten._to_copy.default:
        out_dt = node.kwargs.get("dtype")
        if out_dt is not None and out_dt not in _SUPPORTED_PW_DTYPES:
            return False
        req_dev = node.kwargs.get("device")
        req_dev_type = (
            req_dev.type
            if isinstance(req_dev, torch.device)
            else (str(req_dev).split(":")[0] if req_dev is not None else None)
        )
        if req_dev_type is not None and req_dev_type not in _SUPPORTED_DEVICES:
            return False
        in_node = node.args[0] if node.args else None
        if isinstance(in_node, fx.Node):
            _, _, in_dev = _node_tensor_meta(in_node)
            if req_dev_type is not None and (in_dev is None or req_dev_type != in_dev):
                return False
            if in_dev is not None and dev is not None and in_dev != dev:
                return False
        elif req_dev_type is not None:
            return False
    if dtype is not None and dtype not in _SUPPORTED_PW_DTYPES:
        return False
    if dev is not None and dev not in _SUPPORTED_DEVICES:
        return False
    if shape is None:
        return False
    for a in node.args:
        if isinstance(a, fx.Node):
            _, adt, adev = _node_tensor_meta(a)
            if adt is not None and adt not in _SUPPORTED_PW_DTYPES:
                return False
            if adev is not None and adev not in _SUPPORTED_DEVICES:
                return False
            if adev is not None and dev is not None and adev != dev:
                return False
    return True


def _compress_broadcast(
    in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    """Compress input broadcast layout relative to `out_shape`.

    Returns `(mode, stride_plan)`:
    - `("contig", ())` when `in_shape == out_shape`
    - `("scalar", ())` when `math.prod(in_shape) == 1`
    - `("bcast", ((out_stride_r, dim_size_r, in_stride_r), ...))` for general broadcast.
    """
    if in_shape == out_shape:
        return "contig", ()
    if not in_shape or math.prod(in_shape) == 1:
        return "scalar", ()
    rank = len(out_shape)
    pad = (1,) * (rank - len(in_shape)) + in_shape
    if pad == out_shape:
        return "contig", ()

    in_strides = [1] * rank
    running = 1
    for i in range(rank - 1, -1, -1):
        in_strides[i] = running if pad[i] > 1 else 0
        running *= pad[i]

    out_strides = [1] * rank
    running = 1
    for i in range(rank - 1, -1, -1):
        out_strides[i] = running
        running *= out_shape[i]

    # Merge adjacent active (non-broadcast) dimensions with matching contiguous strides.
    terms: list[tuple[int, int, int]] = []
    i = 0
    while i < rank:
        if pad[i] == 1 or out_shape[i] == 1:
            i += 1
            continue
        dim_prod = out_shape[i]
        j = i + 1
        while (
            j < rank
            and pad[j] == out_shape[j]
            and pad[j] > 1
            and in_strides[j - 1] == in_strides[j] * pad[j]
            and out_strides[j - 1] == out_strides[j] * out_shape[j]
        ):
            dim_prod *= out_shape[j]
            j += 1
        terms.append((out_strides[j - 1], dim_prod, in_strides[j - 1]))
        i = j

    if not terms:
        return "scalar", ()
    return "bcast", tuple(terms)


@dataclass(frozen=True)
class _PWExprNode:
    target: Any
    operands: tuple[Any, ...]
    attrs: tuple[tuple[str, Any], ...]
    out_dtype: torch.dtype


@dataclass
class _FusedPointwisePlan:
    input_names: tuple[str, ...]
    input_modes: tuple[str, ...]
    input_term_counts: tuple[int, ...]
    input_dtypes: tuple[torch.dtype, ...]
    output_names: tuple[str, ...]
    output_dtypes: tuple[torch.dtype, ...]
    out_shape: tuple[int, ...]
    ops: tuple[tuple[str, _PWExprNode], ...]
    kernel: m1triton.JITFunction
    use_vec4: bool
    launch_mode: str  # "flat", "row_bcast", "chan_bcast", "general_bcast"
    input_bcast_terms: tuple[tuple[tuple[int, int, int], ...], ...]
    inner_len: int = 0
    chan_dims: tuple[int, ...] = ()


def _emit_pw_op_lines(var_name: str, expr: _PWExprNode, ref_map: dict[Any, str]) -> list[str]:
    def arg_ref(item: Any) -> str:
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "ref":
            return ref_map[item[1]]
        return _scalar_literal(item)

    attrs = dict(expr.attrs)
    t = expr.target
    ops = expr.operands

    if t in (aten.add.Tensor, aten.add.Scalar):
        a, b = arg_ref(ops[0]), arg_ref(ops[1])
        alpha = attrs.get("alpha", 1)
        if alpha != 1:
            return [f"{var_name} = {a} + ({b} * {_scalar_literal(alpha)})"]
        return [f"{var_name} = {a} + {b}"]

    if t in (aten.sub.Tensor, aten.sub.Scalar):
        a, b = arg_ref(ops[0]), arg_ref(ops[1])
        alpha = attrs.get("alpha", 1)
        if alpha != 1:
            return [f"{var_name} = {a} - ({b} * {_scalar_literal(alpha)})"]
        return [f"{var_name} = {a} - {b}"]

    if t in (aten.rsub.Scalar, aten.rsub.Tensor):
        a, b = arg_ref(ops[0]), arg_ref(ops[1])
        alpha = attrs.get("alpha", 1)
        if alpha != 1:
            return [f"{var_name} = {b} - ({a} * {_scalar_literal(alpha)})"]
        return [f"{var_name} = {b} - {a}"]

    if t in (aten.mul.Tensor, aten.mul.Scalar):
        return [f"{var_name} = {arg_ref(ops[0])} * {arg_ref(ops[1])}"]

    if t in (aten.div.Tensor, aten.div.Scalar, aten.div.Tensor_mode, aten.div.Scalar_mode):
        a, b = arg_ref(ops[0]), arg_ref(ops[1])
        if attrs.get("rounding_mode") == "floor":
            return [f"{var_name} = {a} // {b}"]
        return [f"{var_name} = {a} / {b}"]

    if t in (aten.pow.Tensor_Scalar, aten.pow.Tensor_Tensor, aten.pow.Scalar):
        a = arg_ref(ops[0])
        raw_b = ops[1]
        if not (isinstance(raw_b, tuple) and len(raw_b) == 2 and raw_b[0] == "ref"):
            exp_v = float(raw_b)
            if exp_v == 0.0:
                return [f"{var_name} = ({a} * 0.0) + 1.0"]
            if exp_v == 1.0:
                return [f"{var_name} = {a}"]
            if exp_v == 2.0:
                return [f"{var_name} = {a} * {a}"]
            if exp_v == 3.0:
                return [f"{var_name} = ({a} * {a}) * {a}"]
            if exp_v == 0.5:
                return [f"{var_name} = tl.sqrt({a})"]
            if exp_v == -0.5:
                return [f"{var_name} = tl.rsqrt({a})"]
            if exp_v == -1.0:
                return [f"{var_name} = 1.0 / {a}"]
        return [f"{var_name} = {a} ** {arg_ref(raw_b)}"]

    if t == aten.relu.default:
        return [f"{var_name} = tl.maximum({arg_ref(ops[0])}, 0.0)"]

    if t == aten.leaky_relu.default:
        x = arg_ref(ops[0])
        slope = _float_literal(attrs.get("negative_slope", 0.01))
        return [f"{var_name} = tl.where({x} > 0.0, {x}, {x} * {slope})"]

    if t == aten.sigmoid.default:
        return [f"{var_name} = tl.sigmoid({arg_ref(ops[0])})"]

    if t == aten.tanh.default:
        return [f"{var_name} = tl.tanh({arg_ref(ops[0])})"]

    if t == aten.gelu.default:
        x = arg_ref(ops[0])
        if attrs.get("approximate", "none") == "tanh":
            return [
                f"{var_name}_c = 0.7978845608028654 * ({x} + 0.044715 * ({x} * {x} * {x}))",
                f"{var_name} = 0.5 * {x} * (1.0 + tl.tanh({var_name}_c))",
            ]
        return [f"{var_name} = 0.5 * {x} * (1.0 + tl.erf({x} * 0.7071067811865476))"]

    if t == aten.silu.default:
        x = arg_ref(ops[0])
        return [f"{var_name} = {x} * tl.sigmoid({x})"]

    if t == aten.mish.default:
        x = arg_ref(ops[0])
        return [
            f"{var_name}_sp = tl.where({x} > 20.0, {x}, tl.log(1.0 + tl.exp(tl.where({x} > 20.0, 0.0, {x}))))",
            f"{var_name} = {x} * tl.tanh({var_name}_sp)",
        ]

    if t == aten.hardsigmoid.default:
        x = arg_ref(ops[0])
        return [f"{var_name} = tl.where({x} <= -3.0, 0.0, tl.where({x} >= 3.0, 1.0, ({x} + 3.0) * 0.16666666666666666))"]

    if t == aten.hardswish.default:
        x = arg_ref(ops[0])
        return [
            f"{var_name} = tl.where({x} <= -3.0, 0.0, tl.where({x} >= 3.0, {x}, {x} * ({x} + 3.0) * 0.16666666666666666))"
        ]

    if t == aten.hardtanh.default:
        x = arg_ref(ops[0])
        lo = _float_literal(attrs.get("min_val", -1.0))
        hi = _float_literal(attrs.get("max_val", 1.0))
        return [f"{var_name} = tl.minimum(tl.maximum({x}, {lo}), {hi})"]

    if t in (aten.clamp.default, aten.clamp.Tensor):
        x = arg_ref(ops[0])
        lo = attrs.get("min_val")
        hi = attrs.get("max_val")
        cur = x
        lines: list[str] = []
        if lo is not None:
            lines.append(f"{var_name}_lo = tl.maximum({cur}, {arg_ref(lo)})")
            cur = f"{var_name}_lo"
        if hi is not None:
            lines.append(f"{var_name} = tl.minimum({cur}, {arg_ref(hi)})")
        else:
            lines.append(f"{var_name} = {cur}")
        return lines

    if t in (aten.clamp_min.default, aten.clamp_min.Tensor):
        return [f"{var_name} = tl.maximum({arg_ref(ops[0])}, {arg_ref(ops[1])})"]

    if t in (aten.clamp_max.default, aten.clamp_max.Tensor):
        return [f"{var_name} = tl.minimum({arg_ref(ops[0])}, {arg_ref(ops[1])})"]

    if t == aten.elu.default:
        x = arg_ref(ops[0])
        alpha = _float_literal(attrs.get("alpha", 1.0))
        scale = _float_literal(attrs.get("scale", 1.0))
        in_scale = _float_literal(attrs.get("input_scale", 1.0))
        return [
            f"{var_name}_neg = {scale} * {alpha} * (tl.exp(tl.where({x} > 0.0, 0.0, {x} * {in_scale})) - 1.0)",
            f"{var_name} = tl.where({x} > 0.0, {scale} * {x}, {var_name}_neg)",
        ]

    if t == aten.selu.default:
        x = arg_ref(ops[0])
        return [
            f"{var_name}_pos = 1.0507009873554804934193349852946 * {x}",
            f"{var_name}_neg = 1.7580993408473768599402175208123 * (tl.exp(tl.where({x} > 0.0, 0.0, {x})) - 1.0)",
            f"{var_name} = tl.where({x} > 0.0, {var_name}_pos, {var_name}_neg)",
        ]

    if t == aten.softplus.default:
        x = arg_ref(ops[0])
        beta = float(attrs.get("beta", 1.0))
        thresh = float(attrs.get("threshold", 20.0))
        if beta == 1.0:
            return [
                f"{var_name} = tl.where({x} > {_float_literal(thresh)}, {x}, tl.log(1.0 + tl.exp(tl.where({x} > {_float_literal(thresh)}, 0.0, {x}))))"
            ]
        b_lit = _float_literal(beta)
        inv_b = _float_literal(1.0 / beta)
        t_lit = _float_literal(thresh)
        return [
            f"{var_name}_bx = {x} * {b_lit}",
            f"{var_name} = tl.where({var_name}_bx > {t_lit}, {x}, tl.log(1.0 + tl.exp(tl.where({var_name}_bx > {t_lit}, 0.0, {var_name}_bx))) * {inv_b})",
        ]

    unary_tl = {
        aten.exp.default: "tl.exp",
        aten.log.default: "tl.log",
        aten.abs.default: "tl.abs",
        aten.sqrt.default: "tl.sqrt",
        aten.rsqrt.default: "tl.rsqrt",
        aten.sin.default: "tl.sin",
        aten.cos.default: "tl.cos",
        aten.floor.default: "tl.floor",
        aten.ceil.default: "tl.ceil",
    }
    if t in unary_tl:
        return [f"{var_name} = {unary_tl[t]}({arg_ref(ops[0])})"]

    if t == aten.neg.default:
        return [f"{var_name} = -{arg_ref(ops[0])}"]

    if t == aten.reciprocal.default:
        return [f"{var_name} = 1.0 / {arg_ref(ops[0])}"]

    if t == aten.maximum.default:
        return [f"{var_name} = tl.maximum({arg_ref(ops[0])}, {arg_ref(ops[1])})"]

    if t == aten.minimum.default:
        return [f"{var_name} = tl.minimum({arg_ref(ops[0])}, {arg_ref(ops[1])})"]

    if t in (aten.where.self, aten.where.ScalarOther, aten.where.ScalarSelf):
        return [f"{var_name} = tl.where({arg_ref(ops[0])}, {arg_ref(ops[1])}, {arg_ref(ops[2])})"]

    if t == aten.logical_and.default:
        return [f"{var_name} = (({arg_ref(ops[0])} != 0) & ({arg_ref(ops[1])} != 0))"]

    if t == aten.logical_or.default:
        return [f"{var_name} = (({arg_ref(ops[0])} != 0) | ({arg_ref(ops[1])} != 0))"]

    if t == aten.logical_not.default:
        return [f"{var_name} = ({arg_ref(ops[0])} == 0)"]

    cmp_ops = {
        aten.gt.Tensor: ">",
        aten.gt.Scalar: ">",
        aten.ge.Tensor: ">=",
        aten.ge.Scalar: ">=",
        aten.lt.Tensor: "<",
        aten.lt.Scalar: "<",
        aten.le.Tensor: "<=",
        aten.le.Scalar: "<=",
        aten.eq.Tensor: "==",
        aten.eq.Scalar: "==",
        aten.ne.Tensor: "!=",
        aten.ne.Scalar: "!=",
        aten.bitwise_and.Tensor: "&",
        aten.bitwise_or.Tensor: "|",
    }
    if t in cmp_ops:
        return [f"{var_name} = ({arg_ref(ops[0])} {cmp_ops[t]} {arg_ref(ops[1])})"]

    if t == aten.bitwise_not.default:
        return [f"{var_name} = ~({arg_ref(ops[0])})"]

    if t == aten._to_copy.default:
        tl_dt = _torch_dtype_to_tl_name(expr.out_dtype)
        return [f"{var_name} = ({arg_ref(ops[0])}).to({tl_dt})"]

    if t == aten.clone.default:
        return [f"{var_name} = {arg_ref(ops[0])}"]

    raise NotImplementedError(f"Unsupported pointwise op in synthesis: {t!r}")


def _extract_pw_expr(node: fx.Node, out_dtype: torch.dtype) -> _PWExprNode:
    t = node.target
    attrs: dict[str, Any] = {}

    def norm_arg(a: Any) -> Any:
        if isinstance(a, fx.Node):
            return ("ref", a.name)
        return a

    if t in (aten.add.Tensor, aten.add.Scalar, aten.sub.Tensor, aten.sub.Scalar, aten.rsub.Scalar, aten.rsub.Tensor):
        alpha = node.kwargs.get("alpha", 1)
        if alpha != 1:
            attrs["alpha"] = alpha
        operands = (norm_arg(node.args[0]), norm_arg(node.args[1]))
    elif t in (aten.div.Tensor_mode, aten.div.Scalar_mode):
        mode = node.kwargs.get("rounding_mode", node.args[2] if len(node.args) >= 3 else None)
        if mode is not None:
            attrs["rounding_mode"] = mode
        operands = (norm_arg(node.args[0]), norm_arg(node.args[1]))
    elif t == aten.leaky_relu.default:
        slope = node.args[1] if len(node.args) >= 2 else node.kwargs.get("negative_slope", 0.01)
        attrs["negative_slope"] = float(slope)
        operands = (norm_arg(node.args[0]),)
    elif t == aten.gelu.default:
        approx = node.kwargs.get("approximate", node.args[1] if len(node.args) >= 2 else "none")
        attrs["approximate"] = str(approx)
        operands = (norm_arg(node.args[0]),)
    elif t == aten.hardtanh.default:
        lo = node.args[1] if len(node.args) >= 2 else node.kwargs.get("min_val", -1.0)
        hi = node.args[2] if len(node.args) >= 3 else node.kwargs.get("max_val", 1.0)
        attrs["min_val"] = float(lo)
        attrs["max_val"] = float(hi)
        operands = (norm_arg(node.args[0]),)
    elif t in (aten.clamp.default, aten.clamp.Tensor):
        lo = node.args[1] if len(node.args) >= 2 else node.kwargs.get("min")
        hi = node.args[2] if len(node.args) >= 3 else node.kwargs.get("max")
        if lo is not None:
            attrs["min_val"] = norm_arg(lo)
        if hi is not None:
            attrs["max_val"] = norm_arg(hi)
        operands = (norm_arg(node.args[0]),)
    elif t == aten.elu.default:
        attrs["alpha"] = float(node.args[1] if len(node.args) >= 2 else node.kwargs.get("alpha", 1.0))
        attrs["scale"] = float(node.args[2] if len(node.args) >= 3 else node.kwargs.get("scale", 1.0))
        attrs["input_scale"] = float(node.args[3] if len(node.args) >= 4 else node.kwargs.get("input_scale", 1.0))
        operands = (norm_arg(node.args[0]),)
    elif t == aten.softplus.default:
        attrs["beta"] = float(node.args[1] if len(node.args) >= 2 else node.kwargs.get("beta", 1.0))
        attrs["threshold"] = float(node.args[2] if len(node.args) >= 3 else node.kwargs.get("threshold", 20.0))
        operands = (norm_arg(node.args[0]),)
    elif t in (aten.where.self, aten.where.ScalarOther, aten.where.ScalarSelf):
        operands = (norm_arg(node.args[0]), norm_arg(node.args[1]), norm_arg(node.args[2]))
    elif t in (aten._to_copy.default, aten.clone.default):
        operands = (norm_arg(node.args[0]),)
    else:
        operands = tuple(norm_arg(a) for a in node.args)

    return _PWExprNode(
        target=t,
        operands=operands,
        attrs=tuple(sorted(attrs.items())),
        out_dtype=out_dtype,
    )


def _synthesize_pointwise_plan(
    nodes: Sequence[fx.Node],
    external_outputs: Sequence[fx.Node],
    out_shape: tuple[int, ...],
) -> _FusedPointwisePlan:
    node_set = {n.name for n in nodes}
    input_nodes: list[fx.Node] = []
    seen_inputs: set[str] = set()

    for n in nodes:
        for a in n.args:
            if isinstance(a, fx.Node) and a.name not in node_set and a.name not in seen_inputs:
                seen_inputs.add(a.name)
                input_nodes.append(a)
        for v in n.kwargs.values():
            if isinstance(v, fx.Node) and v.name not in node_set and v.name not in seen_inputs:
                seen_inputs.add(v.name)
                input_nodes.append(v)

    input_names = tuple(n.name for n in input_nodes)
    input_dtypes_list: list[torch.dtype] = []
    input_modes_list: list[str] = []
    input_term_counts_list: list[int] = []
    input_bcast_terms_list: list[tuple[tuple[int, int, int], ...]] = []

    for inp in input_nodes:
        ishape, idt, _ = _node_tensor_meta(inp)
        if idt is None:
            idt = torch.float32
        input_dtypes_list.append(idt)
        mode, terms = _compress_broadcast(ishape or (), out_shape)
        input_modes_list.append(mode)
        input_term_counts_list.append(len(terms))
        input_bcast_terms_list.append(terms)

    output_names = tuple(n.name for n in external_outputs)
    output_dtypes = tuple((_node_tensor_meta(n)[1] or torch.float32) for n in external_outputs)

    # Canonicalize variable names into slot indices for structural cache key.
    slot_of: dict[str, str] = {}
    for idx, name in enumerate(input_names):
        slot_of[name] = f"in_{idx}"

    exprs: list[tuple[str, _PWExprNode]] = []
    canonical_ops: list[Any] = []
    for step_idx, n in enumerate(nodes):
        odt = _node_tensor_meta(n)[1] or torch.float32
        expr = _extract_pw_expr(n, odt)
        exprs.append((n.name, expr))
        c_slot = f"v_{step_idx}"
        canon_operands = tuple(
            ("ref", slot_of[o[1]]) if (isinstance(o, tuple) and len(o) == 2 and o[0] == "ref") else o
            for o in expr.operands
        )
        canon_attrs = tuple(
            (
                k,
                ("ref", slot_of[v[1]]) if (isinstance(v, tuple) and len(v) == 2 and v[0] == "ref") else v,
            )
            for k, v in expr.attrs
        )
        canonical_ops.append((str(expr.target), canon_operands, canon_attrs, str(odt)))
        slot_of[n.name] = c_slot

    canon_outs = tuple(slot_of[name] for name in output_names)
    vec4_dtypes_ok = all(
        dt in (torch.float32, torch.float16, torch.int32) for dt in (*input_dtypes_list, *output_dtypes)
    )

    # Detect specialized 2D-grid float4 broadcast layouts ("row_bcast" and "chan_bcast").
    launch_mode = "flat"
    inner_len = 0
    chan_dims_list: list[int] = []
    if any(m == "bcast" for m in input_modes_list):
        bcast_indices = [idx for idx, m in enumerate(input_modes_list) if m == "bcast"]
        # Row broadcast: every broadcast input is a contiguous trailing slice (out_stride=1, in_stride=1, same dim).
        row_lens = {
            input_bcast_terms_list[idx][0][1]
            for idx in bcast_indices
            if len(input_bcast_terms_list[idx]) == 1
            and input_bcast_terms_list[idx][0][0] == 1
            and input_bcast_terms_list[idx][0][2] == 1
        }
        all_row_bcast = len(row_lens) == 1 and all(
            len(input_bcast_terms_list[idx]) == 1
            and input_bcast_terms_list[idx][0][0] == 1
            and input_bcast_terms_list[idx][0][2] == 1
            for idx in bcast_indices
        )
        # Channel broadcast: every broadcast input is a single outer-channel slice with identical inner stride > 1.
        chan_inner_lens = {
            input_bcast_terms_list[idx][0][0]
            for idx in bcast_indices
            if len(input_bcast_terms_list[idx]) == 1
            and input_bcast_terms_list[idx][0][0] > 1
            and input_bcast_terms_list[idx][0][2] == 1
        }
        all_chan_bcast = len(chan_inner_lens) == 1 and all(
            len(input_bcast_terms_list[idx]) == 1
            and input_bcast_terms_list[idx][0][0] > 1
            and input_bcast_terms_list[idx][0][2] == 1
            for idx in bcast_indices
        )

        if all_row_bcast and out_shape:
            rl = next(iter(row_lens))
            total_el = int(math.prod(out_shape))
            if rl > 0 and (total_el % rl) == 0:
                launch_mode = "row_bcast"
                inner_len = rl
        if launch_mode == "flat" and all_chan_bcast and out_shape:
            il = next(iter(chan_inner_lens))
            total_el = int(math.prod(out_shape))
            if il > 0 and (total_el % il) == 0:
                launch_mode = "chan_bcast"
                inner_len = il
                for idx in bcast_indices:
                    chan_dims_list.append(int(input_bcast_terms_list[idx][0][1]))
        if launch_mode == "flat":
            launch_mode = "general_bcast"

    use_vec4 = vec4_dtypes_ok and launch_mode in ("flat", "row_bcast", "chan_bcast")

    sig_key = (
        tuple(canonical_ops),
        canon_outs,
        tuple(str(dt) for dt in input_dtypes_list),
        tuple(input_modes_list),
        tuple(input_term_counts_list),
        tuple(str(dt) for dt in output_dtypes),
        launch_mode,
        use_vec4,
    )

    cached_kernel = _POINTWISE_KERNEL_CACHE.get(sig_key)
    if cached_kernel is not None:
        _CACHE_STATS["hits"] += 1
        kernel = cached_kernel
    else:
        _CACHE_STATS["misses"] += 1
        digest = hashlib.sha256(repr(sig_key).encode("utf-8")).hexdigest()[:16]
        fn_name = f"m1_fused_pw_{digest}"

        params: list[str] = []
        for idx in range(len(input_names)):
            params.append(f"in{idx}_ptr")
        for idx in range(len(output_names)):
            params.append(f"out{idx}_ptr")

        body_lines: list[str] = []
        if use_vec4:
            body_lines.append("    tlx.vec_width(4)")

        if launch_mode == "row_bcast":
            params.extend(["row_len", "BLOCK_SIZE: tl.constexpr", "USE_INT64: tl.constexpr = 0"])
            body_lines.append("    pid_col = tl.program_id(0)")
            body_lines.append("    row = tl.program_id(1)")
            body_lines.append("    if USE_INT64:")
            body_lines.append("        offs = pid_col.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)")
            body_lines.append("    else:")
            body_lines.append("        offs = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            body_lines.append("    mask = offs < row_len")
            body_lines.append("    row_off = row.to(tl.int64) * row_len")
        elif launch_mode == "chan_bcast":
            for idx, mode in enumerate(input_modes_list):
                if mode == "bcast":
                    params.append(f"c_dim_{idx}")
            params.extend(["inner_len", "BLOCK_SIZE: tl.constexpr", "USE_INT64: tl.constexpr = 0"])
            body_lines.append("    pid_hw = tl.program_id(0)")
            body_lines.append("    nc = tl.program_id(1)")
            body_lines.append("    if USE_INT64:")
            body_lines.append("        offs = pid_hw.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)")
            body_lines.append("    else:")
            body_lines.append("        offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            body_lines.append("    mask = offs < inner_len")
            body_lines.append("    base_off = nc.to(tl.int64) * inner_len")
        else:
            for idx, (mode, tc) in enumerate(zip(input_modes_list, input_term_counts_list)):
                if mode == "bcast":
                    for t_i in range(tc):
                        params.append(f"b{idx}_div_{t_i}")
                        params.append(f"b{idx}_mod_{t_i}")
                        params.append(f"b{idx}_st_{t_i}")
            params.extend(["n_elements", "BLOCK_SIZE: tl.constexpr", "USE_INT64: tl.constexpr = 0"])
            body_lines.append("    pid = tl.program_id(0)")
            body_lines.append("    if USE_INT64:")
            body_lines.append("        offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)")
            body_lines.append("    else:")
            body_lines.append("        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            body_lines.append("    mask = offs < n_elements")

        ref_map: dict[str, str] = {}
        for idx, (orig_name, mode, tc, idt) in enumerate(
            zip(input_names, input_modes_list, input_term_counts_list, input_dtypes_list)
        ):
            vname = f"in_{idx}"
            ref_map[orig_name] = vname
            other_lit = "0.0" if idt in _SUPPORTED_FLOAT_DTYPES else ("False" if idt == torch.bool else "0")
            if launch_mode == "row_bcast":
                if mode == "contig":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + (row_off + offs), mask=mask, other={other_lit})")
                elif mode == "scalar":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr)")
                else:
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + offs, mask=mask, other={other_lit})")
            elif launch_mode == "chan_bcast":
                if mode == "contig":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + (base_off + offs), mask=mask, other={other_lit})")
                elif mode == "scalar":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr)")
                else:
                    body_lines.append(f"    {vname}_c = nc % c_dim_{idx}")
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + {vname}_c)")
            else:
                if mode == "contig":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + offs, mask=mask, other={other_lit})")
                elif mode == "scalar":
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr)")
                else:
                    idx_terms: list[str] = []
                    for t_i in range(tc):
                        idx_terms.append(f"((offs // b{idx}_div_{t_i}) % b{idx}_mod_{t_i}) * b{idx}_st_{t_i}")
                    idx_expr = " + ".join(idx_terms) if idx_terms else "0"
                    body_lines.append(f"    {vname}_idx = {idx_expr}")
                    body_lines.append(f"    {vname} = tl.load(in{idx}_ptr + {vname}_idx, mask=mask, other={other_lit})")

        for step_idx, (orig_name, expr) in enumerate(exprs):
            vname = f"v_{step_idx}"
            op_lines = _emit_pw_op_lines(vname, expr, ref_map)
            for ln in op_lines:
                body_lines.append(f"    {ln}")
            ref_map[orig_name] = vname

        for idx, out_name in enumerate(output_names):
            v_ref = ref_map[out_name]
            if launch_mode == "row_bcast":
                body_lines.append(f"    tl.store(out{idx}_ptr + (row_off + offs), {v_ref}, mask=mask)")
            elif launch_mode == "chan_bcast":
                body_lines.append(f"    tl.store(out{idx}_ptr + (base_off + offs), {v_ref}, mask=mask)")
            else:
                body_lines.append(f"    tl.store(out{idx}_ptr + offs, {v_ref}, mask=mask)")

        src = f"@m1triton.jit\ndef {fn_name}({', '.join(params)}):\n" + "\n".join(body_lines) + "\n"
        kernel = _compile_generated_jit(fn_name, src)
        _POINTWISE_KERNEL_CACHE[sig_key] = kernel

    return _FusedPointwisePlan(
        input_names=input_names,
        input_modes=tuple(input_modes_list),
        input_term_counts=tuple(input_term_counts_list),
        input_dtypes=tuple(input_dtypes_list),
        output_names=output_names,
        output_dtypes=output_dtypes,
        out_shape=out_shape,
        ops=tuple(exprs),
        kernel=kernel,
        use_vec4=use_vec4,
        launch_mode=launch_mode,
        input_bcast_terms=tuple(input_bcast_terms_list),
        inner_len=inner_len,
        chan_dims=tuple(chan_dims_list),
    )


# ============================================================================
# Runtime Batch Executor & Op Dispatchers
# ============================================================================


class _BatchRunner:
    """Manages a single Metal `command_batch` across sequential M1Triton ops with clean MPS sync."""

    def __init__(self) -> None:
        self._ctx: Any = None
        self._mps_dirty: bool = True
        self._pending_cache_commits: list[Callable[[], None]] = []
        self._batch_wino: dict[tuple[Any, ...], torch.Tensor] = {}

    def ensure_contig(self, t: torch.Tensor) -> torch.Tensor:
        if t.is_contiguous():
            return t
        if t.device.type == "mps":
            self.flush()
            out = t.contiguous()
            self._mps_dirty = True
            return out
        return t.contiguous()

    def begin_m1(self, has_mps: bool) -> None:
        if self._ctx is None:
            sync_mps = bool(has_mps and self._mps_dirty)
            ctx = rt.command_batch(wait=True, sync_mps=sync_mps)
            ctx.__enter__()
            self._ctx = ctx
            self._mps_dirty = False

    def defer_cache_commit(self, fn: Callable[[], None]) -> None:
        if self._ctx is None:
            fn()
        else:
            self._pending_cache_commits.append(fn)

    def flush(self) -> None:
        if self._ctx is not None:
            ctx = self._ctx
            self._ctx = None
            ctx.__exit__(None, None, None)
        self._batch_wino.clear()
        if self._pending_cache_commits:
            commits = self._pending_cache_commits
            self._pending_cache_commits = []
            for commit in commits:
                commit()

    def abort(self) -> None:
        self._pending_cache_commits.clear()
        self._batch_wino.clear()
        if self._ctx is not None:
            ctx = self._ctx
            self._ctx = None
            ctx.__exit__(RuntimeError, RuntimeError("batch aborted"), None)

    def run_eager(self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.flush()
        res = fn(*args, **kwargs)
        self._mps_dirty = True
        return res


def _run_fused_pointwise(plan: _FusedPointwisePlan, env: dict[str, Any], runner: _BatchRunner) -> None:
    raw_inputs = [env[name] for name in plan.input_names]
    inputs: list[torch.Tensor] = []
    for t, mode in zip(raw_inputs, plan.input_modes):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor input for fused pointwise op, got {type(t)}")
        inputs.append(runner.ensure_contig(t))

    device = inputs[0].device if inputs else torch.device("mps")
    out_shape = plan.out_shape
    if not out_shape and inputs:
        # Compute broadcasted shape dynamically when shape metadata was symbolic.
        shapes = [tuple(t.shape) for t in inputs]
        out_shape = tuple(torch.broadcast_shapes(*shapes))

    n_elements = int(math.prod(out_shape)) if out_shape else 1
    outputs = [torch.empty(out_shape, dtype=dt, device=device) for dt in plan.output_dtypes]

    if n_elements == 0:
        for name, out_t in zip(plan.output_names, outputs):
            env[name] = out_t
        return

    use_int64 = 1 if n_elements > _INT32_FAST_LIMIT else 0
    runner.begin_m1(device.type == "mps")
    if plan.launch_mode == "row_bcast":
        row_len = plan.inner_len
        num_rows = n_elements // max(row_len, 1)
        block_size = 1024 if row_len >= 1024 else (512 if row_len >= 512 else 256)
        grid = (_cdiv(row_len, block_size), num_rows)
        plan.kernel[grid](
            *inputs,
            *outputs,
            row_len,
            BLOCK_SIZE=block_size,
            USE_INT64=use_int64,
            wait=False,
            sync_mps=False,
        )
    elif plan.launch_mode == "chan_bcast":
        inner_len = plan.inner_len
        num_outer = n_elements // max(inner_len, 1)
        block_size = 1024 if inner_len >= 1024 else (512 if inner_len >= 512 else 256)
        grid = (_cdiv(inner_len, block_size), num_outer)
        plan.kernel[grid](
            *inputs,
            *outputs,
            *plan.chan_dims,
            inner_len,
            BLOCK_SIZE=block_size,
            USE_INT64=use_int64,
            wait=False,
            sync_mps=False,
        )
    else:
        bcast_scalars: list[int] = []
        for t, mode, terms in zip(inputs, plan.input_modes, plan.input_bcast_terms):
            if mode == "bcast":
                dyn_mode, dyn_terms = _compress_broadcast(tuple(int(s) for s in t.shape), out_shape)
                use_terms = dyn_terms if (dyn_mode == "bcast" and len(dyn_terms) == len(terms)) else terms
                for out_st, dim_sz, in_st in use_terms:
                    bcast_scalars.extend((int(out_st), int(dim_sz), int(in_st)))

        if plan.use_vec4 and n_elements >= 262_144 and n_elements <= _INT32_FAST_LIMIT:
            block_size = 4096
        elif n_elements >= 1024:
            block_size = 1024
        else:
            block_size = 256

        grid = (_cdiv(n_elements, block_size),)
        plan.kernel[grid](
            *inputs,
            *outputs,
            *bcast_scalars,
            n_elements,
            BLOCK_SIZE=block_size,
            USE_INT64=use_int64,
            wait=False,
            sync_mps=False,
        )
    for name, out_t in zip(plan.output_names, outputs):
        env[name] = out_t


def _select_sgmm_schedule(
    m: int,
    n: int,
    k: int,
    trans_a: bool = False,
    trans_b: bool = False,
) -> tuple[int, int, int, int, int, int]:
    if m <= 32 and n <= 32:
        return 32, 32, 16, 4, 1, 0
    if k <= 64:
        return 64, 64, 16, 4, 1, 1
    if (trans_a or trans_b) and m >= 64 and n >= 128:
        return 64, 128, 16, 8, 8, 0
    return 64, 64, 16, 4, 8, 0


def _launch_sgmm_2d(
    a_in: torch.Tensor,
    b_in: torch.Tensor,
    *,
    trans_a: bool,
    trans_b: bool,
    runner: _BatchRunner,
) -> torch.Tensor:
    a_c = runner.ensure_contig(a_in)
    b_c = runner.ensure_contig(b_in)
    if trans_a:
        k, m = int(a_c.shape[0]), int(a_c.shape[1])
        stride_am, stride_ak = int(a_c.stride(1)), int(a_c.stride(0))
    else:
        m, k = int(a_c.shape[0]), int(a_c.shape[1])
        stride_am, stride_ak = int(a_c.stride(0)), int(a_c.stride(1))
    if trans_b:
        n, k_b = int(b_c.shape[0]), int(b_c.shape[1])
        stride_bk, stride_bn = int(b_c.stride(1)), int(b_c.stride(0))
    else:
        k_b, n = int(b_c.shape[0]), int(b_c.shape[1])
        stride_bk, stride_bn = int(b_c.stride(0)), int(b_c.stride(1))
    if k != k_b:
        raise ValueError(f"Incompatible GEMM K dimensions: {k} vs {k_b}")

    out = torch.empty((m, n), dtype=a_c.dtype, device=a_c.device)
    if m == 0 or n == 0 or k == 0:
        return out

    bm, bn, bk, sg, sw, n_maj = _select_sgmm_schedule(m, n, k, trans_a, trans_b)
    aligned = 1 if (m % bm == 0 and n % bn == 0 and k % bk == 0) else 0
    grid = (_cdiv(n, bn), _cdiv(m, bm), 1) if n_maj else (_cdiv(m, bm), _cdiv(n, bn), 1)
    runner.begin_m1(a_c.device.type == "mps")
    _sgmm_gemm_kernel[grid](
        a_c,
        b_c,
        out,
        m,
        n,
        k,
        0,
        stride_am,
        stride_ak,
        0,
        stride_bk,
        stride_bn,
        0,
        int(out.stride(0)),
        int(out.stride(1)),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=sg,
        SWIZZLE_W=sw,
        TRANS_A=1 if trans_a else 0,
        TRANS_B=1 if trans_b else 0,
        N_MAJOR=n_maj,
        IS_BATCHED=0,
        ALIGNED_MNK=aligned,
        fast_math=True,
        wait=False,
        sync_mps=False,
    )
    return out


def _launch_sgmm_bmm(a_in: torch.Tensor, b_in: torch.Tensor, runner: _BatchRunner) -> torch.Tensor:
    a_c = runner.ensure_contig(a_in)
    b_c = runner.ensure_contig(b_in)
    batch, m, k = (int(v) for v in a_c.shape)
    b_batch, k_b, n = (int(v) for v in b_c.shape)
    if batch != b_batch or k != k_b:
        raise ValueError(f"Incompatible BMM shapes: {tuple(a_c.shape)} vs {tuple(b_c.shape)}")

    out = torch.empty((batch, m, n), dtype=a_c.dtype, device=a_c.device)
    if out.numel() == 0:
        return out

    bm, bn, bk, sg, sw = (32, 32, 16, 4, 1) if (m <= 32 and n <= 32) else (64, 64, 16, 4, 8)
    aligned = 1 if (m % bm == 0 and n % bn == 0 and k % bk == 0) else 0
    grid = (_cdiv(m, bm), _cdiv(n, bn), batch)
    runner.begin_m1(a_c.device.type == "mps")
    _sgmm_gemm_kernel[grid](
        a_c,
        b_c,
        out,
        m,
        n,
        k,
        int(a_c.stride(0)),
        int(a_c.stride(1)),
        int(a_c.stride(2)),
        int(b_c.stride(0)),
        int(b_c.stride(1)),
        int(b_c.stride(2)),
        int(out.stride(0)),
        int(out.stride(1)),
        int(out.stride(2)),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=sg,
        SWIZZLE_W=sw,
        TRANS_A=0,
        TRANS_B=0,
        N_MAJOR=0,
        IS_BATCHED=0 if batch == 1 else 1,
        ALIGNED_MNK=aligned,
        fast_math=True,
        wait=False,
        sync_mps=False,
    )
    return out


def _conv_output_size(size: int, k: int, s: int, p: int, d: int) -> int:
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def _select_conv_tile(
    c_in_per_group: int,
    c_out_per_group: int,
    k_spatial: int,
    pointwise: bool,
    ndim: int = 2,
    stride_w: int = 1,
) -> tuple[int, int, int, int]:
    k_total = c_in_per_group * k_spatial
    if pointwise:
        bm = 16 if c_out_per_group <= 16 else (32 if c_out_per_group <= 32 else 64)
        bk = 32 if (k_total > 0 and k_total % 32 == 0) else 16
        return bm, 64, bk, 4
    if (
        ndim == 1
        and stride_w > 1
        and c_out_per_group >= 128
        and (c_out_per_group % 128 == 0)
        and k_total >= 192
        and (k_total % 16 == 0)
    ):
        return 128, 64, 16, 8
    if ndim == 2 and c_out_per_group >= 128 and k_total >= 1024 and (k_total % 16 == 0):
        return 64, 64, 16, 4
    if c_out_per_group >= 128 and (c_out_per_group % 128 == 0) and k_total >= 192 and (k_total % 32 == 0):
        return 128, 64, 32, 8
    if c_out_per_group <= 16:
        return 16, 64, 16, 4
    if c_out_per_group <= 32 or (c_out_per_group % 32 == 0 and c_out_per_group % 64 != 0):
        return 32, 64, 16, 4
    return 64, 64, 16, 4


def _dw2d_halo(tile: int, k: int, s: int, d: int) -> int:
    return (tile - 1) * s + d * (k - 1) + 1


def _select_dw2d_tile(
    h_out: int, w_out: int, k: tuple[int, int], s: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int, int, int]:
    if s == (1, 1) and d == (1, 1) and k[0] <= 15 and k[1] <= 15:
        if h_out >= 64:
            return 128, 64, 4, 16
        if h_out >= 32:
            return 128, 32, 4, 8
        return 128, 16, 4, 4
    block_w = 64 if w_out > 32 else 32
    block_h = 32 if h_out > 16 else (16 if h_out > 8 else 8)
    while block_h > 8 and _dw2d_halo(block_h, k[0], s[0], d[0]) * _dw2d_halo(block_w, k[1], s[1], d[1]) * 4 > 16384:
        block_h //= 2
    while block_w > 32 and _dw2d_halo(block_h, k[0], s[0], d[0]) * _dw2d_halo(block_w, k[1], s[1], d[1]) * 4 > 16384:
        block_w //= 2
    return block_w, block_h, 1, 0


_WINO_FILTER_CACHE: dict[tuple[Any, ...], tuple[TensorWeakRef, int, torch.Tensor]] = {}


def _get_or_transform_wino2x3_filter(
    weight_orig: torch.Tensor,
    w_c: torch.Tensor,
    c_in: int,
    c_out: int,
    c_in_pad: int,
    c_out_pad: int,
    runner: _BatchRunner,
) -> torch.Tensor:
    key = (id(weight_orig), c_in, c_out, c_in_pad, c_out_pad, str(w_c.device))
    ver = int(getattr(weight_orig, "_version", -1))
    cached = _WINO_FILTER_CACHE.get(key)
    if cached is not None:
        w_ref, cached_ver, u_cached = cached
        if w_ref() is weight_orig and cached_ver == ver:
            del _WINO_FILTER_CACHE[key]
            _WINO_FILTER_CACHE[key] = cached
            return u_cached
    batch_key = (key, ver)
    pending_u = runner._batch_wino.get(batch_key)
    if pending_u is not None:
        return pending_u
    u = torch.empty((16, c_in_pad // 8, c_out_pad // 8, 8, 8), dtype=torch.float32, device=w_c.device)
    _wino2x3_filter_kernel[(_cdiv(c_out_pad * c_in_pad, 256),)](
        w_c,
        u,
        C_in=c_in,
        C_out=c_out,
        C_IN_PAD=c_in_pad,
        C_OUT_PAD=c_out_pad,
        BLOCK_SIZE=256,
        wait=False,
        sync_mps=False,
    )
    runner._batch_wino[batch_key] = u

    def _commit() -> None:
        if key in _WINO_FILTER_CACHE:
            del _WINO_FILTER_CACHE[key]
        elif len(_WINO_FILTER_CACHE) >= 64:
            del _WINO_FILTER_CACHE[next(iter(_WINO_FILTER_CACHE))]
        _WINO_FILTER_CACHE[key] = (TensorWeakRef(weight_orig), ver, u)

    runner.defer_cache_commit(_commit)
    return u


def _launch_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    runner: _BatchRunner,
) -> torch.Tensor:
    if transposed:
        raise NotImplementedError("Transposed convolution lowering is not enabled on this path")

    ndim = x.ndim - 2
    if ndim not in (1, 2, 3):
        raise NotImplementedError(f"Unsupported convolution rank: {x.ndim}")

    x_c = runner.ensure_contig(x)
    w_c = runner.ensure_contig(weight.detach())
    b_c = runner.ensure_contig(bias.detach()) if bias is not None else None
    b_arg = b_c if b_c is not None else x_c
    has_bias = 1 if b_c is not None else 0

    n, c_in = int(x_c.shape[0]), int(x_c.shape[1])
    c_out = int(w_c.shape[0])
    k = tuple(int(v) for v in w_c.shape[2:])
    s = tuple(int(v) for v in stride)
    p = tuple(int(v) for v in padding)
    d = tuple(int(v) for v in dilation)
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [_conv_output_size(spatial_in[i], k[i], s[i], p[i], d[i]) for i in range(ndim)]

    out = torch.empty((n, c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    runner.begin_m1(x_c.device.type == "mps")

    # Depthwise 2D fast path.
    if ndim == 2 and groups == c_in and groups > 1 and int(w_c.shape[1]) == 1 and (c_out % c_in == 0):
        bw, bh, vw, rpt = _select_dw2d_tile(spatial_out[0], spatial_out[1], (k[0], k[1]), (s[0], s[1]), (d[0], d[1]))
        grid = (_cdiv(spatial_out[1], bw), _cdiv(spatial_out[0], bh), n * c_out)
        _conv_dw2d_kernel[grid](
            x_c, w_c, out, b_arg, n, c_in, c_out, spatial_in[0], spatial_in[1], spatial_out[0], spatial_out[1],
            K_H=k[0], K_W=k[1], S_H=s[0], S_W=s[1], P_H=p[0], P_W=p[1], D_H=d[0], D_W=d[1],
            BLOCK_W=bw, BLOCK_H=bh, VEC_W=vw, ROWS_PER_THREAD=rpt, HAS_BIAS=has_bias, wait=False, sync_mps=False,
        )
        return out

    # Winograd F(2x2, 3x3) fast path for 2D 3x3 stride=1 dilation=1 groups=1 f32.
    if (
        ndim == 2
        and k == (3, 3)
        and s == (1, 1)
        and d == (1, 1)
        and groups == 1
        and x_c.dtype == torch.float32
        and w_c.dtype == torch.float32
        and (b_c is None or b_c.dtype == torch.float32)
        and c_in >= 8
        and c_out >= 16
        and spatial_out[0] * spatial_out[1] >= 256
    ):
        bk = 16 if (c_in >= 16 and c_in % 16 == 0) else 8
        if c_in <= 16 and c_out >= 64:
            bm, bn, tth, ttw, nsg = 32, 16, 1, 16, 8
        else:
            pow2_w = spatial_in[1] > 0 and (spatial_in[1] & (spatial_in[1] - 1)) == 0
            bm, bn, tth, ttw, nsg = 32, 32, 2, 16, (16 if pow2_w else 8)
        c_in_pad = _cdiv(c_in, bk) * bk
        c_out_pad = _cdiv(c_out, bm) * bm
        u = _get_or_transform_wino2x3_filter(weight, w_c, c_in, c_out, c_in_pad, c_out_pad, runner)
        th, tw = _cdiv(spatial_out[0], 2), _cdiv(spatial_out[1], 2)
        grid = (_cdiv(th, tth) * _cdiv(tw, ttw), _cdiv(c_out, bm), n)
        _conv2d_wino2x3_kernel[grid](
            x_c, u, out, b_arg,
            N=n, C_in=c_in, C_out=c_out, H_in=spatial_in[0], W_in=spatial_in[1],
            H_out=spatial_out[0], W_out=spatial_out[1], P_H=p[0], P_W=p[1],
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TILE_TH=tth, TILE_TW=ttw,
            NUM_SG=nsg, WINOGRAD=1, WINO_PAD=1, HAS_BIAS=has_bias, wait=False, sync_mps=False,
        )
        return out

    pixels = n * math.prod(spatial_out)
    pointwise = all(v == 1 for v in k) and all(v == 1 for v in s) and all(v == 0 for v in p)
    c_out_g = c_out // groups
    bm, bn, bk, num_sg = _select_conv_tile(c_in // groups, c_out_g, math.prod(k), pointwise, ndim=ndim, stride_w=s[-1])
    grid = (_cdiv(pixels, bn), _cdiv(c_out_g, bm), groups)

    pad_n = 3 - ndim
    si3 = (1,) * pad_n + tuple(spatial_in)
    so3 = (1,) * pad_n + tuple(spatial_out)
    k3 = (1,) * pad_n + k
    s3 = (1,) * pad_n + s
    p3 = (0,) * pad_n + p
    d3 = (1,) * pad_n + d
    _conv_std_kernel[grid](
        x_c, w_c, out, b_arg,
        N=n, C_in=c_in, C_out=c_out,
        D_in=si3[0], H_in=si3[1], W_in=si3[2], D_out=so3[0], H_out=so3[1], W_out=so3[2],
        K_D=k3[0], K_H=k3[1], K_W=k3[2], S_D=s3[0], S_H=s3[1], S_W=s3[2],
        P_D=p3[0], P_H=p3[1], P_W=p3[2], D_D=d3[0], D_H=d3[1], D_W=d3[2],
        GROUPS=int(groups), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=num_sg,
        NDIM=ndim, HAS_BIAS=has_bias, wait=False, sync_mps=False,
    )
    return out


def _pool_output_size(size: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    span = size + 2 * p - d * (k - 1) - 1
    out = (-(-span // s) if ceil_mode else span // s) + 1
    if ceil_mode and (out - 1) * s >= size + p:
        out -= 1
    return out


def _norm_pool_tuple(val: Any, ndim: int, default: tuple[int, ...]) -> tuple[int, ...]:
    if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
        return default
    if isinstance(val, int):
        return (int(val),) * ndim
    vals = tuple(int(v) for v in val)
    return vals * ndim if len(vals) == 1 else vals


def _select_pool_block(
    ndim: int,
    kind: str,
    k: tuple[int, ...],
    s: tuple[int, ...],
    d: tuple[int, ...],
    spatial_out: list[int],
) -> tuple[int, int, int, int]:
    if ndim == 1:
        return (1024 if s[0] == 1 else 256), 1, 1, 0
    if ndim == 2:
        if kind == "max" and s == (1, 1) and d == (1, 1) and k[0] <= 8 and k[1] <= 8:
            h_out = spatial_out[0]
            if h_out >= 64:
                return 128, 64, 4, 16
            if h_out >= 32:
                return 128, 32, 4, 8
            return 128, 16, 4, 4
        if d[1] == 1 and s[1] >= k[1] and k[1] in (2, 4):
            return 32, 8, 4, 0
        return 32, 32, 1, 0
    return 32, 8, 1, 0


def _launch_pool(
    x: torch.Tensor,
    *,
    kind: str,
    ndim: int,
    kernel_size: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    count_include_pad: bool,
    ceil_mode: bool,
    need_indices: bool,
    runner: _BatchRunner,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    squeezed = False
    x_in = x
    if x_in.ndim == ndim + 1:
        x_in = x_in.unsqueeze(0)
        squeezed = True
    x_c = runner.ensure_contig(x_in)

    k = _norm_pool_tuple(kernel_size, ndim, (2,) * ndim)
    s = _norm_pool_tuple(stride, ndim, k)
    p = _norm_pool_tuple(padding, ndim, (0,) * ndim)
    d = _norm_pool_tuple(dilation, ndim, (1,) * ndim) if kind == "max" else (1,) * ndim

    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [_pool_output_size(spatial_in[i], k[i], s[i], p[i], d[i], ceil_mode) for i in range(ndim)]
    out = torch.empty((n, c, *spatial_out), dtype=x_c.dtype, device=x_c.device)

    if out.numel() == 0:
        idx_out = torch.empty_like(out, dtype=torch.int64) if need_indices else None
        if squeezed:
            out = out.squeeze(0)
            if idx_out is not None:
                idx_out = idx_out.squeeze(0)
        return out, idx_out

    runner.begin_m1(x_c.device.type == "mps")
    if need_indices and kind == "max" and ndim == 2:
        idx_out = torch.empty((n, c, *spatial_out), dtype=torch.int64, device=x_c.device)
        total = n * c * spatial_out[0] * spatial_out[1]
        block = 256
        _max_pool2d_with_indices_kernel[(_cdiv(total, block),)](
            x_c, out, idx_out, n * c, spatial_in[0], spatial_in[1], spatial_out[0], spatial_out[1],
            K_H=k[0], K_W=k[1], S_H=s[0], S_W=s[1], P_H=p[0], P_W=p[1], D_H=d[0], D_W=d[1],
            BLOCK_SIZE=block, wait=False, sync_mps=False,
        )
        return (out.squeeze(0), idx_out.squeeze(0)) if squeezed else (out, idx_out)

    bw, bh, vw, rpt = _select_pool_block(ndim, kind, k, s, d, spatial_out)
    pad_n = 3 - ndim
    si3 = (1,) * pad_n + tuple(spatial_in)
    so3 = (1,) * pad_n + tuple(spatial_out)
    k3 = (1,) * pad_n + k
    s3 = (1,) * pad_n + s
    p3 = (0,) * pad_n + p
    d3 = (1,) * pad_n + d
    grid = (_cdiv(so3[2], bw), _cdiv(so3[1], bh), n * c * so3[0])
    _pool_nd_kernel[grid](
        x_c, out, n, c, si3[0], si3[1], si3[2], so3[0], so3[1], so3[2],
        KIND=kind, K_D=k3[0], K_H=k3[1], K_W=k3[2], S_D=s3[0], S_H=s3[1], S_W=s3[2],
        P_D=p3[0], P_H=p3[1], P_W=p3[2], D_D=d3[0], D_H=d3[1], D_W=d3[2],
        COUNT_INCLUDE_PAD=bool(count_include_pad), CEIL_MODE=bool(ceil_mode),
        BLOCK_W=bw, BLOCK_H=bh, VEC_W=vw, ROWS_PER_THREAD=rpt, NDIM=ndim, wait=False, sync_mps=False,
    )
    if squeezed:
        out = out.squeeze(0)
    return out, None


def _cast_tensor_m1(x: torch.Tensor, target_dtype: torch.dtype, runner: _BatchRunner) -> torch.Tensor:
    if x.dtype == target_dtype:
        return x
    x_c = runner.ensure_contig(x)
    out = torch.empty(x_c.shape, dtype=target_dtype, device=x_c.device)
    n = x_c.numel()
    if n == 0:
        return out
    block = 1024 if n >= 1024 else 256
    runner.begin_m1(x_c.device.type == "mps")
    _cast_kernel[(_cdiv(n, block),)](x_c, out, n, BLOCK_SIZE=block, wait=False, sync_mps=False)
    return out


def _launch_softmax(
    x: torch.Tensor,
    dim: int,
    *,
    log_softmax: bool,
    half_to_float: bool,
    runner: _BatchRunner,
) -> torch.Tensor:
    x_c = runner.ensure_contig(x)
    orig_dtype = x_c.dtype
    work = _cast_tensor_m1(x_c, torch.float32, runner) if x_c.dtype == torch.float16 else x_c
    ndim = work.ndim
    d = dim if dim >= 0 else dim + ndim
    if ndim == 0:
        out = torch.empty_like(work)
        runner.begin_m1(work.device.type == "mps")
        _fill_const_kernel[(1,)](out, 0.0 if log_softmax else 1.0, 1, BLOCK_SIZE=32, wait=False, sync_mps=False)
        return out if (half_to_float or orig_dtype == torch.float32) else _cast_tensor_m1(out, orig_dtype, runner)

    out = torch.empty_like(work)
    if out.numel() == 0:
        return out if (half_to_float or orig_dtype == torch.float32) else _cast_tensor_m1(out, orig_dtype, runner)

    runner.begin_m1(work.device.type == "mps")
    if d == ndim - 1:
        row_len = int(work.shape[-1])
        num_rows = work.numel() // max(row_len, 1)
        _online_softmax_kernel[(num_rows, 1, 1)](
            work, out, num_rows, row_len, bool(log_softmax), wait=False, sync_mps=False
        )
    else:
        shape = tuple(int(s) for s in work.shape)
        outer = math.prod(shape[:d]) if d > 0 else 1
        reduce_len = shape[d]
        inner = math.prod(shape[d + 1 :])
        block_inner = 128
        grid = (_cdiv(inner, block_inner), outer)
        _strided_softmax_kernel[grid](
            work,
            out,
            reduce_len,
            inner,
            BLOCK_INNER=block_inner,
            LOG_SOFTMAX=1 if log_softmax else 0,
            wait=False,
            sync_mps=False,
        )
    if orig_dtype == torch.float16 and not half_to_float:
        return _cast_tensor_m1(out, orig_dtype, runner)
    return out


def _prepare_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    runner: _BatchRunner,
) -> tuple[torch.dtype, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x_c = runner.ensure_contig(x)
    orig_dtype = x_c.dtype
    work = _cast_tensor_m1(x_c, torch.float32, runner) if orig_dtype == torch.float16 else x_c
    w_c = _cast_tensor_m1(runner.ensure_contig(weight), torch.float32, runner) if weight is not None else None
    b_c = _cast_tensor_m1(runner.ensure_contig(bias), torch.float32, runner) if bias is not None else None
    return orig_dtype, work, w_c, b_c


def _extract_norm_stats(
    partial: torch.Tensor,
    stat_shape: tuple[int, ...],
    num_groups: int,
    chunks: int,
    eps: float,
    need_stats: bool,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not need_stats:
        return None, None
    mean_t = torch.empty(stat_shape, dtype=torch.float32, device=device)
    rstd_t = torch.empty(stat_shape, dtype=torch.float32, device=device)
    if num_groups > 0:
        blk = 256
        _welford_partials_extract_kernel[(_cdiv(num_groups, blk),)](
            partial, mean_t, rstd_t, num_groups, CHUNKS=chunks, eps=float(eps), BLOCK_SIZE=blk, wait=False, sync_mps=False
        )
    return mean_t, rstd_t


def _launch_layer_norm(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    need_stats: bool,
    runner: _BatchRunner,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    orig_dtype, work, w_c, b_c = _prepare_norm_inputs(x, weight, bias, runner)
    norm_size = int(math.prod(int(v) for v in normalized_shape))
    outer_shape = tuple(int(v) for v in work.shape[: work.ndim - len(normalized_shape)])
    n = work.numel() // max(norm_size, 1)
    chunks = max(1, min(128, _cdiv(norm_size, 65536)))
    partial = torch.empty(max(n * chunks * 3, 1), dtype=torch.float32, device=work.device)
    out = torch.empty_like(work)

    if out.numel() > 0:
        runner.begin_m1(work.device.type == "mps")
        _welford_norm_stats_kernel[(chunks, n, 1)](
            work, partial, N=n, D1=norm_size, D2=1, CHUNKS=chunks, KIND=0, wait=False, sync_mps=False
        )
        tiles = max(1, _cdiv(norm_size, 8192))
        _welford_norm_apply_kernel[(tiles, n, 1)](
            work, out, w_c if w_c is not None else work, b_c if b_c is not None else work, partial,
            N=n, D1=norm_size, D2=1, CHUNKS=chunks, EPS=float(eps), KIND=0,
            HAS_WEIGHT=1 if w_c is not None else 0, HAS_BIAS=1 if b_c is not None else 0,
            wait=False, sync_mps=False,
        )

    mean_t, rstd_t = _extract_norm_stats(
        partial, outer_shape + (1,) * len(normalized_shape), n, chunks, eps, need_stats, work.device
    )
    final_out = _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype == torch.float16 else out
    return final_out, mean_t, rstd_t


def _launch_group_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    need_stats: bool,
    runner: _BatchRunner,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    orig_dtype, work, w_c, b_c = _prepare_norm_inputs(x, weight, bias, runner)
    n, c = int(work.shape[0]), int(work.shape[1])
    hw = work.numel() // max(n * c, 1)
    chunks = max(1, min(32, _cdiv((c // num_groups) * hw, 65536)))
    partial = torch.empty(max(n * num_groups * chunks * 3, 1), dtype=torch.float32, device=work.device)
    out = torch.empty_like(work)

    if out.numel() > 0:
        runner.begin_m1(work.device.type == "mps")
        _welford_norm_stats_kernel[(chunks, n * num_groups, 1)](
            work, partial, N=n, D1=c, D2=hw, CHUNKS=chunks, KIND=1, GROUPS=int(num_groups), wait=False, sync_mps=False
        )
        tiles = max(1, _cdiv(hw, 8192))
        _welford_norm_apply_kernel[(tiles, n * c, 1)](
            work, out, w_c if w_c is not None else work, b_c if b_c is not None else work, partial,
            N=n, D1=c, D2=hw, CHUNKS=chunks, EPS=float(eps), KIND=1,
            HAS_WEIGHT=1 if w_c is not None else 0, HAS_BIAS=1 if b_c is not None else 0,
            GROUPS=int(num_groups), wait=False, sync_mps=False,
        )

    mean_t, rstd_t = _extract_norm_stats(
        partial, (n, num_groups), n * num_groups, chunks, eps, need_stats, work.device
    )
    final_out = _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype == torch.float16 else out
    return final_out, mean_t, rstd_t


def _launch_batch_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    eps: float,
    need_stats: bool,
    runner: _BatchRunner,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    orig_dtype, work, w_c, b_c = _prepare_norm_inputs(x, weight, bias, runner)
    n, c = int(work.shape[0]), int(work.shape[1])
    hw = work.numel() // max(n * c, 1)
    out = torch.empty_like(work)
    if out.numel() == 0:
        final_out = _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype == torch.float16 else out
        if need_stats:
            return (
                final_out,
                torch.empty((c,), dtype=torch.float32, device=work.device),
                torch.empty((c,), dtype=torch.float32, device=work.device),
            )
        return final_out, None, None

    if not training and running_mean is not None and running_var is not None:
        rm_c = _cast_tensor_m1(runner.ensure_contig(running_mean), torch.float32, runner)
        rv_c = _cast_tensor_m1(runner.ensure_contig(running_var), torch.float32, runner)
        block = 1024 if hw >= 1024 else (512 if hw >= 512 else 256)
        runner.begin_m1(work.device.type == "mps")
        _batch_norm_eval_kernel[(_cdiv(hw, block), n * c)](
            work, out, rm_c, rv_c, w_c if w_c is not None else rm_c, b_c if b_c is not None else rm_c,
            c, hw, float(eps), HAS_WEIGHT=1 if w_c is not None else 0, HAS_BIAS=1 if b_c is not None else 0,
            BLOCK_SIZE=block, wait=False, sync_mps=False,
        )
        final_out = _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype == torch.float16 else out
        return final_out, None, None

    chunks = max(1, min(n, _cdiv(n * hw, 65536)))
    partial = torch.empty(max(c * chunks * 3, 1), dtype=torch.float32, device=work.device)
    runner.begin_m1(work.device.type == "mps")
    _welford_norm_stats_kernel[(chunks, c, 1)](
        work, partial, N=n, D1=c, D2=hw, CHUNKS=chunks, KIND=2, wait=False, sync_mps=False
    )
    tiles = max(1, _cdiv(hw, 8192))
    _welford_norm_apply_kernel[(tiles, n * c, 1)](
        work, out, w_c if w_c is not None else work, b_c if b_c is not None else work, partial,
        N=n, D1=c, D2=hw, CHUNKS=chunks, EPS=float(eps), KIND=2,
        HAS_WEIGHT=1 if w_c is not None else 0, HAS_BIAS=1 if b_c is not None else 0,
        wait=False, sync_mps=False,
    )

    mean_t, rstd_t = _extract_norm_stats(partial, (c,), c, chunks, eps, need_stats, work.device)
    final_out = _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype == torch.float16 else out
    return final_out, mean_t, rstd_t


def _reduce_op_kind(kind: str) -> int:
    return 0 if kind in ("sum", "mean") else (1 if kind == "amax" else 2)


def _fill_empty_reduce_output(out: torch.Tensor, kind: str, dim_label: str, runner: _BatchRunner) -> torch.Tensor:
    if kind in ("amax", "amin"):
        raise RuntimeError(f"{kind}(): cannot perform reduction on empty {dim_label}")
    n_out = out.numel()
    if n_out > 0:
        fill_val = 0.0 if kind == "sum" else float("nan")
        blk = 1024 if n_out >= 1024 else (256 if n_out > 32 else 32)
        runner.begin_m1(out.device.type == "mps")
        _fill_const_kernel[(_cdiv(n_out, blk),)](out, fill_val, n_out, BLOCK_SIZE=blk, wait=False, sync_mps=False)
    return out


def _launch_single_axis_reduce(
    x: torch.Tensor,
    dim: int,
    *,
    kind: str,
    keepdim: bool,
    scale: float | None,
    runner: _BatchRunner,
) -> torch.Tensor:
    x_c = runner.ensure_contig(x)
    ndim = x_c.ndim
    d = dim if dim >= 0 else dim + ndim
    shape = tuple(int(s) for s in x_c.shape)
    outer = math.prod(shape[:d]) if d > 0 else 1
    reduce_len = shape[d]
    inner = math.prod(shape[d + 1 :]) if (d + 1) < ndim else 1
    out_shape = (shape[:d] + (1,) + shape[d + 1 :]) if keepdim else (shape[:d] + shape[d + 1 :])
    out = torch.empty(out_shape, dtype=x_c.dtype, device=x_c.device)
    if reduce_len == 0:
        return _fill_empty_reduce_output(out, kind, f"dim {d}", runner)
    if outer == 0 or inner == 0:
        return out
    runner.begin_m1(x_c.device.type == "mps")
    op_kind = _reduce_op_kind(kind)
    apply_scale = 1 if (op_kind == 0 and scale is not None and scale != 1.0) else 0
    scale_f = 1.0 if scale is None else float(scale)

    if inner == 1:
        block_k = 256 if reduce_len <= 256 else 1024
        _last_axis_reduce_kernel[(outer,)](
            x_c, out, reduce_len, scale_f, BLOCK_K=block_k, OP_KIND=op_kind, APPLY_SCALE=apply_scale,
            wait=False, sync_mps=False,
        )
    else:
        block_inner = 128
        grid = (_cdiv(inner, block_inner), outer)
        _axis_reduce_strided_kernel[grid](
            x_c, out, reduce_len, inner, scale_f, BLOCK_INNER=block_inner, OP_KIND=op_kind, APPLY_SCALE=apply_scale,
            wait=False, sync_mps=False,
        )
    return out


def _launch_reduce(
    x: torch.Tensor,
    *,
    kind: str,
    dims: Sequence[int] | None,
    keepdim: bool,
    out_dtype: torch.dtype | None,
    runner: _BatchRunner,
) -> torch.Tensor:
    orig_dtype = out_dtype if out_dtype is not None else x.dtype
    work = _cast_tensor_m1(runner.ensure_contig(x), torch.float32, runner) if x.dtype != torch.float32 else runner.ensure_contig(x)
    ndim = work.ndim

    if dims is None or len(dims) == 0:
        norm_dims = list(range(ndim))
    else:
        norm_dims = sorted({(int(d) if int(d) >= 0 else int(d) + ndim) for d in dims})

    if ndim == 0 or not norm_dims:
        return _cast_tensor_m1(work, orig_dtype, runner) if orig_dtype != work.dtype else work

    # Full-tensor reduction fast path (2-stage parallel reduction for large tensors).
    if len(norm_dims) == ndim:
        n = work.numel()
        out_shape = (1,) * ndim if keepdim else ()
        out = torch.empty(out_shape, dtype=torch.float32, device=work.device)
        if n == 0:
            _fill_empty_reduce_output(out, kind, "tensor", runner)
            return _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype != out.dtype else out
        runner.begin_m1(work.device.type == "mps")
        op_kind = _reduce_op_kind(kind)
        scale = (1.0 / float(n)) if kind == "mean" else 1.0
        apply_scale = 1 if kind == "mean" else 0
        if n > 8192:
            block = 1024
            num_blocks = _cdiv(n, block)
            partial = torch.empty((num_blocks,), dtype=torch.float32, device=work.device)
            _full_reduce_stage1_kernel[(num_blocks,)](
                work, partial, n, BLOCK_SIZE=block, OP_KIND=op_kind, wait=False, sync_mps=False
            )
            block_k = 256 if num_blocks <= 256 else 1024
            _last_axis_reduce_kernel[(1,)](
                partial, out, num_blocks, scale, BLOCK_K=block_k, OP_KIND=op_kind, APPLY_SCALE=apply_scale,
                wait=False, sync_mps=False,
            )
        else:
            block_k = 256 if n <= 256 else 1024
            _last_axis_reduce_kernel[(1,)](
                work, out, n, scale, BLOCK_K=block_k, OP_KIND=op_kind, APPLY_SCALE=apply_scale,
                wait=False, sync_mps=False,
            )
        return _cast_tensor_m1(out, orig_dtype, runner) if orig_dtype != out.dtype else out

    # Contiguous trailing dims (e.g. spatial dims [2, 3] on NCHW) collapse into one last-axis reduction.
    if norm_dims == list(range(norm_dims[0], ndim)) and len(norm_dims) > 1:
        lead_shape = tuple(int(s) for s in work.shape[: norm_dims[0]])
        red_size = int(math.prod(int(s) for s in work.shape[norm_dims[0] :]))
        flat = work.view(*lead_shape, red_size)
        scale = (1.0 / float(red_size)) if (kind == "mean" and red_size > 0) else None
        reduced = _launch_single_axis_reduce(
            flat, len(lead_shape), kind=kind, keepdim=False, scale=scale, runner=runner
        )
        if keepdim:
            reduced = reduced.view(*lead_shape, *((1,) * len(norm_dims)))
        return _cast_tensor_m1(reduced, orig_dtype, runner) if orig_dtype != reduced.dtype else reduced

    # Sequential multi-dim reduction in descending dim order.
    cur = work
    for d in reversed(norm_dims):
        red_len = int(cur.shape[d])
        scale = (1.0 / float(red_len)) if (kind == "mean" and red_len > 0) else None
        cur = _launch_single_axis_reduce(cur, d, kind=kind, keepdim=keepdim, scale=scale, runner=runner)
    return _cast_tensor_m1(cur, orig_dtype, runner) if orig_dtype != cur.dtype else cur


# ============================================================================
# FX Graph Partitioning & Fusion IR
# ============================================================================


@dataclass
class _FusedPointwiseStep:
    plan: _FusedPointwisePlan


@dataclass
class _GemmStep:
    node: fx.Node
    kind: str  # "mm", "addmm", "bmm", "linear", "matmul"
    a_name: str
    b_name: str
    bias_name: str | None
    trans_a: bool
    trans_b: bool
    alpha: float
    beta: float
    epilogue_plan: _FusedPointwisePlan | None = None
    fused_node_names: tuple[str, ...] = ()


@dataclass
class _ConvStep:
    node: fx.Node


@dataclass
class _PoolStep:
    node: fx.Node
    kind: str
    ndim: int
    returns_tuple: bool
    need_indices: bool


@dataclass
class _SoftmaxStep:
    node: fx.Node
    log_softmax: bool


@dataclass
class _NormStep:
    node: fx.Node
    kind: str  # "layer_norm", "group_norm", "batch_norm"
    need_stats: bool


@dataclass
class _ReduceStep:
    node: fx.Node
    kind: str  # "sum", "mean", "amax", "amin"


@dataclass
class _FillStep:
    node: fx.Node
    fill_val: float
    shape_or_like: Any
    dtype: torch.dtype | None
    device: torch.device | None


@dataclass
class _MetadataStep:
    node: fx.Node


@dataclass
class _EagerStep:
    node: fx.Node


_Step = (
    _FusedPointwiseStep
    | _GemmStep
    | _ConvStep
    | _PoolStep
    | _SoftmaxStep
    | _NormStep
    | _ReduceStep
    | _FillStep
    | _MetadataStep
    | _EagerStep
)


def _arg_node_names(node: fx.Node) -> list[str]:
    names: list[str] = []

    def visit(x: Any) -> None:
        if isinstance(x, fx.Node):
            names.append(x.name)
        elif isinstance(x, (tuple, list)):
            for e in x:
                visit(e)

    visit(node.args)
    for v in node.kwargs.values():
        visit(v)
    return names


def _extract_2d_transpose_src(arg_node: fx.Node) -> fx.Node | None:
    if arg_node.op != "call_function" or len(arg_node.users) != 1:
        return None
    if arg_node.target == aten.t.default and isinstance(arg_node.args[0], fx.Node):
        src = arg_node.args[0]
        shape, _, _ = _node_tensor_meta(src)
        if shape is not None and len(shape) == 2:
            return src
    if arg_node.target == aten.transpose.int and isinstance(arg_node.args[0], fx.Node):
        d0, d1 = int(arg_node.args[1]), int(arg_node.args[2])
        src = arg_node.args[0]
        shape, _, _ = _node_tensor_meta(src)
        if shape is not None and len(shape) == 2 and ({d0 % 2, d1 % 2} == {0, 1}):
            return src
    return None


def _resolve_gemm_2d_transposes(
    gemm_node: fx.Node,
    a_node: fx.Node,
    b_node: fx.Node,
    absorbed_transposes: set[str],
) -> tuple[str, bool, str, bool]:
    """Absorb single-use 2D transpose inputs into GEMM `TRANS_A/B` only when every slot using the node is absorbed."""
    all_refs = _arg_node_names(gemm_node)
    if a_node is b_node:
        src = _extract_2d_transpose_src(a_node)
        if src is not None and all_refs.count(a_node.name) == 2:
            absorbed_transposes.add(a_node.name)
            return src.name, True, src.name, True
        return a_node.name, False, b_node.name, False

    a_name, ta = a_node.name, False
    src_a = _extract_2d_transpose_src(a_node)
    if src_a is not None and all_refs.count(a_node.name) == 1:
        absorbed_transposes.add(a_node.name)
        a_name, ta = src_a.name, True

    b_name, tb = b_node.name, False
    src_b = _extract_2d_transpose_src(b_node)
    if src_b is not None and all_refs.count(b_node.name) == 1:
        absorbed_transposes.add(b_node.name)
        b_name, tb = src_b.name, True

    return a_name, ta, b_name, tb


def _tuple_needs_index(node: fx.Node, idx: int) -> bool:
    for u in node.users:
        if u.op == "call_function" and u.target is operator.getitem:
            if len(u.args) >= 2 and u.args[1] == idx and len(u.users) > 0:
                return True
        else:
            return True
    return False


def _is_supported_gemm(node: fx.Node) -> bool:
    if node.target not in _GEMM_TARGETS:
        return False
    shape, dt, dev = _node_tensor_meta(node)
    if dt not in _SUPPORTED_FLOAT_DTYPES or dev not in _SUPPORTED_DEVICES:
        return False
    if node.target == aten.matmul.default:
        a_sh, _, _ = _node_tensor_meta(node.args[0])
        b_sh, _, _ = _node_tensor_meta(node.args[1])
        if a_sh is None or b_sh is None:
            return False
        if len(a_sh) == 2 and len(b_sh) == 2:
            return True
        if len(a_sh) == 3 and len(b_sh) == 3 and a_sh[0] == b_sh[0]:
            return True
        if len(a_sh) >= 2 and len(b_sh) == 2:
            return True
        return False
    return True


def _is_supported_conv(node: fx.Node) -> bool:
    if node.target not in _CONV_TARGETS:
        return False
    shape, dt, dev = _node_tensor_meta(node)
    if shape is None or dt not in _SUPPORTED_FLOAT_DTYPES or dev not in _SUPPORTED_DEVICES:
        return False
    transposed = bool(node.args[6]) if len(node.args) > 6 else False
    if transposed:
        return False
    return (len(shape) - 2) in (1, 2, 3)


def _is_supported_pool(node: fx.Node) -> bool:
    if node.target not in _POOL_TARGETS:
        return False
    x_node = node.args[0]
    if not isinstance(x_node, fx.Node):
        return False
    x_sh, x_dt, x_dev = _node_tensor_meta(x_node)
    if x_sh is None or x_dt not in _SUPPORTED_FLOAT_DTYPES or x_dev not in _SUPPORTED_DEVICES:
        return False
    if node.target == aten.avg_pool2d.default:
        div_ovr = node.args[6] if len(node.args) >= 7 else node.kwargs.get("divisor_override")
        if div_ovr is not None:
            return False
    if node.target in (aten.max_pool1d_with_indices.default, aten.max_pool3d_with_indices.default):
        if _tuple_needs_index(node, 1):
            return False
    return True


def _is_supported_softmax(node: fx.Node) -> bool:
    if node.target not in _SOFTMAX_TARGETS:
        return False
    _, dt, dev = _node_tensor_meta(node.args[0])
    return dt in _SUPPORTED_FLOAT_DTYPES and dev in _SUPPORTED_DEVICES


def _is_supported_norm(node: fx.Node) -> bool:
    if node.target not in _NORM_TARGETS:
        return False
    x_node = node.args[0]
    if not isinstance(x_node, fx.Node):
        return False
    x_sh, x_dt, x_dev = _node_tensor_meta(x_node)
    if x_sh is None or x_dt not in _SUPPORTED_FLOAT_DTYPES or x_dev not in _SUPPORTED_DEVICES:
        return False
    t = node.target
    need_stats = _tuple_needs_index(node, 1) or _tuple_needs_index(node, 2)
    if t == aten._native_batch_norm_legit_functional.default:
        if _tuple_needs_index(node, 3) or _tuple_needs_index(node, 4):
            return False
        r_mean = node.args[3] if len(node.args) > 3 else None
        r_var = node.args[4] if len(node.args) > 4 else None
        training = node.args[5] if len(node.args) > 5 else True
        if not isinstance(training, bool):
            return False
        if training and (r_mean is not None or r_var is not None):
            return False
        if not training and need_stats:
            return False
    elif t == aten._native_batch_norm_legit_no_training.default:
        if need_stats:
            return False
    elif t in (aten.native_batch_norm.default, aten._native_batch_norm_legit.default):
        r_mean = node.args[3] if len(node.args) > 3 else None
        r_var = node.args[4] if len(node.args) > 4 else None
        training = node.args[5] if len(node.args) > 5 else True
        if not isinstance(training, bool):
            return False
        if training and (r_mean is not None or r_var is not None):
            return False
        if not training and need_stats:
            return False
    return True


def _is_supported_reduce(node: fx.Node) -> bool:
    if node.target not in _REDUCE_TARGETS:
        return False
    x_node = node.args[0]
    if not isinstance(x_node, fx.Node):
        return False
    x_sh, x_dt, x_dev = _node_tensor_meta(x_node)
    if x_dt not in _SUPPORTED_FLOAT_DTYPES or x_dev not in _SUPPORTED_DEVICES:
        return False
    if node.target in (aten.amax.default, aten.amin.default) and x_sh is not None:
        ndim = len(x_sh)
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", [])
        if not dims:
            if math.prod(x_sh) == 0:
                return False
        else:
            for d in dims:
                nd = int(d) if int(d) >= 0 else int(d) + ndim
                if 0 <= nd < ndim and x_sh[nd] == 0:
                    return False
    return True


def _is_supported_fill(node: fx.Node) -> bool:
    if node.target not in _FACTORY_FILL_TARGETS:
        return False
    shape, dt, dev = _node_tensor_meta(node)
    if shape is None or dt not in _SUPPORTED_PW_DTYPES or dev not in _SUPPORTED_DEVICES:
        return False
    return True


def _build_fused_plan(
    gm: fx.GraphModule,
    *,
    strict: bool,
) -> tuple[list[str], list[_Step], Any]:
    placeholders: list[str] = []
    output_spec: Any = None
    nodes = list(gm.graph.nodes)
    node_by_name = {n.name: n for n in nodes}

    # Pre-scan for 2D transpose nodes that can be absorbed directly into GEMM TRANS_A/TRANS_B.
    absorbed_transposes: set[str] = set()
    gemm_transpose_info: dict[str, tuple[str, bool, str, bool]] = {}
    for n in nodes:
        if n.op == "call_function" and _is_supported_gemm(n):
            if n.target in (aten.mm.default, aten.matmul.default):
                gemm_transpose_info[n.name] = _resolve_gemm_2d_transposes(
                    n, n.args[0], n.args[1], absorbed_transposes
                )
            elif n.target == aten.addmm.default:
                gemm_transpose_info[n.name] = _resolve_gemm_2d_transposes(
                    n, n.args[1], n.args[2], absorbed_transposes
                )

    # Identify which node outputs are consumed by the graph output node.
    graph_output_names: set[str] = set()
    for n in nodes:
        if n.op == "output":
            output_spec = n.args[0]
            graph_output_names.update(_arg_node_names(n))

    steps: list[_Step] = []
    consumed: set[str] = set()

    i = 0
    while i < len(nodes):
        n = nodes[i]
        if n.op == "placeholder":
            placeholders.append(n.name)
            i += 1
            continue
        if n.op == "output":
            i += 1
            continue
        if n.name in consumed or n.name in absorbed_transposes:
            i += 1
            continue

        # 1. Greedy contiguous pointwise DAG grouping (including broadcast-compatible intermediates).
        if _is_supported_pointwise_node(n):
            group: list[fx.Node] = [n]
            group_names: set[str] = {n.name}
            out_shape, _, _ = _node_tensor_meta(n)
            assert out_shape is not None
            j = i + 1
            while j < len(nodes):
                cand = nodes[j]
                if cand.op != "call_function" or cand.name in consumed or cand.name in absorbed_transposes:
                    break
                if not _is_supported_pointwise_node(cand):
                    break
                c_shape, _, _ = _node_tensor_meta(cand)
                if c_shape is None:
                    break
                if c_shape != out_shape:
                    try:
                        merged_shape = tuple(int(d) for d in torch.broadcast_shapes(out_shape, c_shape))
                    except RuntimeError:
                        break
                    # Only promote/absorb broadcast-different intermediates that have no external consumers.
                    if merged_shape != out_shape:
                        if any(
                            (_node_tensor_meta(gn)[0] != merged_shape) and (gn.name in graph_output_names)
                            for gn in group
                        ):
                            break
                        out_shape = merged_shape
                    elif cand.name in graph_output_names:
                        break
                group.append(cand)
                group_names.add(cand.name)
                j += 1

            # Verify no smaller-shape intermediate inside `group` escapes to an external consumer.
            while len(group) > 1:
                bad_escape = False
                for gn in group:
                    g_sh = _node_tensor_meta(gn)[0]
                    if g_sh != out_shape:
                        used_outside = (gn.name in graph_output_names) or any(
                            u.name not in group_names for u in gn.users
                        )
                        if used_outside:
                            bad_escape = True
                            break
                if not bad_escape:
                    break
                dropped = group.pop()
                group_names.remove(dropped.name)
                j -= 1
                out_shape = tuple(int(d) for d in torch.broadcast_shapes(*[_node_tensor_meta(gn)[0] for gn in group]))

            ext_outs: list[fx.Node] = []
            for gn in group:
                used_outside = (gn.name in graph_output_names) or any(
                    u.name not in group_names for u in gn.users
                )
                if used_outside:
                    ext_outs.append(gn)
            if not ext_outs:
                ext_outs.append(group[-1])

            plan = _synthesize_pointwise_plan(group, ext_outs, out_shape)
            steps.append(_FusedPointwiseStep(plan=plan))
            consumed.update(group_names)
            i = j
            continue

        # 2. Linear / Matmul / Addmm / BMM + fused bias + trailing pointwise epilogue chain.
        if n.op == "call_function" and _is_supported_gemm(n):
            if n.target == aten.linear.default:
                a_name = n.args[0].name
                b_name = n.args[1].name
                bias_arg = n.args[2] if len(n.args) >= 3 else n.kwargs.get("bias")
                bias_name = bias_arg.name if isinstance(bias_arg, fx.Node) else None
                kind, ta, tb, alpha, beta = "linear", False, True, 1.0, 1.0
            elif n.target == aten.addmm.default:
                bias_name = n.args[0].name
                a_name, ta, b_name, tb = gemm_transpose_info[n.name]
                beta = float(n.kwargs.get("beta", 1.0))
                alpha = float(n.kwargs.get("alpha", 1.0))
                kind = "addmm"
            elif n.target == aten.bmm.default:
                a_name, b_name = n.args[0].name, n.args[1].name
                bias_name, kind, ta, tb, alpha, beta = None, "bmm", False, False, 1.0, 1.0
            elif n.target == aten.mm.default:
                a_name, ta, b_name, tb = gemm_transpose_info[n.name]
                bias_name, kind, alpha, beta = None, "mm", 1.0, 1.0
            else:
                a_name, ta, b_name, tb = gemm_transpose_info[n.name]
                bias_name, kind, alpha, beta = None, "matmul", 1.0, 1.0

            gemm_shape, gemm_dt, _ = _node_tensor_meta(n)
            assert gemm_shape is not None and gemm_dt is not None

            # Greedily fuse bias + trailing pointwise chain into a single epilogue kernel.
            epi_nodes: list[fx.Node] = []
            epi_names: set[str] = set()
            j = i + 1
            while j < len(nodes):
                cand = nodes[j]
                if cand.op != "call_function" or cand.name in consumed or cand.name in absorbed_transposes:
                    break
                if not _is_supported_pointwise_node(cand):
                    break
                c_shape, _, _ = _node_tensor_meta(cand)
                if c_shape != gemm_shape:
                    break
                epi_nodes.append(cand)
                epi_names.add(cand.name)
                j += 1

            epilogue_plan: _FusedPointwisePlan | None = None
            fused_names: tuple[str, ...] = ()
            need_bias_epi = (bias_name is not None) or (alpha != 1.0)
            if need_bias_epi or epi_nodes:
                # Synthesize a single fused epilogue kernel taking `_gemm_raw` (+ optional bias + other inputs).
                raw_sym = f"__m1_gemm_raw_{n.name}"
                # Create synthetic FX-like lightweight descriptors for bias/alpha/beta before trailing ops.
                syn_nodes: list[fx.Node] = []
                # Reuse a tiny scratch Graph only for epilogue synthesis so we share 100% of pointwise codegen!
                scratch_g = fx.Graph()
                sym_map: dict[str, fx.Node] = {}

                def get_ph(orig_name: str, shape: tuple[int, ...], dt: torch.dtype) -> fx.Node:
                    if orig_name not in sym_map:
                        ph = scratch_g.placeholder(orig_name)
                        ph.meta["val"] = torch.empty(shape, dtype=dt, device="meta")
                        sym_map[orig_name] = ph
                    return sym_map[orig_name]

                cur_node = get_ph(raw_sym, gemm_shape, gemm_dt)
                if alpha != 1.0:
                    mul_n = scratch_g.call_function(aten.mul.Scalar, args=(cur_node, alpha))
                    mul_n.meta["val"] = torch.empty(gemm_shape, dtype=gemm_dt, device="meta")
                    syn_nodes.append(mul_n)
                    cur_node = mul_n
                if bias_name is not None:
                    b_sh, b_dt, _ = _node_tensor_meta(node_by_name[bias_name])
                    b_ph = get_ph(bias_name, b_sh or (), b_dt or gemm_dt)
                    if beta != 1.0:
                        b_scaled = scratch_g.call_function(aten.mul.Scalar, args=(b_ph, beta))
                        b_scaled.meta["val"] = torch.empty(gemm_shape, dtype=gemm_dt, device="meta")
                        syn_nodes.append(b_scaled)
                        add_n = scratch_g.call_function(aten.add.Tensor, args=(cur_node, b_scaled))
                    else:
                        add_n = scratch_g.call_function(aten.add.Tensor, args=(cur_node, b_ph))
                    add_n.meta["val"] = torch.empty(gemm_shape, dtype=gemm_dt, device="meta")
                    syn_nodes.append(add_n)
                    cur_node = add_n

                # Map original gemm node name to `cur_node`.
                orig_to_scratch: dict[str, fx.Node] = {n.name: cur_node}
                for en in epi_nodes:
                    def remap(arg: Any) -> Any:
                        if isinstance(arg, fx.Node):
                            if arg.name in orig_to_scratch:
                                return orig_to_scratch[arg.name]
                            ash, adt, _ = _node_tensor_meta(arg)
                            return get_ph(arg.name, ash or gemm_shape, adt or gemm_dt)
                        return arg

                    new_args = tuple(remap(a) for a in en.args)
                    new_kwargs = {k: remap(v) for k, v in en.kwargs.items()}
                    cloned = scratch_g.call_function(en.target, args=new_args, kwargs=new_kwargs)
                    cloned.name = en.name
                    esh, edt, _ = _node_tensor_meta(en)
                    cloned.meta["val"] = torch.empty(esh or gemm_shape, dtype=edt or gemm_dt, device="meta")
                    syn_nodes.append(cloned)
                    orig_to_scratch[en.name] = cloned

                # Determine which outputs from (n + epi_nodes) are needed externally.
                ext_out_map: list[tuple[str, fx.Node]] = []
                gemm_used_outside = (n.name in graph_output_names) or any(
                    u.name not in epi_names for u in n.users
                )
                if gemm_used_outside:
                    # Ensure gemm output has its own name in scratch graph.
                    if cur_node.name != n.name:
                        pass_n = scratch_g.call_function(aten.clone.default, args=(cur_node,))
                        pass_n.name = n.name
                        pass_n.meta["val"] = torch.empty(gemm_shape, dtype=gemm_dt, device="meta")
                        syn_nodes.append(pass_n)
                        ext_out_map.append((n.name, pass_n))
                    else:
                        ext_out_map.append((n.name, cur_node))

                for en in epi_nodes:
                    used_out = (en.name in graph_output_names) or any(
                        u.name not in epi_names for u in en.users
                    )
                    if used_out:
                        ext_out_map.append((en.name, orig_to_scratch[en.name]))

                if not ext_out_map:
                    last_orig = epi_nodes[-1].name if epi_nodes else n.name
                    ext_out_map.append((last_orig, orig_to_scratch[last_orig]))

                epilogue_plan = _synthesize_pointwise_plan(
                    syn_nodes, [pair[1] for pair in ext_out_map], gemm_shape
                )
                # Ensure output names match original FX node names.
                epilogue_plan.output_names = tuple(pair[0] for pair in ext_out_map)
                fused_names = tuple(epi_names)

            steps.append(
                _GemmStep(
                    node=n,
                    kind=kind,
                    a_name=a_name,
                    b_name=b_name,
                    bias_name=bias_name,
                    trans_a=ta,
                    trans_b=tb,
                    alpha=alpha,
                    beta=beta,
                    epilogue_plan=epilogue_plan,
                    fused_node_names=fused_names,
                )
            )
            consumed.add(n.name)
            consumed.update(epi_names)
            i = j
            continue

        # 3. Convolution (1D/2D/3D standard, depthwise, pointwise, Winograd 3x3).
        if n.op == "call_function" and _is_supported_conv(n):
            steps.append(_ConvStep(node=n))
            consumed.add(n.name)
            i += 1
            continue

        # 4. Pooling (avg_pool{1,2,3}d, max_pool2d, max_pool{1,2,3}d_with_indices).
        if n.op == "call_function" and _is_supported_pool(n):
            t = n.target
            if t == aten.avg_pool1d.default:
                pkind, pndim, rtuple = "avg", 1, False
            elif t == aten.avg_pool2d.default:
                pkind, pndim, rtuple = "avg", 2, False
            elif t == aten.avg_pool3d.default:
                pkind, pndim, rtuple = "avg", 3, False
            elif t == aten.max_pool2d.default:
                pkind, pndim, rtuple = "max", 2, False
            elif t == aten.max_pool1d_with_indices.default:
                pkind, pndim, rtuple = "max", 1, True
            elif t == aten.max_pool2d_with_indices.default:
                pkind, pndim, rtuple = "max", 2, True
            else:
                pkind, pndim, rtuple = "max", 3, True
            need_idx = rtuple and _tuple_needs_index(n, 1)
            steps.append(_PoolStep(node=n, kind=pkind, ndim=pndim, returns_tuple=rtuple, need_indices=need_idx))
            consumed.add(n.name)
            i += 1
            continue

        # 5. Softmax / LogSoftmax.
        if n.op == "call_function" and _is_supported_softmax(n):
            steps.append(_SoftmaxStep(node=n, log_softmax=(n.target == aten._log_softmax.default)))
            consumed.add(n.name)
            i += 1
            continue

        # 6. LayerNorm / GroupNorm / BatchNorm.
        if n.op == "call_function" and _is_supported_norm(n):
            if n.target == aten.native_layer_norm.default:
                nkind = "layer_norm"
            elif n.target == aten.native_group_norm.default:
                nkind = "group_norm"
            else:
                nkind = "batch_norm"
            need_stats = _tuple_needs_index(n, 1) or _tuple_needs_index(n, 2)
            steps.append(_NormStep(node=n, kind=nkind, need_stats=need_stats))
            consumed.add(n.name)
            i += 1
            continue

        # 7. Reductions (sum / mean / amax / amin).
        if n.op == "call_function" and _is_supported_reduce(n):
            if n.target in (aten.sum.default, aten.sum.dim_IntList):
                rkind = "sum"
            elif n.target in (aten.mean.default, aten.mean.dim):
                rkind = "mean"
            elif n.target == aten.amax.default:
                rkind = "amax"
            else:
                rkind = "amin"
            steps.append(_ReduceStep(node=n, kind=rkind))
            consumed.add(n.name)
            i += 1
            continue

        # 8. Constant tensor factories (zeros / ones / full / *_like).
        if n.op == "call_function" and _is_supported_fill(n):
            t = n.target
            if t in (aten.zeros.default, aten.zeros_like.default):
                fval = 0.0
            elif t in (aten.ones.default, aten.ones_like.default):
                fval = 1.0
            else:
                fval = float(n.args[1])
            _, dt, dev_str = _node_tensor_meta(n)
            dev = torch.device(dev_str) if dev_str else None
            steps.append(_FillStep(node=n, fill_val=fval, shape_or_like=n.args[0], dtype=dt, device=dev))
            consumed.add(n.name)
            i += 1
            continue

        # 9. Zero-copy view / tuple extraction metadata ops.
        if n.op == "call_function" and n.target in _METADATA_TARGETS:
            steps.append(_MetadataStep(node=n))
            consumed.add(n.name)
            i += 1
            continue

        # 10. Unsupported op -> strict error or clean eager partition.
        if strict:
            raise NotImplementedError(
                f"M1Triton strict compile mode does not support FX op {n.target!r} (node '{n.name}', op='{n.op}')."
            )
        steps.append(_EagerStep(node=n))
        consumed.add(n.name)
        i += 1

    return placeholders, steps, output_spec


def _resolve_runtime_value(val: Any, env: dict[str, Any]) -> Any:
    if isinstance(val, fx.Node):
        return env[val.name]
    if isinstance(val, tuple):
        return tuple(_resolve_runtime_value(v, env) for v in val)
    if isinstance(val, list):
        return [_resolve_runtime_value(v, env) for v in val]
    if isinstance(val, dict):
        return {k: _resolve_runtime_value(v, env) for k, v in val.items()}
    return val


def _execute_plan(
    placeholders: Sequence[str],
    steps: Sequence[_Step],
    output_spec: Any,
    runtime_args: Sequence[Any],
    gm: fx.GraphModule,
) -> Any:
    env: dict[str, Any] = dict(zip(placeholders, runtime_args))
    runner = _BatchRunner()
    try:
        for step in steps:
            if isinstance(step, _FusedPointwiseStep):
                _run_fused_pointwise(step.plan, env, runner)

            elif isinstance(step, _GemmStep):
                a_t = env[step.a_name]
                b_t = env[step.b_name]
                if step.kind == "bmm":
                    raw = _launch_sgmm_bmm(a_t, b_t, runner)
                elif step.kind == "linear" and a_t.ndim != 2:
                    lead = tuple(int(s) for s in a_t.shape[:-1])
                    k = int(a_t.shape[-1])
                    a_c = runner.ensure_contig(a_t)
                    raw_2d = _launch_sgmm_2d(
                        a_c.view(-1, k), b_t, trans_a=False, trans_b=True, runner=runner
                    )
                    raw = raw_2d.view(*lead, int(b_t.shape[0]))
                elif step.kind == "matmul":
                    if a_t.ndim == 2 and b_t.ndim == 2:
                        raw = _launch_sgmm_2d(a_t, b_t, trans_a=step.trans_a, trans_b=step.trans_b, runner=runner)
                    elif a_t.ndim == 3 and b_t.ndim == 3:
                        raw = _launch_sgmm_bmm(a_t, b_t, runner)
                    else:
                        lead = tuple(int(s) for s in a_t.shape[:-1])
                        k = int(a_t.shape[-1])
                        a_c = runner.ensure_contig(a_t)
                        raw_2d = _launch_sgmm_2d(a_c.view(-1, k), b_t, trans_a=False, trans_b=step.trans_b, runner=runner)
                        n_out = int(b_t.shape[0]) if step.trans_b else int(b_t.shape[1])
                        raw = raw_2d.view(*lead, n_out)
                else:
                    raw = _launch_sgmm_2d(a_t, b_t, trans_a=step.trans_a, trans_b=step.trans_b, runner=runner)

                if step.epilogue_plan is not None:
                    raw_sym = f"__m1_gemm_raw_{step.node.name}"
                    env[raw_sym] = raw
                    _run_fused_pointwise(step.epilogue_plan, env, runner)
                    env.pop(raw_sym, None)
                else:
                    env[step.node.name] = raw

            elif isinstance(step, _ConvStep):
                args = _resolve_runtime_value(step.node.args, env)
                out = _launch_conv(
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    bool(args[6]),
                    args[7],
                    int(args[8]),
                    runner,
                )
                env[step.node.name] = out

            elif isinstance(step, _PoolStep):
                args = _resolve_runtime_value(step.node.args, env)
                kwargs = _resolve_runtime_value(step.node.kwargs, env)
                x_t = args[0]
                k_sz = args[1]
                stride = args[2] if len(args) > 2 else kwargs.get("stride")
                padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
                if step.kind == "avg":
                    ceil_mode = bool(args[4] if len(args) > 4 else kwargs.get("ceil_mode", False))
                    count_inc = bool(args[5] if len(args) > 5 else kwargs.get("count_include_pad", True))
                    dilation = 1
                else:
                    dilation = args[4] if len(args) > 4 else kwargs.get("dilation", 1)
                    ceil_mode = bool(args[5] if len(args) > 5 else kwargs.get("ceil_mode", False))
                    count_inc = True
                out_t, idx_t = _launch_pool(
                    x_t,
                    kind=step.kind,
                    ndim=step.ndim,
                    kernel_size=k_sz,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    count_include_pad=count_inc,
                    ceil_mode=ceil_mode,
                    need_indices=step.need_indices,
                    runner=runner,
                )
                env[step.node.name] = (out_t, idx_t) if step.returns_tuple else out_t

            elif isinstance(step, _SoftmaxStep):
                args = _resolve_runtime_value(step.node.args, env)
                x_t = args[0]
                dim = int(args[1])
                h2f = bool(args[2]) if len(args) > 2 else False
                env[step.node.name] = _launch_softmax(
                    x_t, dim, log_softmax=step.log_softmax, half_to_float=h2f, runner=runner
                )

            elif isinstance(step, _NormStep):
                args = _resolve_runtime_value(step.node.args, env)
                if step.kind == "layer_norm":
                    out_tuple = _launch_layer_norm(
                        args[0], args[1], args[2], args[3], float(args[4]), step.need_stats, runner
                    )
                    env[step.node.name] = out_tuple
                elif step.kind == "group_norm":
                    out_tuple = _launch_group_norm(
                        args[0], args[1], args[2], int(args[6]), float(args[7]), step.need_stats, runner
                    )
                    env[step.node.name] = out_tuple
                else:
                    t = step.node.target
                    if t == aten._native_batch_norm_legit_no_training.default:
                        out_tuple = _launch_batch_norm(
                            args[0], args[1], args[2], args[3], args[4], False, float(args[6]), step.need_stats, runner
                        )
                    elif t == aten._native_batch_norm_legit.no_stats:
                        out_tuple = _launch_batch_norm(
                            args[0], args[1], args[2], None, None, bool(args[3]), float(args[5]), step.need_stats, runner
                        )
                    else:
                        # native_batch_norm / _native_batch_norm_legit: (x, w, b, r_mean, r_var, training, mom, eps)
                        out_tuple = _launch_batch_norm(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            bool(args[5]),
                            float(args[7]),
                            step.need_stats,
                            runner,
                        )
                    env[step.node.name] = out_tuple

            elif isinstance(step, _ReduceStep):
                args = _resolve_runtime_value(step.node.args, env)
                kwargs = _resolve_runtime_value(step.node.kwargs, env)
                x_t = args[0]
                t = step.node.target
                if t in (aten.sum.default, aten.mean.default):
                    dims = None
                    keepdim = False
                    out_dt = kwargs.get("dtype")
                elif t in (aten.sum.dim_IntList, aten.mean.dim):
                    dims = args[1] if len(args) > 1 else kwargs.get("dim")
                    keepdim = bool(args[2] if len(args) > 2 else kwargs.get("keepdim", False))
                    out_dt = kwargs.get("dtype")
                else:
                    dims = args[1] if len(args) > 1 else kwargs.get("dim", [])
                    keepdim = bool(args[2] if len(args) > 2 else kwargs.get("keepdim", False))
                    out_dt = None
                env[step.node.name] = _launch_reduce(
                    x_t, kind=step.kind, dims=dims, keepdim=keepdim, out_dtype=out_dt, runner=runner
                )

            elif isinstance(step, _FillStep):
                if isinstance(step.shape_or_like, fx.Node):
                    like_t = env[step.shape_or_like.name]
                    shape = tuple(int(s) for s in like_t.shape)
                    dt = step.dtype or like_t.dtype
                    dev = step.device or like_t.device
                else:
                    shape = tuple(int(s) for s in _resolve_runtime_value(step.shape_or_like, env))
                    dt = step.dtype or torch.float32
                    dev = step.device or torch.device("mps")
                out_t = torch.empty(shape, dtype=dt, device=dev)
                n = out_t.numel()
                if n > 0:
                    blk = 1024 if n >= 1024 else 256
                    runner.begin_m1(dev.type == "mps")
                    fill_arg: Any = (
                        bool(step.fill_val)
                        if dt == torch.bool
                        else (int(step.fill_val) if dt in (torch.int32, torch.int64) else float(step.fill_val))
                    )
                    _fill_const_kernel[(_cdiv(n, blk),)](
                        out_t, fill_arg, n, BLOCK_SIZE=blk, wait=False, sync_mps=False
                    )
                env[step.node.name] = out_t

            elif isinstance(step, _MetadataStep):
                args = _resolve_runtime_value(step.node.args, env)
                kwargs = _resolve_runtime_value(step.node.kwargs, env)
                t = step.node.target
                if t in (aten.reshape.default, aten.contiguous.default):
                    src = args[0]
                    if isinstance(src, torch.Tensor) and not src.is_contiguous():
                        env[step.node.name] = runner.run_eager(t, args, kwargs)
                    else:
                        env[step.node.name] = t(*args, **kwargs)
                else:
                    env[step.node.name] = t(*args, **kwargs)

            elif isinstance(step, _EagerStep):
                n = step.node
                args = _resolve_runtime_value(n.args, env)
                kwargs = _resolve_runtime_value(n.kwargs, env)
                if n.op == "call_function":
                    env[n.name] = runner.run_eager(n.target, args, kwargs)
                elif n.op == "call_method":
                    fn = getattr(args[0], str(n.target))
                    env[n.name] = runner.run_eager(fn, args[1:], kwargs)
                elif n.op == "call_module":
                    mod = gm.get_submodule(str(n.target))
                    env[n.name] = runner.run_eager(mod, args, kwargs)
                elif n.op == "get_attr":
                    env[n.name] = getattr(gm, str(n.target))
                else:
                    raise NotImplementedError(f"Unsupported FX node op {n.op!r} for node {n.name!r}")

        runner.flush()
    except Exception:
        runner.abort()
        raise

    return _resolve_runtime_value(output_spec, env)


# ============================================================================
# Public Compiler Entrypoints & TorchDynamo Backend Registration
# ============================================================================


def compile_fx_graph(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    strict: bool | None = None,
) -> Callable[..., Any]:
    """Compile a captured ATen `torch.fx.GraphModule` into a fused M1Triton callable."""
    use_strict = _env_strict() if strict is None else bool(strict)

    # Populate shape/dtype metadata if missing (e.g. standalone FX GraphModule without AOTAutograd).
    needs_prop = any(
        n.op == "placeholder" and _node_val(n) is None and n.meta.get("tensor_meta") is None
        for n in gm.graph.nodes
    )
    if needs_prop and example_inputs:
        ShapeProp(gm).propagate(*example_inputs)

    placeholders, steps, output_spec = _build_fused_plan(gm, strict=use_strict)

    def compiled_callable(*runtime_args: Any) -> Any:
        return _execute_plan(placeholders, steps, output_spec, runtime_args, gm)

    setattr(compiled_callable, "m1triton_steps", steps)
    return compiled_callable


def _make_aot_compiler(*, strict: bool) -> Callable[[fx.GraphModule, Sequence[Any]], Callable[..., Any]]:
    def _fw_compiler(gm: fx.GraphModule, example_inputs: Sequence[Any]) -> Callable[..., Any]:
        return make_boxed_func(compile_fx_graph(gm, example_inputs, strict=strict))

    def _bw_compiler(gm: fx.GraphModule, example_inputs: Sequence[Any]) -> Callable[..., Any]:
        return make_boxed_func(compile_fx_graph(gm, example_inputs, strict=strict))

    return aot_autograd(fw_compiler=_fw_compiler, bw_compiler=_bw_compiler)


def m1triton_backend(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """TorchDynamo backend entry point registered under the name `'m1triton'`."""
    opts = dict(options or {})
    opts.update(kwargs)
    if "strict" in opts and opts["strict"] is not None:
        strict = bool(opts["strict"])
    else:
        strict = _env_strict()
    aot_backend = _make_aot_compiler(strict=strict)
    return aot_backend(gm, example_inputs)


def _register_backend() -> None:
    if "m1triton" not in _dynamo.list_backends(exclude_tags=()):
        _dynamo.register_backend(name="m1triton")(m1triton_backend)


_register_backend()


def compile(
    model: Any,
    *,
    strict: bool | None = None,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper around `torch.compile(model, backend='m1triton')`."""
    _register_backend()
    opts = dict(kwargs.pop("options", None) or {})
    if strict is not None:
        opts["strict"] = bool(strict)
    return torch.compile(
        model,
        backend="m1triton",
        fullgraph=fullgraph,
        dynamic=dynamic,
        options=opts if opts else None,
        **kwargs,
    )


__all__ = [
    "clear_kernel_cache",
    "compile",
    "compile_fx_graph",
    "get_cache_stats",
    "m1triton_backend",
]
