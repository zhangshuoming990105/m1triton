"""Reusable M1Triton 1D/2D/3D standard and pointwise implicit-GEMM convolution (KernelBench L1 M5 standard conv)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, out_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    W_in: tl.constexpr, W_out: tl.constexpr,
    K_W: tl.constexpr, S_W: tl.constexpr, P_W: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, N, C_in, C_out, W_in, W_out,
        ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,), groups=GROUPS,
    )


@m1triton.jit
def conv1d_bias_kernel(
    x_ptr, w_ptr, out_ptr, b_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    W_in: tl.constexpr, W_out: tl.constexpr,
    K_W: tl.constexpr, S_W: tl.constexpr, P_W: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, W_in, W_out,
        ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,), groups=GROUPS,
        has_bias=True,
    )


@m1triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
        groups=GROUPS,
    )


@m1triton.jit
def conv2d_bias_kernel(
    x_ptr, w_ptr, out_ptr, b_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
        groups=GROUPS, has_bias=True,
    )


@m1triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    S_D: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_D: tl.constexpr, P_H: tl.constexpr, P_W: tl.constexpr,
    D_D: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
        dilation=(D_D, D_H, D_W), groups=GROUPS,
    )


@m1triton.jit
def conv3d_bias_kernel(
    x_ptr, w_ptr, out_ptr, b_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    S_D: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_D: tl.constexpr, P_H: tl.constexpr, P_W: tl.constexpr,
    D_D: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
):
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
        dilation=(D_D, D_H, D_W), groups=GROUPS, has_bias=True,
    )


@m1triton.jit
def winograd_filter_2x3_kernel(
    w_ptr,
    u_ptr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    C_IN_PAD: tl.constexpr,
    C_OUT_PAD: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C_OUT_PAD * C_IN_PAD
    valid = idx < total
    co = idx // C_IN_PAD
    ci = idx - co * C_IN_PAD
    in_bounds = valid & (co < C_out) & (ci < C_in)
    if TRANSPOSED == 0:
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
    else:
        base = (ci * C_out + co) * 9
        g00 = tl.load(w_ptr + (base + 8), mask=in_bounds, other=0.0)
        g01 = tl.load(w_ptr + (base + 7), mask=in_bounds, other=0.0)
        g02 = tl.load(w_ptr + (base + 6), mask=in_bounds, other=0.0)
        g10 = tl.load(w_ptr + (base + 5), mask=in_bounds, other=0.0)
        g11 = tl.load(w_ptr + (base + 4), mask=in_bounds, other=0.0)
        g12 = tl.load(w_ptr + (base + 3), mask=in_bounds, other=0.0)
        g20 = tl.load(w_ptr + (base + 2), mask=in_bounds, other=0.0)
        g21 = tl.load(w_ptr + (base + 1), mask=in_bounds, other=0.0)
        g22 = tl.load(w_ptr + (base + 0), mask=in_bounds, other=0.0)

    s0 = g00 + g20
    s1 = g01 + g21
    s2 = g02 + g22
    t00 = g00
    t01 = g01
    t02 = g02
    t10 = 0.5 * (s0 + g10)
    t11 = 0.5 * (s1 + g11)
    t12 = 0.5 * (s2 + g12)
    t20 = 0.5 * (s0 - g10)
    t21 = 0.5 * (s1 - g11)
    t22 = 0.5 * (s2 - g12)
    t30 = g20
    t31 = g21
    t32 = g22

    r0 = t00 + t02
    u0 = t00
    u1 = 0.5 * (r0 + t01)
    u2 = 0.5 * (r0 - t01)
    u3 = t02

    r1 = t10 + t12
    u4 = t10
    u5 = 0.5 * (r1 + t11)
    u6 = 0.5 * (r1 - t11)
    u7 = t12

    r2 = t20 + t22
    u8 = t20
    u9 = 0.5 * (r2 + t21)
    u10 = 0.5 * (r2 - t21)
    u11 = t22

    r3 = t30 + t32
    u12 = t30
    u13 = 0.5 * (r3 + t31)
    u14 = 0.5 * (r3 - t31)
    u15 = t32

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
def winograd_filter_4x3_rm_kernel(
    w_ptr,
    u_ptr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C_out * C_in
    valid = idx < total
    co = idx // C_in
    ci = idx - co * C_in
    if TRANSPOSED == 0:
        base = idx * 9
        g00 = tl.load(w_ptr + (base + 0), mask=valid, other=0.0)
        g01 = tl.load(w_ptr + (base + 1), mask=valid, other=0.0)
        g02 = tl.load(w_ptr + (base + 2), mask=valid, other=0.0)
        g10 = tl.load(w_ptr + (base + 3), mask=valid, other=0.0)
        g11 = tl.load(w_ptr + (base + 4), mask=valid, other=0.0)
        g12 = tl.load(w_ptr + (base + 5), mask=valid, other=0.0)
        g20 = tl.load(w_ptr + (base + 6), mask=valid, other=0.0)
        g21 = tl.load(w_ptr + (base + 7), mask=valid, other=0.0)
        g22 = tl.load(w_ptr + (base + 8), mask=valid, other=0.0)
    else:
        base = (ci * C_out + co) * 9
        g00 = tl.load(w_ptr + (base + 8), mask=valid, other=0.0)
        g01 = tl.load(w_ptr + (base + 7), mask=valid, other=0.0)
        g02 = tl.load(w_ptr + (base + 6), mask=valid, other=0.0)
        g10 = tl.load(w_ptr + (base + 5), mask=valid, other=0.0)
        g11 = tl.load(w_ptr + (base + 4), mask=valid, other=0.0)
        g12 = tl.load(w_ptr + (base + 3), mask=valid, other=0.0)
        g20 = tl.load(w_ptr + (base + 2), mask=valid, other=0.0)
        g21 = tl.load(w_ptr + (base + 1), mask=valid, other=0.0)
        g22 = tl.load(w_ptr + (base + 0), mask=valid, other=0.0)

    c_n16 = -0.1666666716337204
    c_124 = 0.0416666679084301
    c_112 = 0.0833333358168602
    c_16 = 0.1666666716337204

    t00 = 0.25 * g00
    s0 = g00 + g20
    t10 = c_n16 * (s0 + g10)
    t20 = c_n16 * (s0 - g10)
    p0 = c_124 * g00 + c_16 * g20
    q0 = c_112 * g10
    t30 = p0 + q0
    t40 = p0 - q0
    t50 = g20

    t01 = 0.25 * g01
    s1 = g01 + g21
    t11 = c_n16 * (s1 + g11)
    t21 = c_n16 * (s1 - g11)
    p1 = c_124 * g01 + c_16 * g21
    q1 = c_112 * g11
    t31 = p1 + q1
    t41 = p1 - q1
    t51 = g21

    t02 = 0.25 * g02
    s2 = g02 + g22
    t12 = c_n16 * (s2 + g12)
    t22 = c_n16 * (s2 - g12)
    p2 = c_124 * g02 + c_16 * g22
    q2 = c_112 * g12
    t32 = p2 + q2
    t42 = p2 - q2
    t52 = g22

    alpha_stride = C_out * C_in
    dst = idx

    sr = t00 + t02
    pr = c_124 * t00 + c_16 * t02
    qr = c_112 * t01
    tl.store(u_ptr + (0 * alpha_stride + dst), 0.25 * t00, mask=valid)
    tl.store(u_ptr + (1 * alpha_stride + dst), c_n16 * (sr + t01), mask=valid)
    tl.store(u_ptr + (2 * alpha_stride + dst), c_n16 * (sr - t01), mask=valid)
    tl.store(u_ptr + (3 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (4 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (5 * alpha_stride + dst), t02, mask=valid)

    sr = t10 + t12
    pr = c_124 * t10 + c_16 * t12
    qr = c_112 * t11
    tl.store(u_ptr + (6 * alpha_stride + dst), 0.25 * t10, mask=valid)
    tl.store(u_ptr + (7 * alpha_stride + dst), c_n16 * (sr + t11), mask=valid)
    tl.store(u_ptr + (8 * alpha_stride + dst), c_n16 * (sr - t11), mask=valid)
    tl.store(u_ptr + (9 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (10 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (11 * alpha_stride + dst), t12, mask=valid)

    sr = t20 + t22
    pr = c_124 * t20 + c_16 * t22
    qr = c_112 * t21
    tl.store(u_ptr + (12 * alpha_stride + dst), 0.25 * t20, mask=valid)
    tl.store(u_ptr + (13 * alpha_stride + dst), c_n16 * (sr + t21), mask=valid)
    tl.store(u_ptr + (14 * alpha_stride + dst), c_n16 * (sr - t21), mask=valid)
    tl.store(u_ptr + (15 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (16 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (17 * alpha_stride + dst), t22, mask=valid)

    sr = t30 + t32
    pr = c_124 * t30 + c_16 * t32
    qr = c_112 * t31
    tl.store(u_ptr + (18 * alpha_stride + dst), 0.25 * t30, mask=valid)
    tl.store(u_ptr + (19 * alpha_stride + dst), c_n16 * (sr + t31), mask=valid)
    tl.store(u_ptr + (20 * alpha_stride + dst), c_n16 * (sr - t31), mask=valid)
    tl.store(u_ptr + (21 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (22 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (23 * alpha_stride + dst), t32, mask=valid)

    sr = t40 + t42
    pr = c_124 * t40 + c_16 * t42
    qr = c_112 * t41
    tl.store(u_ptr + (24 * alpha_stride + dst), 0.25 * t40, mask=valid)
    tl.store(u_ptr + (25 * alpha_stride + dst), c_n16 * (sr + t41), mask=valid)
    tl.store(u_ptr + (26 * alpha_stride + dst), c_n16 * (sr - t41), mask=valid)
    tl.store(u_ptr + (27 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (28 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (29 * alpha_stride + dst), t42, mask=valid)

    sr = t50 + t52
    pr = c_124 * t50 + c_16 * t52
    qr = c_112 * t51
    tl.store(u_ptr + (30 * alpha_stride + dst), 0.25 * t50, mask=valid)
    tl.store(u_ptr + (31 * alpha_stride + dst), c_n16 * (sr + t51), mask=valid)
    tl.store(u_ptr + (32 * alpha_stride + dst), c_n16 * (sr - t51), mask=valid)
    tl.store(u_ptr + (33 * alpha_stride + dst), pr + qr, mask=valid)
    tl.store(u_ptr + (34 * alpha_stride + dst), pr - qr, mask=valid)
    tl.store(u_ptr + (35 * alpha_stride + dst), t52, mask=valid)


@m1triton.jit
def wino4x3_input_kernel(
    x_ptr, dummy_ptr, v_ptr,
    row_start, chunk_rows, p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr,
    WINOGRAD: tl.constexpr = 10,
):
    tlx.implicit_gemm_conv(
        x_ptr, dummy_ptr, v_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def wino4x3_bmm_kernel(
    v_ptr, u_ptr, m_ptr,
    p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    WINOGRAD: tl.constexpr = 11,
):
    tlx.implicit_gemm_conv(
        v_ptr, u_ptr, m_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def wino4x3_output_kernel(
    m_ptr, dummy_ptr, out_ptr,
    row_start, chunk_rows, p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    WINOGRAD: tl.constexpr = 12,
):
    tlx.implicit_gemm_conv(
        m_ptr, dummy_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def wino4x3_output_bias_kernel(
    m_ptr, dummy_ptr, out_ptr, b_ptr,
    row_start, chunk_rows, p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    WINOGRAD: tl.constexpr = 12,
):
    tlx.implicit_gemm_conv(
        m_ptr, dummy_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        groups=1, has_bias=True,
    )


def select_wino4x3_bmm_tile(c_in: int, c_out: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, NUM_SG) schedule for 36-batch Winograd F(4x4, 3x3) SGMM."""
    bk = 32 if (c_in >= 32 and c_in % 32 == 0) else 16
    if c_out >= 128:
        return 64, 128, bk, 8
    if c_out >= 64:
        return 64, 64, 16, 4
    if c_out <= 16:
        return 16, 64, bk, 4
    return 32, 64, bk, 4


_WINO_WS_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_wino_workspace(device: torch.device, v_elems: int, m_elems: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index or 0)
    cached = _WINO_WS_CACHE.get(key)
    if cached is not None and cached[0].numel() >= v_elems and cached[1].numel() >= m_elems:
        return cached[0], cached[1]
    v_alloc = max(v_elems, cached[0].numel() if cached is not None else 0)
    m_alloc = max(m_elems, cached[1].numel() if cached is not None else 0)
    # Keep resident workspace bounded well inside 32 GB UMA.
    if (v_alloc + m_alloc) * 4 <= 256 * 1024 * 1024:
        bufs = (
            torch.empty((v_alloc,), dtype=torch.float32, device=device),
            torch.empty((m_alloc,), dtype=torch.float32, device=device),
        )
        _WINO_WS_CACHE[key] = bufs
        return bufs
    return (
        torch.empty((v_elems,), dtype=torch.float32, device=device),
        torch.empty((m_elems,), dtype=torch.float32, device=device),
    )


def conv2d_winograd_4x3_staged(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    padding: tuple[int, int] = (0, 0),
    transposed: bool = False,
    slc_budget_bytes: int = 192 * 1024 * 1024,
    bmm_tile: tuple[int, int, int, int] | None = None,
    cached_u: torch.Tensor | None = None,
) -> torch.Tensor:
    """3-stage chunked Winograd F(4x4, 3x3) with coalesced input, 36-batch SGMM, and float4 output."""
    x_c = x.contiguous()
    w_c = weight.detach().contiguous()
    n, c_in, h_in, w_in = (int(v) for v in x_c.shape)
    c_out = int(w_c.shape[1]) if transposed else int(w_c.shape[0])
    ph, pw = int(padding[0]), int(padding[1])
    h_out = conv_output_size(h_in, 3, 1, ph, 1)
    w_out = conv_output_size(w_in, 3, 1, pw, 1)
    out = torch.empty((n, c_out, h_out, w_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    if cached_u is not None:
        u = cached_u
        sync_first = True
    else:
        u = torch.empty((36, c_out, c_in), dtype=torch.float32, device=x_c.device)
        f_grid = (m1triton.cdiv(c_out * c_in, 256),)
        winograd_filter_4x3_rm_kernel[f_grid](
            w_c, u, C_in=c_in, C_out=c_out, TRANSPOSED=1 if transposed else 0, BLOCK_SIZE=256,
            sync_mps=True, wait=False,
        )
        sync_first = False

    bm, bn, bk, nsg = bmm_tile if bmm_tile is not None else select_wino4x3_bmm_tile(c_in, c_out)

    def _pad_pld(p_raw: int) -> int:
        p = m1triton.cdiv(p_raw, bn) * bn
        if (p % 256) == 0:
            p += bn
        return p

    th = m1triton.cdiv(h_out, 4)
    tw = m1triton.cdiv(w_out, 4)
    total_rows = n * th
    bytes_per_row = 36 * (c_in + c_out) * tw * 4
    max_rows = max(1, min(total_rows, slc_budget_bytes // max(bytes_per_row, 1)))
    p_alloc = _pad_pld(max_rows * tw)
    v_buf, m_buf = _get_wino_workspace(x_c.device, 36 * (c_in * p_alloc + 64), 36 * (c_out * p_alloc + 64))

    b_c = bias.detach().contiguous() if bias is not None else None
    in_gx = m1triton.cdiv(tw, 64)

    for r0 in range(0, total_rows, max_rows):
        rc = min(max_rows, total_rows - r0)
        p_cur = _pad_pld(rc * tw)
        is_last = (r0 + rc >= total_rows)
        in_sync = sync_first and (r0 == 0)

        wino4x3_input_kernel[(in_gx, rc, c_in)](
            x_c, u, v_buf, r0, rc, p_cur,
            N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
            P_H=ph, P_W=pw, WINOGRAD=10, sync_mps=in_sync, wait=False,
        )
        wino4x3_bmm_kernel[(m1triton.cdiv(p_cur, bn), m1triton.cdiv(c_out, bm), 36)](
            v_buf, u, m_buf, p_cur,
            N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=nsg, WINOGRAD=11,
            sync_mps=False, wait=False,
        )
        if b_c is not None:
            wino4x3_output_bias_kernel[(in_gx, rc, c_out)](
                m_buf, u, out, b_c, r0, rc, p_cur,
                N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                WINOGRAD=12, sync_mps=False, wait=is_last,
            )
        else:
            wino4x3_output_kernel[(in_gx, rc, c_out)](
                m_buf, u, out, r0, rc, p_cur,
                N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                WINOGRAD=12, sync_mps=False, wait=is_last,
            )
    return out


@m1triton.jit
def conv2d_wino2x3_kernel(
    x_ptr, u_ptr, out_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_TH: tl.constexpr, TILE_TW: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
    WINOGRAD: tl.constexpr = 1,
    WINO_PAD: tl.constexpr = 1,
):
    tlx.implicit_gemm_conv(
        x_ptr, u_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def conv2d_wino2x3_bias_kernel(
    x_ptr, u_ptr, out_ptr, b_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_TH: tl.constexpr, TILE_TW: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
    WINOGRAD: tl.constexpr = 1,
    WINO_PAD: tl.constexpr = 1,
):
    tlx.implicit_gemm_conv(
        x_ptr, u_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1),
        groups=1, has_bias=True,
    )


def select_winograd_tile(c_in: int, c_out: int) -> tuple[int, int, int, int, int, int, int, int]:
    """(MODE, BLOCK_M, BLOCK_N, BLOCK_K, TILE_TH, TILE_TW, NUM_SG, WINO_PAD) for fused Winograd F(2x2, 3x3)."""
    bk = 16 if (c_in >= 16 and c_in % 16 == 0) else 8
    if c_in <= 16 and c_out >= 64:
        return 1, 32, 16, bk, 1, 16, 8, 1
    return 1, 32, 32, bk, 2, 16, 16, 1


def conv2d_winograd_3x3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    padding: tuple[int, int] = (0, 0),
    transposed: bool = False,
    wino_tile: tuple[int, int, int, int, int, int, int, int] | None = None,
    cached_u: torch.Tensor | None = None,
    use_staged_4x3: bool | None = None,
) -> torch.Tensor:
    """Dispatches Winograd F(2x2/4x4, 3x3) Metal kernels for 2D 3x3 s=1 d=1 g=1 conv and conv_transpose."""
    x_c = x.contiguous()
    w_c = weight.detach().contiguous()
    n, c_in, h_in, w_in = (int(v) for v in x_c.shape)
    c_out = int(w_c.shape[1]) if transposed else int(w_c.shape[0])
    ph, pw = int(padding[0]), int(padding[1])
    h_out = conv_output_size(h_in, 3, 1, ph, 1)
    w_out = conv_output_size(w_in, 3, 1, pw, 1)
    if use_staged_4x3 is None:
        use_staged_4x3 = wino_tile is None and c_in >= 64 and c_out >= 128 and h_out * w_out >= 65536
    if use_staged_4x3:
        return conv2d_winograd_4x3_staged(
            x_c, w_c, bias, padding=(ph, pw), transposed=transposed, cached_u=cached_u,
        )
    out = torch.empty((n, c_out, h_out, w_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    mode, bm, bn, bk, tth, ttw, nsg, wpad = wino_tile if wino_tile is not None else select_winograd_tile(c_in, c_out)
    c_in_pad = m1triton.cdiv(c_in, bk) * bk
    c_out_pad = m1triton.cdiv(c_out, bm) * bm
    if cached_u is not None:
        u = cached_u
        sync_first = True
    else:
        u = torch.empty((16, c_in_pad // 8, c_out_pad // 8, 8, 8), dtype=torch.float32, device=x_c.device)
        f_grid = (m1triton.cdiv(c_out_pad * c_in_pad, 256),)
        winograd_filter_2x3_kernel[f_grid](
            w_c,
            u,
            C_in=c_in,
            C_out=c_out,
            C_IN_PAD=c_in_pad,
            C_OUT_PAD=c_out_pad,
            TRANSPOSED=1 if transposed else 0,
            BLOCK_SIZE=256,
            sync_mps=True,
            wait=False,
        )
        sync_first = False

    th = m1triton.cdiv(h_out, 2)
    tw = m1triton.cdiv(w_out, 2)
    grid = (m1triton.cdiv(th, tth) * m1triton.cdiv(tw, ttw), m1triton.cdiv(c_out, bm), n)
    if bias is not None:
        conv2d_wino2x3_bias_kernel[grid](
            x_c, u, out, bias.detach().contiguous(),
            N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
            P_H=ph, P_W=pw, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TILE_TH=tth, TILE_TW=ttw,
            NUM_SG=nsg, WINOGRAD=mode, WINO_PAD=wpad, sync_mps=sync_first, wait=True,
        )
    else:
        conv2d_wino2x3_kernel[grid](
            x_c, u, out,
            N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
            P_H=ph, P_W=pw, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TILE_TH=tth, TILE_TW=ttw,
            NUM_SG=nsg, WINOGRAD=mode, WINO_PAD=wpad, sync_mps=sync_first, wait=True,
        )
    return out


def prepare_winograd_filter(
    weight: torch.Tensor, *, transposed: bool, staged_4x3: bool
) -> torch.Tensor:
    """Pre-transforms 3x3 weight into Winograd filter domain via M1Triton kernel."""
    w_c = weight.detach().contiguous()
    c_in = int(w_c.shape[0]) if transposed else int(w_c.shape[1])
    c_out = int(w_c.shape[1]) if transposed else int(w_c.shape[0])
    if staged_4x3:
        u = torch.empty((36, c_out, c_in), dtype=torch.float32, device=w_c.device)
        winograd_filter_4x3_rm_kernel[(m1triton.cdiv(c_out * c_in, 256),)](
            w_c, u, C_in=c_in, C_out=c_out, TRANSPOSED=1 if transposed else 0, BLOCK_SIZE=256,
            sync_mps=True, wait=True,
        )
        return u
    _, bm, _, bk, _, _, _, _ = select_winograd_tile(c_in, c_out)
    c_in_pad = m1triton.cdiv(c_in, bk) * bk
    c_out_pad = m1triton.cdiv(c_out, bm) * bm
    u = torch.empty((16, c_in_pad // 8, c_out_pad // 8, 8, 8), dtype=torch.float32, device=w_c.device)
    winograd_filter_2x3_kernel[(m1triton.cdiv(c_out_pad * c_in_pad, 256),)](
        w_c, u, C_in=c_in, C_out=c_out, C_IN_PAD=c_in_pad, C_OUT_PAD=c_out_pad,
        TRANSPOSED=1 if transposed else 0, BLOCK_SIZE=256, sync_mps=True, wait=True,
    )
    return u


_KERNELS = {
    (1, False): conv1d_kernel,
    (1, True): conv1d_bias_kernel,
    (2, False): conv2d_kernel,
    (2, True): conv2d_bias_kernel,
    (3, False): conv3d_kernel,
    (3, True): conv3d_bias_kernel,
}

_SHAPE_NAMES = {
    1: ("W_in", "W_out"),
    2: ("H_in", "W_in", "H_out", "W_out"),
    3: ("D_in", "H_in", "W_in", "D_out", "H_out", "W_out"),
}

_KEY_PREFIX = {1: ("W",), 2: ("H", "W"), 3: ("D", "H", "W")}


def as_tuple(v: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(v, int):
        return (int(v),) * ndim
    vals = tuple(int(x) for x in v)
    if len(vals) == 1:
        return vals * ndim
    if len(vals) != ndim:
        raise ValueError(f"expected {ndim} convolution parameters, got {vals!r}")
    return vals


def conv_output_size(size: int, k: int, s: int, p: int, d: int) -> int:
    """PyTorch convolution output extent: floor((in + 2p - d*(k-1) - 1) / s) + 1."""
    out = (size + 2 * p - d * (k - 1) - 1) // s + 1
    if out <= 0:
        raise ValueError("convolution kernel larger than padded input")
    return out


def select_tile(
    c_in_per_group: int,
    c_out_per_group: int,
    k_spatial: int,
    pointwise: bool,
    ndim: int = 2,
    stride_w: int = 1,
) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, NUM_SG) schedule tuned for Apple M1 Max 32 KB SRAM occupancy."""
    k_total = c_in_per_group * k_spatial
    if pointwise:
        bm = 16 if c_out_per_group <= 16 else (32 if c_out_per_group <= 32 else 64)
        bk = 32 if (k_total > 0 and k_total % 32 == 0) else 16
        return bm, 64, bk, 4
    if ndim == 1 and stride_w > 1 and c_out_per_group >= 128 and (c_out_per_group % 128 == 0) and k_total >= 192 and (k_total % 16 == 0):
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


def conv_nd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    ndim: int,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    tile: tuple[int, int, int, int] | None = None,
    cached_wino_u: torch.Tensor | None = None,
    use_staged_4x3: bool | None = None,
) -> torch.Tensor:
    """Dispatches a single M1Triton implicit-GEMM Metal convolution over an (N, C, *spatial) tensor."""
    if x.dim() != ndim + 2 or weight.dim() != ndim + 2:
        raise ValueError(f"conv{ndim}d expects {ndim + 2}D input and weight, got {tuple(x.shape)}, {tuple(weight.shape)}")
    x_c = x.contiguous()
    w_c = weight.detach().contiguous()
    n, c_in = int(x_c.shape[0]), int(x_c.shape[1])
    c_out = int(w_c.shape[0])
    if c_in % groups != 0 or c_out % groups != 0 or int(w_c.shape[1]) * groups != c_in:
        raise ValueError("channel counts are inconsistent with groups")
    k = tuple(int(v) for v in w_c.shape[2:])
    s = as_tuple(stride, ndim)
    p = as_tuple(padding, ndim)
    d = as_tuple(dilation, ndim)
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [conv_output_size(spatial_in[i], k[i], s[i], p[i], d[i]) for i in range(ndim)]
    if (
        tile is None
        and ndim == 2
        and k == (3, 3)
        and s == (1, 1)
        and d == (1, 1)
        and groups == 1
        and x_c.dtype == torch.float32
        and w_c.dtype == torch.float32
        and (bias is None or bias.dtype == torch.float32)
        and c_in >= 8
        and c_out >= 16
        and spatial_out[0] * spatial_out[1] >= 256
    ):
        return conv2d_winograd_3x3(
            x_c, w_c, bias, padding=(p[0], p[1]), transposed=False,
            cached_u=cached_wino_u, use_staged_4x3=use_staged_4x3,
        )
    out = torch.empty((n, c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out
    pixels = n * math.prod(spatial_out)
    if pixels >= 2**32:
        raise ValueError("N * prod(spatial_out) must fit in 32 bits")
    pointwise = all(v == 1 for v in k) and all(v == 1 for v in s) and all(v == 0 for v in p)
    c_out_g = c_out // groups
    bm, bn, bk, num_sg = (
        tile
        if tile is not None
        else select_tile(c_in // groups, c_out_g, math.prod(k), pointwise, ndim=ndim, stride_w=s[-1])
    )
    grid = (m1triton.cdiv(pixels, bn), m1triton.cdiv(c_out_g, bm), groups)

    names = _KEY_PREFIX[ndim]
    consts: dict[str, int] = dict(zip(_SHAPE_NAMES[ndim], (*spatial_in, *spatial_out)))
    for i, nm in enumerate(names):
        consts[f"K_{nm}"] = k[i]
        consts[f"S_{nm}"] = s[i]
        consts[f"P_{nm}"] = p[i]
        consts[f"D_{nm}"] = d[i]
    kernel = _KERNELS[(ndim, bias is not None)]
    args: list[object] = [x_c, w_c, out]
    if bias is not None:
        args.append(bias.detach().contiguous())
    kernel[grid](
        *args, N=n, C_in=c_in, C_out=c_out, GROUPS=int(groups),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=num_sg, wait=True, **consts,
    )
    return out


class ConvStdModule(nn.Module):
    """Parameter-only convolution module: holds weight/bias as nn.Parameter and dispatches conv_nd."""

    _NDIM = 2
    _REF_ATTR = "conv2d"

    def _setup(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        nd = self._NDIM
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = as_tuple(kernel_size, nd)
        self.stride = as_tuple(stride, nd)
        self.padding = as_tuple(padding, nd)
        self.dilation = as_tuple(dilation, nd)
        self.groups = int(groups)
        fan_in = (self.in_channels // self.groups) * math.prod(self.kernel_size)
        bound = 1.0 / math.sqrt(max(fan_in, 1))
        w = torch.empty(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        self.weight = nn.Parameter(w.uniform_(-bound, bound))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels).uniform_(-bound, bound))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ConvStdModule":
        conv = getattr(ref_model, cls._REF_ATTR)
        if isinstance(conv.padding, str):
            raise NotImplementedError("string padding modes are not supported")
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model._setup(
            conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
            conv.dilation, conv.groups, conv.bias is not None,
        )
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        conv = getattr(ref_model, self._REF_ATTR)
        with torch.no_grad():
            self.weight.copy_(conv.weight.detach().to(self.weight.device))
            if self.bias is not None and conv.bias is not None:
                self.bias.copy_(conv.bias.detach().to(self.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_nd(
            x,
            self.weight.detach(),
            self.bias,
            ndim=self._NDIM,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
