"""Reusable M1Triton depthwise and pointwise convolution kernels (KernelBench L1 #82-#86)."""

from __future__ import annotations

from typing import Sequence

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

_HALO_BUDGET_BYTES = 16 * 1024


@m1triton.jit
def conv_dw2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
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
):
    tlx.implicit_gemm_conv(
        x_ptr,
        w_ptr,
        out_ptr,
        N,
        C_in,
        C_out,
        H_in,
        W_in,
        H_out,
        W_out,
        ndim=2,
        kernel_size=(K_H, K_W),
        stride=(S_H, S_W),
        padding=(P_H, P_W),
        dilation=(D_H, D_W),
        is_depthwise=True,
        has_bias=False,
    )


@m1triton.jit
def conv_dw2d_bias_kernel(
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
):
    tlx.implicit_gemm_conv(
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
        ndim=2,
        kernel_size=(K_H, K_W),
        stride=(S_H, S_W),
        padding=(P_H, P_W),
        dilation=(D_H, D_W),
        is_depthwise=True,
        has_bias=True,
    )


@m1triton.jit
def pointwise_conv_kernel(
    w_ptr,
    x_ptr,
    out_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_xb,
    stride_ob,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Batched 1x1 convolution: each program owns 16 output channels by BLOCK_N pixels of one image."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    co0 = pid_m * 16
    x_base = x_ptr + pid_b.to(tl.int64) * stride_xb
    o_base = out_ptr + pid_b.to(tl.int64) * stride_ob

    m0 = (co0 + 0) < M
    m1 = (co0 + 1) < M
    m2 = (co0 + 2) < M
    m3 = (co0 + 3) < M
    m4 = (co0 + 4) < M
    m5 = (co0 + 5) < M
    m6 = (co0 + 6) < M
    m7 = (co0 + 7) < M
    m8 = (co0 + 8) < M
    m9 = (co0 + 9) < M
    m10 = (co0 + 10) < M
    m11 = (co0 + 11) < M
    m12 = (co0 + 12) < M
    m13 = (co0 + 13) < M
    m14 = (co0 + 14) < M
    m15 = (co0 + 15) < M

    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc8 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc9 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc13 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc14 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc15 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        acc0 = acc0 + tl.load(bias_ptr + co0 + 0, mask=m0, other=0.0)
        acc1 = acc1 + tl.load(bias_ptr + co0 + 1, mask=m1, other=0.0)
        acc2 = acc2 + tl.load(bias_ptr + co0 + 2, mask=m2, other=0.0)
        acc3 = acc3 + tl.load(bias_ptr + co0 + 3, mask=m3, other=0.0)
        acc4 = acc4 + tl.load(bias_ptr + co0 + 4, mask=m4, other=0.0)
        acc5 = acc5 + tl.load(bias_ptr + co0 + 5, mask=m5, other=0.0)
        acc6 = acc6 + tl.load(bias_ptr + co0 + 6, mask=m6, other=0.0)
        acc7 = acc7 + tl.load(bias_ptr + co0 + 7, mask=m7, other=0.0)
        acc8 = acc8 + tl.load(bias_ptr + co0 + 8, mask=m8, other=0.0)
        acc9 = acc9 + tl.load(bias_ptr + co0 + 9, mask=m9, other=0.0)
        acc10 = acc10 + tl.load(bias_ptr + co0 + 10, mask=m10, other=0.0)
        acc11 = acc11 + tl.load(bias_ptr + co0 + 11, mask=m11, other=0.0)
        acc12 = acc12 + tl.load(bias_ptr + co0 + 12, mask=m12, other=0.0)
        acc13 = acc13 + tl.load(bias_ptr + co0 + 13, mask=m13, other=0.0)
        acc14 = acc14 + tl.load(bias_ptr + co0 + 14, mask=m14, other=0.0)
        acc15 = acc15 + tl.load(bias_ptr + co0 + 15, mask=m15, other=0.0)

    for k in range(0, K):
        xv = tl.load(x_base + k * N + offs_n, mask=mask_n, other=0.0)
        acc0 = acc0 + tl.load(w_ptr + (co0 + 0) * K + k, mask=m0, other=0.0) * xv
        acc1 = acc1 + tl.load(w_ptr + (co0 + 1) * K + k, mask=m1, other=0.0) * xv
        acc2 = acc2 + tl.load(w_ptr + (co0 + 2) * K + k, mask=m2, other=0.0) * xv
        acc3 = acc3 + tl.load(w_ptr + (co0 + 3) * K + k, mask=m3, other=0.0) * xv
        acc4 = acc4 + tl.load(w_ptr + (co0 + 4) * K + k, mask=m4, other=0.0) * xv
        acc5 = acc5 + tl.load(w_ptr + (co0 + 5) * K + k, mask=m5, other=0.0) * xv
        acc6 = acc6 + tl.load(w_ptr + (co0 + 6) * K + k, mask=m6, other=0.0) * xv
        acc7 = acc7 + tl.load(w_ptr + (co0 + 7) * K + k, mask=m7, other=0.0) * xv
        acc8 = acc8 + tl.load(w_ptr + (co0 + 8) * K + k, mask=m8, other=0.0) * xv
        acc9 = acc9 + tl.load(w_ptr + (co0 + 9) * K + k, mask=m9, other=0.0) * xv
        acc10 = acc10 + tl.load(w_ptr + (co0 + 10) * K + k, mask=m10, other=0.0) * xv
        acc11 = acc11 + tl.load(w_ptr + (co0 + 11) * K + k, mask=m11, other=0.0) * xv
        acc12 = acc12 + tl.load(w_ptr + (co0 + 12) * K + k, mask=m12, other=0.0) * xv
        acc13 = acc13 + tl.load(w_ptr + (co0 + 13) * K + k, mask=m13, other=0.0) * xv
        acc14 = acc14 + tl.load(w_ptr + (co0 + 14) * K + k, mask=m14, other=0.0) * xv
        acc15 = acc15 + tl.load(w_ptr + (co0 + 15) * K + k, mask=m15, other=0.0) * xv

    tl.store(o_base + (co0 + 0) * N + offs_n, acc0, mask=mask_n & m0)
    tl.store(o_base + (co0 + 1) * N + offs_n, acc1, mask=mask_n & m1)
    tl.store(o_base + (co0 + 2) * N + offs_n, acc2, mask=mask_n & m2)
    tl.store(o_base + (co0 + 3) * N + offs_n, acc3, mask=mask_n & m3)
    tl.store(o_base + (co0 + 4) * N + offs_n, acc4, mask=mask_n & m4)
    tl.store(o_base + (co0 + 5) * N + offs_n, acc5, mask=mask_n & m5)
    tl.store(o_base + (co0 + 6) * N + offs_n, acc6, mask=mask_n & m6)
    tl.store(o_base + (co0 + 7) * N + offs_n, acc7, mask=mask_n & m7)
    tl.store(o_base + (co0 + 8) * N + offs_n, acc8, mask=mask_n & m8)
    tl.store(o_base + (co0 + 9) * N + offs_n, acc9, mask=mask_n & m9)
    tl.store(o_base + (co0 + 10) * N + offs_n, acc10, mask=mask_n & m10)
    tl.store(o_base + (co0 + 11) * N + offs_n, acc11, mask=mask_n & m11)
    tl.store(o_base + (co0 + 12) * N + offs_n, acc12, mask=mask_n & m12)
    tl.store(o_base + (co0 + 13) * N + offs_n, acc13, mask=mask_n & m13)
    tl.store(o_base + (co0 + 14) * N + offs_n, acc14, mask=mask_n & m14)
    tl.store(o_base + (co0 + 15) * N + offs_n, acc15, mask=mask_n & m15)


def _pair(v: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(v, int):
        return (int(v), int(v))
    vals = tuple(int(x) for x in v)
    if len(vals) == 1:
        return (vals[0], vals[0])
    if len(vals) != 2:
        raise ValueError(f"expected 2 conv parameters, got {vals!r}")
    return vals


def conv_output_size(size: int, k: int, s: int, p: int, d: int) -> int:
    """PyTorch convolution output extent: floor((in + 2p - d*(k-1) - 1) / s) + 1."""
    span = size + 2 * p - d * (k - 1) - 1
    if span < 0:
        raise ValueError("convolution kernel larger than padded input")
    return span // s + 1


def _halo(tile: int, k: int, s: int, d: int) -> int:
    return (tile - 1) * s + d * (k - 1) + 1


def _tile_shape(h_out: int, w_out: int, k: tuple[int, int], s: tuple[int, int], d: tuple[int, int]) -> tuple[int, int]:
    block_w = 64 if w_out > 32 else 32
    block_h = 32 if h_out > 16 else (16 if h_out > 8 else 8)
    while block_h > 8 and _halo(block_h, k[0], s[0], d[0]) * _halo(block_w, k[1], s[1], d[1]) * 4 > _HALO_BUDGET_BYTES:
        block_h //= 2
    while block_w > 32 and _halo(block_h, k[0], s[0], d[0]) * _halo(block_w, k[1], s[1], d[1]) * 4 > _HALO_BUDGET_BYTES:
        block_w //= 2
    return block_w, block_h


def depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> torch.Tensor:
    """Dispatches one SRAM-halo depthwise Metal kernel over an (N, C_in, H, W) tensor.

    `weight` has shape (C_out, 1, K_H, K_W); output channel c reads input channel c // (C_out // C_in).
    """
    if x.dim() != 4 or weight.dim() != 4 or weight.shape[1] != 1:
        raise ValueError("depthwise_conv2d expects x (N, C, H, W) and weight (C_out, 1, K_H, K_W)")
    x_c = x.contiguous()
    w_c = weight.contiguous()
    n, c_in, h_in, w_in = (int(v) for v in x_c.shape)
    c_out, k_h, k_w = int(w_c.shape[0]), int(w_c.shape[2]), int(w_c.shape[3])
    if c_out % c_in != 0:
        raise ValueError("depthwise out_channels must be a multiple of in_channels")
    s, p, d = _pair(stride), _pair(padding), _pair(dilation)
    h_out = conv_output_size(h_in, k_h, s[0], p[0], d[0])
    w_out = conv_output_size(w_in, k_w, s[1], p[1], d[1])
    out = torch.empty((n, c_out, h_out, w_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    block_w, block_h = _tile_shape(h_out, w_out, (k_h, k_w), s, d)
    grid = (m1triton.cdiv(w_out, block_w), m1triton.cdiv(h_out, block_h), n * c_out)
    consts = dict(
        K_H=k_h, K_W=k_w, S_H=s[0], S_W=s[1], P_H=p[0], P_W=p[1], D_H=d[0], D_W=d[1],
        BLOCK_W=block_w, BLOCK_H=block_h,
    )
    if bias is None:
        conv_dw2d_kernel[grid](x_c, w_c, out, n, c_in, c_out, h_in, w_in, h_out, w_out, **consts)
    else:
        conv_dw2d_bias_kernel[grid](
            x_c, w_c, out, bias.contiguous(), n, c_in, c_out, h_in, w_in, h_out, w_out, **consts
        )
    return out


def pointwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Batched 1x1 convolution out[n] = W (C_out, C_in) @ x[n] (C_in, H*W) with channel-blocked FMA tiles."""
    if x.dim() != 4:
        raise ValueError("pointwise_conv2d expects an (N, C_in, H, W) tensor")
    x_c = x.contiguous()
    n, c_in, h, w = (int(v) for v in x_c.shape)
    w_c = weight.contiguous().reshape(int(weight.shape[0]), c_in)
    c_out = int(w_c.shape[0])
    hw = h * w
    out = torch.empty((n, c_out, h, w), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    block_n = 128
    grid = (m1triton.cdiv(hw, block_n), m1triton.cdiv(c_out, 16), n)
    bias_arg = bias.contiguous() if bias is not None else w_c
    pointwise_conv_kernel[grid](
        w_c, x_c, out, bias_arg, c_out, hw, c_in, c_in * hw, c_out * hw,
        BLOCK_N=block_n, HAS_BIAS=bias is not None,
    )
    return out
