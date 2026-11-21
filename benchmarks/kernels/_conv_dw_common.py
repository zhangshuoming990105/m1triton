"""Reusable M1Triton depthwise and pointwise convolution kernels (KernelBench L1 #82-#86)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_std_common import conv_nd

_HALO_BUDGET_BYTES = 16 * 1024
_REG_SW_MAX_KERNEL = 15


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
    VEC_W: tl.constexpr = 1,
    ROWS_PER_THREAD: tl.constexpr = 0,
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
    VEC_W: tl.constexpr = 1,
    ROWS_PER_THREAD: tl.constexpr = 0,
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


def _tile_shape(
    h_out: int, w_out: int, k: tuple[int, int], s: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int, int, int]:
    if s == (1, 1) and d == (1, 1) and k[0] <= _REG_SW_MAX_KERNEL and k[1] <= _REG_SW_MAX_KERNEL:
        block_w = 128
        if h_out >= 64:
            return block_w, 64, 4, 16
        if h_out >= 32:
            return block_w, 32, 4, 8
        return block_w, 16, 4, 4

    block_w = 64 if w_out > 32 else 32
    block_h = 32 if h_out > 16 else (16 if h_out > 8 else 8)
    while block_h > 8 and _halo(block_h, k[0], s[0], d[0]) * _halo(block_w, k[1], s[1], d[1]) * 4 > _HALO_BUDGET_BYTES:
        block_h //= 2
    while block_w > 32 and _halo(block_h, k[0], s[0], d[0]) * _halo(block_w, k[1], s[1], d[1]) * 4 > _HALO_BUDGET_BYTES:
        block_w //= 2
    return block_w, block_h, 1, 0


def depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    wait: bool = True,
) -> torch.Tensor:
    """Dispatches one M1Triton depthwise 2D Metal kernel over an (N, C_in, H, W) tensor.

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

    block_w, block_h, vec_w, rows_per_thread = _tile_shape(h_out, w_out, (k_h, k_w), s, d)
    grid = (m1triton.cdiv(w_out, block_w), m1triton.cdiv(h_out, block_h), n * c_out)
    consts = dict(
        K_H=k_h, K_W=k_w, S_H=s[0], S_W=s[1], P_H=p[0], P_W=p[1], D_H=d[0], D_W=d[1],
        BLOCK_W=block_w, BLOCK_H=block_h, VEC_W=vec_w, ROWS_PER_THREAD=rows_per_thread,
        wait=wait,
    )
    if bias is None:
        conv_dw2d_kernel[grid](x_c, w_c, out, n, c_in, c_out, h_in, w_in, h_out, w_out, **consts)
    else:
        conv_dw2d_bias_kernel[grid](
            x_c, w_c, out, bias.contiguous(), n, c_in, c_out, h_in, w_in, h_out, w_out, **consts
        )
    return out


def pointwise_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """1x1 convolution via M1Triton implicit-GEMM simdgroup-matrix Metal kernel."""
    if x.dim() != 4:
        raise ValueError("pointwise_conv2d expects an (N, C_in, H, W) tensor")
    c_in = int(x.shape[1])
    w_4d = weight if weight.dim() == 4 else weight.reshape(int(weight.shape[0]), c_in, 1, 1)
    return conv_nd(x, w_4d, bias, ndim=2)
