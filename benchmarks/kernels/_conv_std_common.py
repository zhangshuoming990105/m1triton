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
    c_in_per_group: int, c_out_per_group: int, k_spatial: int, pointwise: bool
) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, NUM_SG) schedule mirroring ConvLoweringPass."""
    k_total = c_in_per_group * k_spatial
    if pointwise:
        bm = 16 if c_out_per_group <= 16 else (32 if c_out_per_group <= 32 else 64)
        bk = 32 if (k_total > 0 and k_total % 32 == 0) else 16
        return bm, 64, bk, 4
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
    out = torch.empty((n, c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out
    pixels = n * math.prod(spatial_out)
    if pixels >= 2**32:
        raise ValueError("N * prod(spatial_out) must fit in 32 bits")
    pointwise = all(v == 1 for v in k) and all(v == 1 for v in s) and all(v == 0 for v in p)
    c_out_g = c_out // groups
    bm, bn, bk, num_sg = tile if tile is not None else select_tile(c_in // groups, c_out_g, math.prod(k), pointwise)
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
            self.weight,
            self.bias,
            ndim=self._NDIM,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
