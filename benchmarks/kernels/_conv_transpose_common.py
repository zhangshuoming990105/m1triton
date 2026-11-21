"""Reusable M1Triton transposed-convolution kernels (KernelBench L1 ConvTranspose{1,2,3}d tasks).

Each launch compiles a single output-centric implicit-GEMM Metal kernel using ``simdgroup_float8x8``
with stride-phase splitting (zero scatter atomics, zero wasted multiply-by-zero taps). When channels
are multiples of 8 in float32, weights are packed into 8x8 MMA blocks and input patches are staged in
an SRAM spatial halo tile; all other shapes use the general padded-SRAM fallback path.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

_MAX_HALO_SMEM_BYTES = 28672


@m1triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    N,
    W_in,
    W_out,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K_W: tl.constexpr,
    S_W: tl.constexpr,
    P_W: tl.constexpr,
    D_W: tl.constexpr,
    OP_W: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    W_BLK8: tl.constexpr,
    HALO: tl.constexpr,
    TILE_W: tl.constexpr,
):
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, b_ptr, N, C_IN, C_OUT, W_in, W_out,
            ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,),
            output_padding=(OP_W,), groups=GROUPS, transposed=True, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, N, C_IN, C_OUT, W_in, W_out,
            ndim=1, kernel_size=(K_W,), stride=(S_W,), padding=(P_W,), dilation=(D_W,),
            output_padding=(OP_W,), groups=GROUPS, transposed=True, has_bias=False,
        )


@m1triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    N,
    H_in,
    W_in,
    H_out,
    W_out,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    OP_H: tl.constexpr,
    OP_W: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    W_BLK8: tl.constexpr,
    HALO: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
):
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, b_ptr, N, C_IN, C_OUT, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
            output_padding=(OP_H, OP_W), groups=GROUPS, transposed=True, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, N, C_IN, C_OUT, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
            output_padding=(OP_H, OP_W), groups=GROUPS, transposed=True, has_bias=False,
        )


@m1triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    N,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
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
    OP_D: tl.constexpr,
    OP_H: tl.constexpr,
    OP_W: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    W_BLK8: tl.constexpr,
    HALO: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
):
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, b_ptr, N, C_IN, C_OUT, D_in, H_in, W_in, D_out, H_out, W_out,
            ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
            dilation=(D_D, D_H, D_W), output_padding=(OP_D, OP_H, OP_W), groups=GROUPS,
            transposed=True, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, N, C_IN, C_OUT, D_in, H_in, W_in, D_out, H_out, W_out,
            ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
            dilation=(D_D, D_H, D_W), output_padding=(OP_D, OP_H, OP_W), groups=GROUPS,
            transposed=True, has_bias=False,
        )


def as_tuple(v: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(v, int):
        return (int(v),) * ndim
    vals = tuple(int(e) for e in v)
    if len(vals) == 1:
        return vals * ndim
    if len(vals) != ndim:
        raise ValueError(f"expected {ndim} convolution parameters, got {vals!r}")
    return vals


def conv_transpose_output_size(size: int, k: int, s: int, p: int, d: int, op: int) -> int:
    """PyTorch ConvTranspose extent: (in - 1) * s - 2p + d * (k - 1) + op + 1."""
    return (size - 1) * s - 2 * p + d * (k - 1) + op + 1


def pack_weight_blk8(weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Packs (C_in, C_out/G, *k) float32 weights into contiguous (G, Ci_g/8, KT, Co_g/8, 8, 8) MMA blocks."""
    c_in = int(weight.shape[0])
    c_in_g = c_in // groups
    c_out_g = int(weight.shape[1])
    kt = math.prod(int(v) for v in weight.shape[2:])
    return (
        weight.contiguous()
        .view(groups, c_in_g // 8, 8, c_out_g // 8, 8, kt)
        .permute(0, 1, 5, 3, 4, 2)
        .contiguous()
    )


def _halo_overhangs(k: tuple[int, ...], s: tuple[int, ...], d: tuple[int, ...]) -> list[int]:
    overhangs: list[int] = []
    for ki, si, di in zip(k, s, d):
        g = math.gcd(si, di)
        step = si // g
        delta = di // g
        nmax = (ki + step - 1) // step
        overhangs.append((nmax - 1) * delta)
    return overhangs


def _choose_halo_tile(
    ndim: int, bn: int, bk: int, overhangs: list[int]
) -> tuple[tuple[int, ...], int] | None:
    candidates: list[tuple[int, ...]] = []
    if ndim == 1:
        candidates = [(bn,)]
    elif ndim == 2:
        for tw in (64, 32, 16, 8):
            if bn % tw == 0:
                candidates.append((bn // tw, tw))
    else:
        for tw in (64, 32, 16, 8):
            if bn % tw != 0:
                continue
            rem = bn // tw
            for th in (8, 4, 2, 1):
                if rem % th == 0:
                    candidates.append((rem // th, th, tw))

    best_tile: tuple[int, ...] | None = None
    best_key: tuple[int, int] | None = None
    best_bytes = 0
    for cand in candidates:
        hplane = 1
        for ti, ov in zip(cand, overhangs):
            hplane *= ti + ov
        hplane_ld = hplane + (4 if hplane % 16 == 0 else 0)
        smem_bytes = bk * hplane_ld * 4
        if smem_bytes > _MAX_HALO_SMEM_BYTES:
            continue
        key = (hplane_ld, -cand[-1])
        if best_key is None or key < best_key:
            best_key = key
            best_tile = cand
            best_bytes = smem_bytes
    return (best_tile, best_bytes) if best_tile is not None else None


def select_schedule(
    ndim: int,
    c_in_g: int,
    c_out_g: int,
    k: tuple[int, ...],
    s: tuple[int, ...],
    d: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[int, int, int, int, bool, bool, tuple[int, ...]]:
    """Selects (bm, bn, bk, num_sg, w_blk8, use_halo, spatial_tiles) tuned for M1 Max 32 KB SRAM."""
    if c_out_g >= 128:
        bm, bn, num_sg = 128, 64, 8
        bk = 32 if (ndim == 1 and c_in_g % 32 == 0) else (16 if c_in_g % 16 == 0 else 8)
    elif c_out_g >= 64:
        bm, bn, num_sg = 64, 64, 4
        bk = 16 if c_in_g % 16 == 0 else 8
    else:
        bm = max(8, -(-c_out_g // 8) * 8)
        bn = 64
        num_sg = 4
        bk = 16 if c_in_g % 16 == 0 else 8

    w_blk8 = dtype == torch.float32 and c_in_g % bk == 0 and c_out_g % bm == 0
    use_halo = False
    tiles = (1,) * (ndim - 1) + (bn,)
    if w_blk8:
        overhangs = _halo_overhangs(k, s, d)
        picked = _choose_halo_tile(ndim, bn, bk, overhangs)
        if picked is None and bk > 8 and c_in_g % 8 == 0:
            picked = _choose_halo_tile(ndim, bn, 8, overhangs)
            if picked is not None:
                bk = 8
        if picked is not None:
            tiles = picked[0]
            use_halo = True
    return bm, bn, bk, num_sg, w_blk8, use_halo, tiles


_KERNELS = {1: conv_transpose1d_kernel, 2: conv_transpose2d_kernel, 3: conv_transpose3d_kernel}
_DIM_TAGS = {1: ("W",), 2: ("H", "W"), 3: ("D", "H", "W")}


def conv_transpose_nd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    ndim: int,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    output_padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    packed_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatches one M1Triton Metal implicit-GEMM transposed-convolution kernel on (N, C_in, *spatial)."""
    if x.dim() != ndim + 2 or weight.dim() != ndim + 2:
        raise ValueError(f"conv_transpose{ndim}d expects {ndim + 2}D input and weight tensors")
    x_c = x.contiguous()
    n, c_in = int(x_c.shape[0]), int(x_c.shape[1])
    groups = int(groups)
    if int(weight.shape[0]) != c_in or c_in % groups != 0:
        raise ValueError("weight must have shape (C_in, C_out / groups, *kernel_size)")
    c_out = int(weight.shape[1]) * groups
    k = tuple(int(v) for v in weight.shape[2:])
    s = as_tuple(stride, ndim)
    p = as_tuple(padding, ndim)
    op = as_tuple(output_padding, ndim)
    d = as_tuple(dilation, ndim)
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [conv_transpose_output_size(spatial_in[i], k[i], s[i], p[i], d[i], op[i]) for i in range(ndim)]
    out = torch.empty((n, c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    bm, bn, bk, num_sg, w_blk8, use_halo, tiles = select_schedule(
        ndim, c_in // groups, c_out // groups, k, s, d, weight.dtype
    )
    q_dims = [-(-spatial_out[i] // s[i]) for i in range(ndim)]
    phases = math.prod(s)
    if use_halo:
        grid_x = math.prod(m1triton.cdiv(qd, td) for qd, td in zip(q_dims, tiles))
    else:
        grid_x = m1triton.cdiv(math.prod(q_dims), bn)
    grid = (grid_x, m1triton.cdiv(c_out // groups, bm), n * groups * phases)

    if w_blk8:
        w_arg = packed_weight if packed_weight is not None else pack_weight_blk8(weight, groups)
    else:
        w_arg = weight.contiguous()
    b_arg = bias.contiguous() if bias is not None else w_arg

    meta: dict[str, object] = dict(
        C_IN=c_in,
        C_OUT=c_out,
        GROUPS=groups,
        HAS_BIAS=bias is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=num_sg,
        W_BLK8=int(w_blk8),
        HALO=int(use_halo),
        wait=True,
    )
    for i, tag in enumerate(_DIM_TAGS[ndim]):
        meta[f"K_{tag}"] = k[i]
        meta[f"S_{tag}"] = s[i]
        meta[f"P_{tag}"] = p[i]
        meta[f"D_{tag}"] = d[i]
        meta[f"OP_{tag}"] = op[i]
        meta[f"TILE_{tag}"] = tiles[i]

    _KERNELS[ndim][grid](x_c, w_arg, out, b_arg, n, *spatial_in, *spatial_out, **meta)
    return out


class ConvTransposeNdBase(nn.Module):
    """Parameter-only ConvTranspose{1,2,3}d replacement dispatching the M1Triton implicit-GEMM kernel."""

    ndim: int = 2
    ref_attr: str = "conv_transpose2d"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        nd = self.ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = as_tuple(kernel_size, nd)
        self.stride = as_tuple(stride, nd)
        self.padding = as_tuple(padding, nd)
        self.output_padding = as_tuple(output_padding, nd)
        self.dilation = as_tuple(dilation, nd)
        self.groups = int(groups)
        if self.in_channels % self.groups != 0 or self.out_channels % self.groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.weight = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels // self.groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self._packed_w: torch.Tensor | None = None
        self._packed_key: tuple[int, int, torch.device, torch.dtype] | None = None

    def _get_packed_weight(self) -> torch.Tensor | None:
        w = self.weight.detach()
        if w.dtype != torch.float32:
            return None
        g = self.groups
        c_in_g = int(w.shape[0]) // g
        c_out_g = int(w.shape[1])
        if c_in_g % 8 != 0 or c_out_g % 8 != 0:
            return None
        key = (int(w.data_ptr()), int(w._version), w.device, w.dtype)
        if self._packed_key != key or self._packed_w is None:
            self._packed_w = pack_weight_blk8(w, g)
            self._packed_key = key
        return self._packed_w

    def _apply_reference_geometry(self, conv: nn.Module) -> None:
        nd = self.ndim
        self.stride = as_tuple(conv.stride, nd)
        self.padding = as_tuple(conv.padding, nd)
        self.output_padding = as_tuple(conv.output_padding, nd)
        self.dilation = as_tuple(conv.dilation, nd)
        self.groups = int(conv.groups)

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ConvTransposeNdBase":
        conv = getattr(ref_model, cls.ref_attr)
        model = cls.__new__(cls)
        ConvTransposeNdBase.__init__(
            model,
            int(conv.in_channels),
            int(conv.out_channels),
            tuple(int(v) for v in conv.kernel_size),
            stride=tuple(conv.stride),
            padding=tuple(conv.padding),
            output_padding=tuple(conv.output_padding),
            dilation=tuple(conv.dilation),
            groups=int(conv.groups),
            bias=conv.bias is not None,
        )
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        conv = getattr(ref_model, self.ref_attr)
        self._apply_reference_geometry(conv)
        with torch.no_grad():
            self.weight.copy_(conv.weight.detach().to(self.weight.device))
            if self.bias is not None and conv.bias is not None:
                self.bias.copy_(conv.bias.detach().to(self.bias.device))
        self._packed_w = None
        self._packed_key = None
        self._get_packed_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose_nd(
            x,
            self.weight.detach(),
            None if self.bias is None else self.bias.detach(),
            ndim=self.ndim,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            packed_weight=self._get_packed_weight(),
        )
