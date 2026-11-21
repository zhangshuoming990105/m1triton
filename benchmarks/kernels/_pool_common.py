"""Reusable M1Triton 1D/2D/3D max and average pooling kernels (KernelBench L1 #41-#46)."""

from __future__ import annotations

from typing import Sequence

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.jit
def pool1d_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    W_in,
    W_out,
    KIND: tl.constexpr,
    K_W: tl.constexpr,
    S_W: tl.constexpr,
    P_W: tl.constexpr,
    D_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    CEIL_MODE: tl.constexpr,
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
        padding=(P_W,),
        dilation=(D_W,),
        count_include_pad=COUNT_INCLUDE_PAD,
        ceil_mode=CEIL_MODE,
    )


@m1triton.jit
def pool2d_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H_in,
    W_in,
    H_out,
    W_out,
    KIND: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    D_H: tl.constexpr,
    D_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    CEIL_MODE: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tlx.pool(
        x_ptr,
        out_ptr,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        kind=KIND,
        ndim=2,
        kernel_size=(K_H, K_W),
        stride=(S_H, S_W),
        padding=(P_H, P_W),
        dilation=(D_H, D_W),
        count_include_pad=COUNT_INCLUDE_PAD,
        ceil_mode=CEIL_MODE,
    )


@m1triton.jit
def pool3d_kernel(
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
    BLOCK_H: tl.constexpr,
):
    tlx.pool(
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
        kind=KIND,
        ndim=3,
        kernel_size=(K_D, K_H, K_W),
        stride=(S_D, S_H, S_W),
        padding=(P_D, P_H, P_W),
        dilation=(D_D, D_H, D_W),
        count_include_pad=COUNT_INCLUDE_PAD,
        ceil_mode=CEIL_MODE,
    )


def _as_tuple(v: int | Sequence[int] | None, ndim: int, default: Sequence[int]) -> tuple[int, ...]:
    if v is None:
        return tuple(int(x) for x in default)
    if isinstance(v, int):
        return (int(v),) * ndim
    vals = tuple(int(x) for x in v)
    if len(vals) == 1:
        return vals * ndim
    if len(vals) != ndim:
        raise ValueError(f"expected {ndim} pooling parameters, got {vals!r}")
    return vals


def pool_output_size(size: int, k: int, s: int, p: int, d: int, ceil_mode: bool) -> int:
    """PyTorch pooling output extent: floor/ceil((in + 2p - d*(k-1) - 1) / s) + 1."""
    span = size + 2 * p - d * (k - 1) - 1
    if span < 0:
        raise ValueError("pooling window larger than padded input")
    out = (-(-span // s) if ceil_mode else span // s) + 1
    if ceil_mode and (out - 1) * s >= size + p:
        out -= 1
    return out


def _block_shape(ndim: int, use_halo: bool) -> tuple[int, int]:
    if ndim == 1:
        return (1024 if use_halo else 256), 1
    if ndim == 2:
        return 32, 32
    return 32, 8


def pool_nd(
    x: torch.Tensor,
    *,
    kind: str,
    ndim: int,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None = None,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    count_include_pad: bool = True,
    ceil_mode: bool = False,
) -> torch.Tensor:
    """Dispatches a single M1Triton Metal pooling kernel over an (N, C, *spatial) tensor."""
    if x.dim() != ndim + 2:
        raise ValueError(f"pool{ndim}d expects a {ndim + 2}D (N, C, *spatial) tensor, got {tuple(x.shape)}")
    kind = kind.lower()
    k = _as_tuple(kernel_size, ndim, ())
    s = _as_tuple(stride if stride is not None else k, ndim, k)
    p = _as_tuple(padding, ndim, (0,) * ndim)
    d = _as_tuple(dilation, ndim, (1,) * ndim) if kind == "max" else (1,) * ndim

    x_c = x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [
        pool_output_size(spatial_in[i], k[i], s[i], p[i], d[i], ceil_mode) for i in range(ndim)
    ]
    out = torch.empty((n, c, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    use_halo = ndim == 1 and s[0] == 1
    block_w, block_h = _block_shape(ndim, use_halo)
    w_out = spatial_out[-1]
    h_out = spatial_out[-2] if ndim >= 2 else 1
    d_out = spatial_out[-3] if ndim == 3 else 1
    grid = (m1triton.cdiv(w_out, block_w), m1triton.cdiv(h_out, block_h), n * c * d_out)
    common = dict(
        KIND=kind,
        COUNT_INCLUDE_PAD=bool(count_include_pad),
        CEIL_MODE=bool(ceil_mode),
        BLOCK_W=block_w,
        wait=True,
    )
    if ndim == 1:
        pool1d_kernel[grid](
            x_c, out, n, c, spatial_in[0], spatial_out[0],
            K_W=k[0], S_W=s[0], P_W=p[0], D_W=d[0], **common,
        )
    elif ndim == 2:
        pool2d_kernel[grid](
            x_c, out, n, c, spatial_in[0], spatial_in[1], spatial_out[0], spatial_out[1],
            K_H=k[0], K_W=k[1], S_H=s[0], S_W=s[1], P_H=p[0], P_W=p[1], D_H=d[0], D_W=d[1],
            BLOCK_H=block_h, **common,
        )
    else:
        pool3d_kernel[grid](
            x_c, out, n, c, *spatial_in, *spatial_out,
            K_D=k[0], K_H=k[1], K_W=k[2], S_D=s[0], S_H=s[1], S_W=s[2],
            P_D=p[0], P_H=p[1], P_W=p[2], D_D=d[0], D_H=d[1], D_W=d[2],
            BLOCK_H=block_h, **common,
        )
    return out
