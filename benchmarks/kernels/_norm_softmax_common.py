"""Reusable M1Triton online softmax and Welford normalization kernels for Apple M1 Max."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

_TG_THREADS = 256
_STATS_ELEMS_PER_TG = 65536
_APPLY_ELEMS_PER_TG = 8192
_FROB_STAGE1_TGS = 2048
_RMS_TILE = 64


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _scratch(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.empty(max(int(n), 1), device=like.device, dtype=torch.float32)


@m1triton.jit
def online_softmax_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
    LOG_SOFTMAX: tl.constexpr,
):
    tlx.online_softmax(x_ptr, out_ptr, NUM_ROWS, ROW_LEN, axis=1, log_softmax=LOG_SOFTMAX)


@m1triton.jit
def row_norm_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
    KIND: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm(KIND, x_ptr, out_ptr, NUM_ROWS, ROW_LEN, eps=EPS, axis=1)


@m1triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm("rms_norm", x_ptr, out_ptr, N, C, HW, eps=EPS, axis=1)


@m1triton.jit
def frobenius_stage1_kernel(x_ptr, partial_ptr, TOTAL: tl.constexpr):
    tlx.welford_norm("frobenius_norm", x_ptr, partial_ptr, TOTAL, stage=1)


@m1triton.jit
def frobenius_stage2_kernel(x_ptr, out_ptr, partial_ptr, TOTAL: tl.constexpr, NUM_PARTIALS: tl.constexpr):
    tlx.welford_norm("frobenius_norm", x_ptr, out_ptr, partial_ptr, TOTAL, NUM_PARTIALS, stage=2)


@m1triton.jit
def batch_norm_stats_kernel(
    x_ptr,
    partial_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    tlx.welford_norm("batch_norm", x_ptr, partial_ptr, N, C, HW, CHUNKS, stage=1, axis=1)


@m1triton.jit
def batch_norm_apply_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHUNKS: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm(
        "batch_norm",
        x_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        N,
        C,
        HW,
        CHUNKS,
        eps=EPS,
        stage=2,
        axis=1,
        has_weight=True,
        has_bias=True,
    )


@m1triton.jit
def group_norm_stats_kernel(
    x_ptr,
    partial_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHUNKS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    tlx.welford_norm("group_norm", x_ptr, partial_ptr, N, C, HW, CHUNKS, num_groups=GROUPS, stage=1, axis=1)


@m1triton.jit
def group_norm_apply_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHUNKS: tl.constexpr,
    GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm(
        "group_norm",
        x_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        N,
        C,
        HW,
        CHUNKS,
        eps=EPS,
        num_groups=GROUPS,
        stage=2,
        axis=1,
        has_weight=True,
        has_bias=True,
    )


@m1triton.jit
def layer_norm_stats_kernel(x_ptr, partial_ptr, N: tl.constexpr, NORM_SIZE: tl.constexpr, CHUNKS: tl.constexpr):
    tlx.welford_norm("layer_norm", x_ptr, partial_ptr, N, NORM_SIZE, CHUNKS, stage=1)


@m1triton.jit
def layer_norm_apply_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    N: tl.constexpr,
    NORM_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.welford_norm(
        "layer_norm",
        x_ptr,
        out_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        N,
        NORM_SIZE,
        CHUNKS,
        eps=EPS,
        stage=2,
        has_weight=True,
        has_bias=True,
    )


def softmax_rows(x: torch.Tensor, *, log_softmax: bool = False) -> torch.Tensor:
    """Row-wise (last-dim) softmax / log-softmax of a 2D tensor via one fused online-softmax kernel."""
    x = _contig(x)
    rows, row_len = int(x.shape[0]), int(x.shape[-1])
    out = torch.empty_like(x)
    online_softmax_kernel[(rows, 1, 1)](x, out, rows, row_len, bool(log_softmax))
    return out


def row_normalize(x: torch.Tensor, kind: str, row_len: int, *, eps: float = 1e-5) -> torch.Tensor:
    """Per-row normalization (`instance_norm`, `l1_norm`, `l2_norm`) over contiguous rows of `row_len`."""
    x = _contig(x)
    row_len = int(row_len)
    rows = x.numel() // max(row_len, 1)
    out = torch.empty_like(x)
    row_norm_kernel[(rows, 1, 1)](x, out, rows, row_len, kind, float(eps))
    return out


def rms_norm_channels(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    """RMS normalization across dim=1 of an (N, C, *) tensor."""
    x = _contig(x)
    n, c = int(x.shape[0]), int(x.shape[1])
    hw = x.numel() // max(n * c, 1)
    out = torch.empty_like(x)
    rms_norm_kernel[(_cdiv(hw, _RMS_TILE), n, 1)](x, out, n, c, hw, float(eps))
    return out


def frobenius_normalize(x: torch.Tensor) -> torch.Tensor:
    """x / ||x||_F via a two-stage grid reduction plus a fused scaling pass."""
    x = _contig(x)
    total = int(x.numel())
    g1 = max(1, min(_FROB_STAGE1_TGS, _cdiv(total, _TG_THREADS * 16)))
    partial = _scratch(g1, x)
    out = torch.empty_like(x)
    frobenius_stage1_kernel[(g1, 1, 1)](x, partial, total)
    g2 = max(1, _cdiv(total, _APPLY_ELEMS_PER_TG * 2))
    frobenius_stage2_kernel[(g2, 1, 1)](x, out, partial, total, g1)
    return out


def batch_norm_train(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Training-mode BatchNorm (biased batch statistics over N, H, W) with a per-channel affine."""
    x = _contig(x)
    n, c = int(x.shape[0]), int(x.shape[1])
    hw = x.numel() // max(n * c, 1)
    chunks = max(1, min(n, _cdiv(n * hw, _STATS_ELEMS_PER_TG)))
    partial = _scratch(c * chunks * 3, x)
    out = torch.empty_like(x)
    batch_norm_stats_kernel[(chunks, c, 1)](x, partial, n, c, hw, chunks)
    tiles = max(1, _cdiv(hw, _APPLY_ELEMS_PER_TG))
    batch_norm_apply_kernel[(tiles, n * c, 1)](
        x, out, _contig(weight), _contig(bias), partial, n, c, hw, chunks, float(eps)
    )
    return out


def group_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, num_groups: int, eps: float
) -> torch.Tensor:
    """GroupNorm over (C/G, *) per (n, g) followed by a per-channel affine."""
    x = _contig(x)
    n, c = int(x.shape[0]), int(x.shape[1])
    hw = x.numel() // max(n * c, 1)
    group_elems = (c // num_groups) * hw
    chunks = max(1, min(32, _cdiv(group_elems, _STATS_ELEMS_PER_TG)))
    partial = _scratch(n * num_groups * chunks * 3, x)
    out = torch.empty_like(x)
    group_norm_stats_kernel[(chunks, n * num_groups, 1)](x, partial, n, c, hw, chunks, int(num_groups))
    tiles = max(1, _cdiv(hw, _APPLY_ELEMS_PER_TG))
    group_norm_apply_kernel[(tiles, n * c, 1)](
        x, out, _contig(weight), _contig(bias), partial, n, c, hw, chunks, int(num_groups), float(eps)
    )
    return out


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float) -> torch.Tensor:
    """LayerNorm over the trailing `weight.numel()` elements with elementwise affine."""
    x = _contig(x)
    norm_size = int(weight.numel())
    n = x.numel() // max(norm_size, 1)
    chunks = max(1, min(128, _cdiv(norm_size, _STATS_ELEMS_PER_TG)))
    partial = _scratch(n * chunks * 3, x)
    out = torch.empty_like(x)
    layer_norm_stats_kernel[(chunks, n, 1)](x, partial, n, norm_size, chunks)
    tiles = max(1, _cdiv(norm_size, _APPLY_ELEMS_PER_TG))
    layer_norm_apply_kernel[(tiles, n, 1)](
        x, out, _contig(weight), _contig(bias), partial, n, norm_size, chunks, float(eps)
    )
    return out
