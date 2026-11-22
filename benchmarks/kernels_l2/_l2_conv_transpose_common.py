"""Reusable fused ConvTranspose2d/3d epilogue, tail, BN-folding, and post-op M1Triton kernels (KernelBench L2)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

_L1_KERNELS_DIR = str(Path(__file__).resolve().parents[1] / "kernels")
if _L1_KERNELS_DIR not in sys.path:
    sys.path.insert(0, _L1_KERNELS_DIR)

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx
from _axis_reduce_common import axis_max, axis_mean, axis_min, axis_sum
from _conv_transpose_common import (
    _MAX_HALO_SMEM_BYTES,
    _active_w_info,
    _choose_halo_tile,
    _halo_overhangs,
    _sg_frags,
    as_tuple,
    conv_transpose_output_size,
    pack_weight_blk8,
)
from _norm_softmax_common import (
    _APPLY_ELEMS_PER_TG,
    _STATS_ELEMS_PER_TG,
    _cdiv,
    group_norm,
    group_norm_stats_kernel,
    layer_norm,
    online_softmax_kernel,
)
from _pool_common import (
    _as_tuple as _pool_as_tuple,
    _block_shape as _pool_block_shape,
    pool1d_kernel,
    pool2d_kernel,
    pool3d_kernel,
    pool_output_size,
)


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
    """Batch-safe M1Triton 1D/2D/3D pooling dispatch (respects active command_batch)."""
    kind = kind.lower()
    k = _pool_as_tuple(kernel_size, ndim, ())
    s = _pool_as_tuple(stride if stride is not None else k, ndim, k)
    p = _pool_as_tuple(padding, ndim, (0,) * ndim)
    d = _pool_as_tuple(dilation, ndim, (1,) * ndim) if kind == "max" else (1,) * ndim

    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [
        pool_output_size(spatial_in[i], k[i], s[i], p[i], d[i], ceil_mode) for i in range(ndim)
    ]
    out = torch.empty((n, c, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    use_halo = ndim == 1 and s[0] == 1
    block_w, block_h, vec_w, rows_per_thread = _pool_block_shape(ndim, kind, k, s, d, spatial_out, use_halo)
    w_out = spatial_out[-1]
    h_out = spatial_out[-2] if ndim >= 2 else 1
    d_out = spatial_out[-3] if ndim == 3 else 1
    grid = (m1triton.cdiv(w_out, block_w), m1triton.cdiv(h_out, block_h), n * c * d_out)
    common = dict(
        KIND=kind,
        COUNT_INCLUDE_PAD=bool(count_include_pad),
        CEIL_MODE=bool(ceil_mode),
        BLOCK_W=block_w,
        **_launch_opts(),
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
            BLOCK_H=block_h, VEC_W=vec_w, ROWS_PER_THREAD=rows_per_thread, **common,
        )
    else:
        pool3d_kernel[grid](
            x_c, out, n, c, *spatial_in, *spatial_out,
            K_D=k[0], K_H=k[1], K_W=k[2], S_D=s[0], S_H=s[1], S_W=s[2],
            P_D=p[0], P_H=p[1], P_W=p[2], D_D=d[0], D_H=d[1], D_W=d[2],
            BLOCK_H=block_h, **common,
        )
    return out


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _launch_opts() -> dict[str, bool]:
    if m1triton.runtime.in_batch():
        return {"sync_mps": False, "wait": False}
    return {"sync_mps": True, "wait": True}


@m1triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    eb0,
    eb1,
    es0,
    es1,
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
    FUSE_SW: tl.constexpr = 0,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    if HAS_BIAS:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, b_ptr, N, C_IN, C_OUT, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W),
            dilation=(D_H, D_W), output_padding=(OP_H, OP_W), groups=GROUPS,
            transposed=True, has_bias=True,
        )
    else:
        tlx.implicit_gemm_conv(
            x_ptr, w_ptr, out_ptr, N, C_IN, C_OUT, H_in, W_in, H_out, W_out,
            ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W),
            dilation=(D_H, D_W), output_padding=(OP_H, OP_W), groups=GROUPS,
            transposed=True, has_bias=False,
        )


@m1triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    b_ptr,
    eb0,
    eb1,
    es0,
    es1,
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
    FUSE_SW: tl.constexpr = 0,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
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


_FUSED_KERNELS = {2: conv_transpose2d_fused_kernel, 3: conv_transpose3d_fused_kernel}
_DIM_TAGS = {2: ("H", "W"), 3: ("D", "H", "W")}


def pack_weight_blk8_pad(weight: torch.Tensor, groups: int, bk: int) -> torch.Tensor:
    """Packs (C_in, C_out/G, *k) f32 weights into (G, ceil(Ci_g/bk)*(bk/8), KT, Co_g/8, 8, 8) MMA blocks."""
    c_in = int(weight.shape[0])
    c_in_g = c_in // groups
    c_out_g = int(weight.shape[1])
    if c_in_g % bk == 0:
        return pack_weight_blk8(weight, groups)
    kt = math.prod(int(v) for v in weight.shape[2:])
    c_in_pad = ((c_in_g + bk - 1) // bk) * bk
    w_pad = torch.zeros((groups, c_in_pad, c_out_g, kt), dtype=weight.dtype, device=weight.device)
    w_pad[:, :c_in_g, :, :].copy_(weight.contiguous().view(groups, c_in_g, c_out_g, kt))
    return (
        w_pad.view(groups, c_in_pad // 8, 8, c_out_g // 8, 8, kt)
        .permute(0, 1, 5, 3, 4, 2)
        .contiguous()
    )


def select_schedule_l2(
    ndim: int,
    c_in_g: int,
    c_out_g: int,
    k: tuple[int, ...],
    s: tuple[int, ...],
    d: tuple[int, ...],
    dtype: torch.dtype,
    p: tuple[int, ...] | None = None,
    has_tail: bool = False,
) -> tuple[int, int, int, int, bool, bool, tuple[int, ...], bool]:
    """Selects (bm, bn, bk, num_sg, w_blk8, use_halo, spatial_tiles, fuse_sw), unlocking W_BLK8 for small C_in."""
    smem_limit = _MAX_HALO_SMEM_BYTES
    if p is not None and s[-1] > 1 and dtype == torch.float32 and c_out_g % 8 == 0:
        n_act, ov_w = _active_w_info(k[-1], s[-1], p[-1], d[-1])
        if n_act >= 1:
            overhangs = _halo_overhangs(k, s, d)
            overhangs[-1] = ov_w
            cands: list[tuple[int, int, int, int]] = []
            bk_pref = 16 if (c_in_g >= 16 and c_in_g % 16 == 0) else 8
            if c_out_g >= 128 and c_out_g % 128 == 0:
                cands.extend([(128, 64, bk_pref, 8), (64, 64, bk_pref, 8), (64, 64, bk_pref, 4)])
            elif c_out_g >= 64 and c_out_g % 64 == 0:
                if n_act == 1:
                    cands.extend([(64, 64, bk_pref, 4), (64, 64, bk_pref, 8)])
                elif s[-1] >= 4:
                    cands.extend([(32, 64, bk_pref, 4), (64, 32, bk_pref, 4), (64, 64, bk_pref, 8)])
                else:
                    cands.extend([(64, 64, bk_pref, 8), (64, 32, bk_pref, 4)])
            else:
                bm_s = max(8, -(-c_out_g // 8) * 8)
                cands.extend([(bm_s, 64, bk_pref, 4), (bm_s, 64, bk_pref, 8), (bm_s, 32, bk_pref, 4)])
            max_tw = 32 if (ndim == 2 and s[-1] >= 4) else None
            for bm_c, bn_c, bk_c, nsg_c in cands:
                if c_out_g % bm_c != 0 or (nsg_c * 32) % bn_c != 0:
                    continue
                frags = _sg_frags(bm_c, bn_c, nsg_c)
                if frags is None or n_act * frags > 24:
                    continue
                for bk_try in (bk_c, 8):
                    if bk_try > bk_c:
                        continue
                    picked = _choose_halo_tile(ndim, bn_c, bk_try, overhangs, max_tw=max_tw)
                    if picked is not None and 8 * n_act * bn_c * 4 <= smem_limit:
                        return bm_c, bn_c, bk_try, nsg_c, True, True, picked[0], True

    if c_out_g >= 128:
        bm, bn, num_sg = 128, 64, 8
        bk = 16 if (c_in_g >= 16 and c_in_g % 16 == 0) else 8
    elif c_out_g >= 64:
        bm, bn, num_sg = 64, 64, 4
        bk = 16 if (c_in_g >= 16 and c_in_g % 16 == 0) else 8
    else:
        bm = max(8, -(-c_out_g // 8) * 8)
        bn = 64
        num_sg = 4
        bk = 16 if (c_in_g >= 16 and c_in_g % 16 == 0) else 8

    w_blk8 = dtype == torch.float32 and c_out_g % bm == 0
    use_halo = False
    tiles = (1,) * (ndim - 1) + (bn,)
    if w_blk8:
        overhangs = _halo_overhangs(k, s, d)
        picked = _choose_halo_tile(ndim, bn, bk, overhangs)
        if picked is None and bk > 8:
            picked = _choose_halo_tile(ndim, bn, 8, overhangs)
            if picked is not None:
                bk = 8
        if picked is not None and (picked[1] + (bm * 4 if has_tail else 0)) <= 32768:
            tiles = picked[0]
            use_halo = True
    return bm, bn, bk, num_sg, w_blk8, use_halo, tiles, False


class PreparedConvTranspose:
    """Host-prepared packed weights and launch metadata (safe to prepare before entering command_batch)."""

    __slots__ = (
        "ndim", "stride", "padding", "output_padding", "dilation", "groups",
        "c_in", "c_out", "k", "w_arg", "b_arg", "has_bias", "sched",
    )

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        ndim: int,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        has_tail: bool = False,
    ) -> None:
        self.ndim = int(ndim)
        self.groups = int(groups)
        self.c_in = int(weight.shape[0])
        self.c_out = int(weight.shape[1]) * self.groups
        self.k = tuple(int(v) for v in weight.shape[2:])
        self.stride = as_tuple(stride, self.ndim)
        self.padding = as_tuple(padding, self.ndim)
        self.output_padding = as_tuple(output_padding, self.ndim)
        self.dilation = as_tuple(dilation, self.ndim)
        self.sched = select_schedule_l2(
            self.ndim,
            self.c_in // self.groups,
            self.c_out // self.groups,
            self.k,
            self.stride,
            self.dilation,
            weight.dtype,
            p=self.padding,
            has_tail=has_tail,
        )
        bk = self.sched[2]
        w_blk8 = self.sched[4]
        self.w_arg = pack_weight_blk8_pad(weight, self.groups, bk) if w_blk8 else weight.contiguous()
        self.has_bias = bias is not None
        self.b_arg = bias.contiguous() if bias is not None else self.w_arg


def launch_conv_transpose_fused(
    x: torch.Tensor,
    prep: PreparedConvTranspose,
    *,
    epilogue: tuple[Any, ...] = (),
    eb0: torch.Tensor | None = None,
    eb1: torch.Tensor | None = None,
    es0: float = 0.0,
    es1: float = 0.0,
    tail_kind: str = "none",
    tail_scale: float = 1.0,
    allow_winograd: bool = False,
    raw_weight: torch.Tensor | None = None,
    raw_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatches one fused ConvTranspose{2,3}d Metal kernel with optional epilogue DAG and spatial global tail."""
    x_c = x if x.is_contiguous() else x.contiguous()
    ndim = prep.ndim
    n = int(x_c.shape[0])
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [
        conv_transpose_output_size(
            spatial_in[i], prep.k[i], prep.stride[i], prep.padding[i], prep.dilation[i], prep.output_padding[i]
        )
        for i in range(ndim)
    ]

    if (
        allow_winograd
        and not epilogue
        and tail_kind == "none"
        and raw_weight is not None
        and ndim == 2
        and prep.k == (3, 3)
        and prep.stride == (1, 1)
        and prep.dilation == (1, 1)
        and prep.output_padding == (0, 0)
        and prep.groups == 1
        and prep.padding[0] <= 2
        and prep.padding[1] <= 2
        and x_c.dtype == torch.float32
        and raw_weight.dtype == torch.float32
        and (raw_bias is None or raw_bias.dtype == torch.float32)
        and prep.c_in >= 8
        and prep.c_out >= 16
        and spatial_out[0] * spatial_out[1] >= 256
    ):
        from _conv_std_common import conv2d_winograd_3x3

        return conv2d_winograd_3x3(
            x_c, raw_weight, raw_bias, padding=(2 - prep.padding[0], 2 - prep.padding[1]),
            transposed=True, use_staged_4x3=(prep.c_in <= 128 and prep.padding == (0, 0)),
        )

    bm, bn, bk, num_sg, w_blk8, use_halo, tiles, fuse_sw = prep.sched
    q_dims = [-(-spatial_out[i] // prep.stride[i]) for i in range(ndim)]
    phases = math.prod(prep.stride[:-1]) if fuse_sw else math.prod(prep.stride)
    if use_halo:
        grid_x = math.prod(m1triton.cdiv(qd, td) for qd, td in zip(q_dims, tiles))
    else:
        grid_x = m1triton.cdiv(math.prod(q_dims), bn)
    grid = (grid_x, m1triton.cdiv(prep.c_out // prep.groups, bm), n * prep.groups * phases)

    if tail_kind != "none":
        out = torch.empty((n, prep.c_out, phases * grid_x), dtype=x_c.dtype, device=x_c.device)
    else:
        out = torch.empty((n, prep.c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    eb0_arg = (eb0 if eb0.is_contiguous() else eb0.contiguous()) if eb0 is not None else prep.w_arg
    eb1_arg = (eb1 if eb1.is_contiguous() else eb1.contiguous()) if eb1 is not None else prep.w_arg

    meta: dict[str, object] = dict(
        C_IN=prep.c_in,
        C_OUT=prep.c_out,
        GROUPS=prep.groups,
        HAS_BIAS=prep.has_bias,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=num_sg,
        W_BLK8=int(w_blk8),
        HALO=int(use_halo),
        FUSE_SW=int(fuse_sw),
        EPILOGUE=tuple(epilogue),
        TAIL_KIND=str(tail_kind),
        TAIL_SCALE=float(tail_scale),
        **_launch_opts(),
    )
    for i, tag in enumerate(_DIM_TAGS[ndim]):
        meta[f"K_{tag}"] = prep.k[i]
        meta[f"S_{tag}"] = prep.stride[i]
        meta[f"P_{tag}"] = prep.padding[i]
        meta[f"D_{tag}"] = prep.dilation[i]
        meta[f"OP_{tag}"] = prep.output_padding[i]
        meta[f"TILE_{tag}"] = tiles[i]

    _FUSED_KERNELS[ndim][grid](
        x_c, prep.w_arg, out, prep.b_arg,
        eb0_arg, eb1_arg, float(es0), float(es1),
        n, *spatial_in, *spatial_out,
        **meta,
    )
    return out


def conv_transpose_nd_epilogue(
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
    epilogue: tuple[Any, ...] = (),
    eb0: torch.Tensor | None = None,
    eb1: torch.Tensor | None = None,
    es0: float = 0.0,
    es1: float = 0.0,
    tail_kind: str = "none",
    tail_scale: float = 1.0,
    allow_winograd: bool = False,
) -> torch.Tensor:
    prep = PreparedConvTranspose(
        weight, bias, ndim=ndim, stride=stride, padding=padding,
        output_padding=output_padding, dilation=dilation, groups=groups,
        has_tail=(tail_kind != "none"),
    )
    return launch_conv_transpose_fused(
        x, prep, epilogue=epilogue, eb0=eb0, eb1=eb1, es0=es0, es1=es1,
        tail_kind=tail_kind, tail_scale=tail_scale, allow_winograd=allow_winograd,
        raw_weight=weight, raw_bias=bias,
    )


@m1triton.jit
def bn_fold_affine_kernel(
    w_ptr,
    b_ptr,
    rm_ptr,
    rv_ptr,
    scale_ptr,
    shift_ptr,
    C,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(rm_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(rv_ptr + offs, mask=mask, other=1.0)
    sc = w / tl.sqrt(rv + EPS)
    sh = b - rm * sc
    tl.store(scale_ptr + offs, sc, mask=mask)
    tl.store(shift_ptr + offs, sh, mask=mask)


def bn_fold_affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes live inference BatchNorm affine (scale, shift) on GPU with zero cross-call caching."""
    c = int(weight.numel())
    scale = torch.empty((c,), dtype=torch.float32, device=weight.device)
    shift = torch.empty((c,), dtype=torch.float32, device=weight.device)
    block = 128 if c <= 128 else 256
    bn_fold_affine_kernel[(m1triton.cdiv(c, block),)](
        weight.contiguous(),
        bias.contiguous(),
        running_mean.contiguous(),
        running_var.contiguous(),
        scale,
        shift,
        c,
        EPS=float(eps),
        BLOCK=block,
        **_launch_opts(),
    )
    return scale, shift


@m1triton.jit
def reduce_partials_sum_kernel(
    partials_ptr,
    bias_ptr,
    out_ptr,
    C,
    P,
    ADD_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = partials_ptr + row.to(tl.int64) * P.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, P, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < P
        acc = acc + tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=0.0)
    res = tl.sum(acc, axis=0)
    if ADD_BIAS:
        ch = row % C
        res = res + tl.load(bias_ptr + ch.to(tl.int64))
    tl.store(out_ptr + row.to(tl.int64), res)


def finalize_spatial_partials_sum(
    partials: torch.Tensor,
    out_shape: tuple[int, ...],
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    n, c, p = int(partials.shape[0]), int(partials.shape[1]), int(partials.shape[2])
    out = torch.empty(out_shape, dtype=partials.dtype, device=partials.device)
    block_k = 256 if p <= 256 else 1024
    b_arg = bias.contiguous().view(-1) if bias is not None else partials
    reduce_partials_sum_kernel[(n * c,)](
        partials,
        b_arg,
        out,
        c,
        p,
        ADD_BIAS=int(bias is not None),
        BLOCK_K=block_k,
        **_launch_opts(),
    )
    return out


@m1triton.jit
def row_logsumexp_scale_kernel(
    x_ptr,
    out_ptr,
    C,
    SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_base = x_ptr + row * C.to(tl.int64)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    v = tl.load(row_base + offs.to(tl.int64), mask=mask, other=-1.0e38)
    m = tl.max(v, axis=0)
    ex = tl.where(mask, tl.exp(v - m), 0.0)
    l = tl.sum(ex, axis=0)
    res = (m + tl.log(l)) * SCALE
    tl.store(out_ptr + row, res)


@m1triton.jit
def row_max_kernel(x_ptr, out_ptr, C, BLOCK_C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    row_base = x_ptr + row * C.to(tl.int64)
    ar = tl.arange(0, BLOCK_C)
    acc = tl.full((BLOCK_C,), -1.0e38, dtype=tl.float32)
    for k in range(0, C, BLOCK_C):
        offs = k + ar
        mask = offs < C
        v = tl.load(row_base + offs.to(tl.int64), mask=mask, other=-1.0e38)
        acc = tl.maximum(acc, v)
    tl.store(out_ptr + row, tl.max(acc, axis=0))


@m1triton.jit
def row_lse_from_max_scale_kernel(x_ptr, max_ptr, out_ptr, C, SCALE: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    row_base = x_ptr + row * C.to(tl.int64)
    m = tl.load(max_ptr + row)
    ar = tl.arange(0, BLOCK_C)
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for k in range(0, C, BLOCK_C):
        offs = k + ar
        mask = offs < C
        v = tl.load(row_base + offs.to(tl.int64), mask=mask, other=0.0)
        acc = acc + tl.where(mask, tl.exp(v - m), 0.0)
    res = (m + tl.log(tl.sum(acc, axis=0))) * SCALE
    tl.store(out_ptr + row, res)


def row_logsumexp_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    n, c = int(x.shape[0]), int(x.shape[1])
    out = torch.empty((n, 1), dtype=x.dtype, device=x.device)
    if c <= 1024:
        block_c = max(32, _next_pow2(c))
        row_logsumexp_scale_kernel[(n,)](
            x, out, c, SCALE=float(scale), BLOCK_C=block_c, **_launch_opts()
        )
        return out
    m_buf = torch.empty((n,), dtype=x.dtype, device=x.device)
    row_max_kernel[(n,)](x, m_buf, c, BLOCK_C=1024, **_launch_opts())
    row_lse_from_max_scale_kernel[(n,)](x, m_buf, out, c, SCALE=float(scale), BLOCK_C=1024, **_launch_opts())
    return out


@m1triton.jit
def plane_mean_kernel(
    x_ptr,
    mean_ptr,
    S,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base_in = x_ptr + row * S.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, S, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < S
        acc = acc + tl.load(base_in + k_offs.to(tl.int64), mask=k_mask, other=0.0)
    mean = tl.sum(acc, axis=0) / S.to(tl.float32)
    tl.store(mean_ptr + row, mean)


@m1triton.jit
def plane_sub_mean_vec4_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    S,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_s = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    s_i64 = S.to(tl.int64)
    offs = pid_s.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < s_i64
    idx = row * s_i64 + offs
    m = tl.load(mean_ptr + row)
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, v - m, mask=mask)


def subtract_spatial_mean(x: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    s = x_c.numel() // max(n * c, 1)
    means = torch.empty((n * c,), dtype=torch.float32, device=x_c.device)
    out = torch.empty_like(x_c)
    block_k = 256 if s <= 256 else 1024
    opts = _launch_opts()
    plane_mean_kernel[(n * c,)](x_c, means, s, BLOCK_K=block_k, **opts)
    plane_sub_mean_vec4_kernel[(m1triton.cdiv(s, 1024), n * c)](x_c, means, out, s, BLOCK=1024, **opts)
    return out


@m1triton.jit
def layernorm_lastdim_gelu_scale_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    W,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    w_i64 = W.to(tl.int64)
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    offs_i64 = offs.to(tl.int64)
    x = tl.load(x_ptr + row * w_i64 + offs_i64, mask=mask, other=0.0)
    inv_w = 1.0 / W.to(tl.float32)
    mean = tl.sum(x, axis=0) * inv_w
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) * inv_w
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(w_ptr + offs_i64, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs_i64, mask=mask, other=0.0)
    y = diff * rstd * g + b
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    if SCALE != 1.0:
        y = y * SCALE
    tl.store(out_ptr + row * w_i64 + offs_i64, y, mask=mask)


def layernorm_gelu_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
    scale: float,
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    w_size = int(weight.numel())
    if w_size <= 1024 and w_size == int(x_c.shape[-1]):
        rows = x_c.numel() // max(w_size, 1)
        out = torch.empty_like(x_c)
        block_w = max(32, _next_pow2(w_size))
        layernorm_lastdim_gelu_scale_kernel[(rows,)](
            x_c, weight, bias, out, w_size, EPS=float(eps), SCALE=float(scale), BLOCK_W=block_w, **_launch_opts()
        )
        return out
    y = layer_norm(x_c, weight, bias, eps=eps)
    out = torch.empty_like(y)
    n = y.numel()
    gelu_scale_1d_kernel[(m1triton.cdiv(n, 1024),)](y, out, n, SCALE=float(scale), BLOCK=1024, **_launch_opts())
    return out


@m1triton.jit
def gelu_scale_1d_kernel(x_ptr, out_ptr, N, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N.to(tl.int64)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476)) * SCALE
    tl.store(out_ptr + offs, y, mask=mask)


@m1triton.jit
def layernorm_avgpool3d_2x2x2_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    D_p,
    H_p,
    W_p,
    EPS: tl.constexpr,
    BLOCK_WP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    hp_i64 = H_p.to(tl.int64)
    dp_i64 = D_p.to(tl.int64)
    wp_i64 = W_p.to(tl.int64)
    oh = pid % hp_i64
    t1 = pid // hp_i64
    od = t1 % dp_i64
    nc = t1 // dp_i64

    hin_i64 = H_in.to(tl.int64)
    win_i64 = W_in.to(tl.int64)
    din_i64 = D_in.to(tl.int64)
    plane_base = nc * (din_i64 * hin_i64 * win_i64)

    offs = tl.arange(0, BLOCK_WP)
    mask = offs < W_p
    e_idx = (offs * 2).to(tl.int64)
    o_idx = e_idx + 1
    g0 = tl.load(w_ptr + e_idx, mask=mask, other=1.0)
    g1 = tl.load(w_ptr + o_idx, mask=mask, other=1.0)
    b0 = tl.load(b_ptr + e_idx, mask=mask, other=0.0)
    b1 = tl.load(b_ptr + o_idx, mask=mask, other=0.0)
    inv_w = 1.0 / W_in.to(tl.float32)

    id0 = od * 2
    ih0 = oh * 2

    rb00 = plane_base + (id0 * hin_i64 + ih0) * win_i64
    x0 = tl.load(x_ptr + rb00 + e_idx, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + rb00 + o_idx, mask=mask, other=0.0)
    m00 = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0)) * inv_w
    d0 = tl.where(mask, x0 - m00, 0.0)
    d1 = tl.where(mask, x1 - m00, 0.0)
    r00 = 1.0 / tl.sqrt((tl.sum(d0 * d0, axis=0) + tl.sum(d1 * d1, axis=0)) * inv_w + EPS)
    s00 = (d0 * r00 * g0 + b0) + (d1 * r00 * g1 + b1)

    rb01 = plane_base + (id0 * hin_i64 + (ih0 + 1)) * win_i64
    x0 = tl.load(x_ptr + rb01 + e_idx, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + rb01 + o_idx, mask=mask, other=0.0)
    m01 = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0)) * inv_w
    d0 = tl.where(mask, x0 - m01, 0.0)
    d1 = tl.where(mask, x1 - m01, 0.0)
    r01 = 1.0 / tl.sqrt((tl.sum(d0 * d0, axis=0) + tl.sum(d1 * d1, axis=0)) * inv_w + EPS)
    s01 = (d0 * r01 * g0 + b0) + (d1 * r01 * g1 + b1)

    rb10 = plane_base + ((id0 + 1) * hin_i64 + ih0) * win_i64
    x0 = tl.load(x_ptr + rb10 + e_idx, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + rb10 + o_idx, mask=mask, other=0.0)
    m10 = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0)) * inv_w
    d0 = tl.where(mask, x0 - m10, 0.0)
    d1 = tl.where(mask, x1 - m10, 0.0)
    r10 = 1.0 / tl.sqrt((tl.sum(d0 * d0, axis=0) + tl.sum(d1 * d1, axis=0)) * inv_w + EPS)
    s10 = (d0 * r10 * g0 + b0) + (d1 * r10 * g1 + b1)

    rb11 = plane_base + ((id0 + 1) * hin_i64 + (ih0 + 1)) * win_i64
    x0 = tl.load(x_ptr + rb11 + e_idx, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + rb11 + o_idx, mask=mask, other=0.0)
    m11 = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0)) * inv_w
    d0 = tl.where(mask, x0 - m11, 0.0)
    d1 = tl.where(mask, x1 - m11, 0.0)
    r11 = 1.0 / tl.sqrt((tl.sum(d0 * d0, axis=0) + tl.sum(d1 * d1, axis=0)) * inv_w + EPS)
    s11 = (d0 * r11 * g0 + b0) + (d1 * r11 * g1 + b1)

    y = (s00 + s01 + s10 + s11) * 0.125
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    tl.store(out_ptr + pid * wp_i64 + offs.to(tl.int64), y, mask=mask)


def layernorm_avgpool3d_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
    pool_k: tuple[int, int, int],
    pool_s: tuple[int, int, int],
    pool_p: tuple[int, int, int],
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c, d_in, h_in, w_in = (int(v) for v in x_c.shape)
    d_p = pool_output_size(d_in, pool_k[0], pool_s[0], pool_p[0], 1, False)
    h_p = pool_output_size(h_in, pool_k[1], pool_s[1], pool_p[1], 1, False)
    w_p = pool_output_size(w_in, pool_k[2], pool_s[2], pool_p[2], 1, False)
    if (
        int(weight.numel()) == w_in
        and w_in % 2 == 0
        and w_p <= 1024
        and pool_k == (2, 2, 2)
        and pool_s == (2, 2, 2)
        and pool_p == (0, 0, 0)
    ):
        out = torch.empty((n, c, d_p, h_p, w_p), dtype=x_c.dtype, device=x_c.device)
        block_wp = max(32, _next_pow2(w_p))
        layernorm_avgpool3d_2x2x2_gelu_kernel[(n * c * d_p * h_p,)](
            x_c, weight, bias, out,
            d_in, h_in, w_in, d_p, h_p, w_p,
            EPS=float(eps), BLOCK_WP=block_wp, **_launch_opts(),
        )
        return out
    y = layer_norm(x_c, weight, bias, eps=eps)
    y = pool_nd(y, kind="avg", ndim=3, kernel_size=pool_k, stride=pool_s, padding=pool_p)
    out = torch.empty_like(y)
    tot = y.numel()
    gelu_scale_1d_kernel[(m1triton.cdiv(tot, 1024),)](y, out, tot, SCALE=1.0, BLOCK=1024, **_launch_opts())
    return out


@m1triton.jit
def maxpool2d_2x2_hardtanh_mean_tanh_kernel(
    x_ptr,
    out_ptr,
    H_in,
    W_in,
    H_p,
    W_p,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    HMIN: tl.constexpr,
    HMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0).to(tl.int64)
    hin_i64 = H_in.to(tl.int64)
    win_i64 = W_in.to(tl.int64)
    base = x_ptr + nc * (hin_i64 * win_i64)
    total_p = H_p * W_p
    ar = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p0 in range(0, total_p, BLOCK):
        p_idx = p0 + ar
        mask = p_idx < total_p
        oh = p_idx // W_p
        ow = p_idx - oh * W_p
        ih0 = (oh * S_H).to(tl.int64)
        iw0 = (ow * S_W).to(tl.int64)
        r0 = ih0 * win_i64 + iw0
        r1 = (ih0 + 1) * win_i64 + iw0
        v00 = tl.load(base + r0, mask=mask, other=-1.0e38)
        v01 = tl.load(base + r0 + 1, mask=mask, other=-1.0e38)
        v10 = tl.load(base + r1, mask=mask, other=-1.0e38)
        v11 = tl.load(base + r1 + 1, mask=mask, other=-1.0e38)
        mval = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        ht = tl.where(mval < HMIN, HMIN, tl.where(mval > HMAX, HMAX, mval))
        acc = acc + tl.where(mask, ht, 0.0)
    mean_v = tl.sum(acc, axis=0) / total_p.to(tl.float32)
    tl.store(out_ptr + nc, tl.tanh(mean_v))


@m1triton.jit
def plane_hardtanh_mean_tanh_kernel(
    x_ptr,
    out_ptr,
    S,
    HMIN: tl.constexpr,
    HMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0).to(tl.int64)
    base = x_ptr + nc * S.to(tl.int64)
    ar = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p0 in range(0, S, BLOCK):
        p_idx = p0 + ar
        mask = p_idx < S
        v = tl.load(base + p_idx.to(tl.int64), mask=mask, other=0.0)
        ht = tl.where(v < HMIN, HMIN, tl.where(v > HMAX, HMAX, v))
        acc = acc + tl.where(mask, ht, 0.0)
    mean_v = tl.sum(acc, axis=0) / S.to(tl.float32)
    tl.store(out_ptr + nc, tl.tanh(mean_v))


def maxpool2d_hardtanh_mean_tanh(
    x: torch.Tensor,
    *,
    k: tuple[int, int],
    s: tuple[int, int],
    p: tuple[int, int],
    d: tuple[int, int],
    hmin: float,
    hmax: float,
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c, h_in, w_in = (int(v) for v in x_c.shape)
    if k == (2, 2) and p == (0, 0) and d == (1, 1):
        h_p = pool_output_size(h_in, k[0], s[0], 0, 1, False)
        w_p = pool_output_size(w_in, k[1], s[1], 0, 1, False)
        out = torch.empty((n, c, 1, 1), dtype=x_c.dtype, device=x_c.device)
        maxpool2d_2x2_hardtanh_mean_tanh_kernel[(n * c,)](
            x_c, out, h_in, w_in, h_p, w_p,
            S_H=s[0], S_W=s[1], HMIN=float(hmin), HMAX=float(hmax), BLOCK=256,
            **_launch_opts(),
        )
        return out
    y = pool_nd(x_c, kind="max", ndim=2, kernel_size=k, stride=s, padding=p, dilation=d)
    s_len = int(y.shape[2]) * int(y.shape[3])
    out = torch.empty((n, c, 1, 1), dtype=x_c.dtype, device=x_c.device)
    plane_hardtanh_mean_tanh_kernel[(n * c,)](
        y, out, s_len, HMIN=float(hmin), HMAX=float(hmax), BLOCK=256, **_launch_opts()
    )
    return out


@m1triton.jit
def maxpool3d_2x2x2_gap_clamp_kernel(
    x_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    D_p,
    H_p,
    W_p,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0).to(tl.int64)
    din_i64 = D_in.to(tl.int64)
    hin_i64 = H_in.to(tl.int64)
    win_i64 = W_in.to(tl.int64)
    hw_in = hin_i64 * win_i64
    base = x_ptr + nc * (din_i64 * hw_in)
    hwp = H_p * W_p
    total_p = D_p * hwp
    ar = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p0 in range(0, total_p, BLOCK):
        p_idx = p0 + ar
        mask = p_idx < total_p
        od = p_idx // hwp
        rem = p_idx - od * hwp
        oh = rem // W_p
        ow = rem - oh * W_p
        id0 = (od * S_D).to(tl.int64)
        ih0 = (oh * S_H).to(tl.int64)
        iw0 = (ow * S_W).to(tl.int64)
        o00 = (id0 * hin_i64 + ih0) * win_i64 + iw0
        o01 = o00 + win_i64
        o10 = o00 + hw_in
        o11 = o01 + hw_in
        m0 = tl.maximum(
            tl.maximum(tl.load(base + o00, mask=mask, other=-1.0e38), tl.load(base + o00 + 1, mask=mask, other=-1.0e38)),
            tl.maximum(tl.load(base + o01, mask=mask, other=-1.0e38), tl.load(base + o01 + 1, mask=mask, other=-1.0e38)),
        )
        m1 = tl.maximum(
            tl.maximum(tl.load(base + o10, mask=mask, other=-1.0e38), tl.load(base + o10 + 1, mask=mask, other=-1.0e38)),
            tl.maximum(tl.load(base + o11, mask=mask, other=-1.0e38), tl.load(base + o11 + 1, mask=mask, other=-1.0e38)),
        )
        acc = acc + tl.where(mask, tl.maximum(m0, m1), 0.0)
    mean_v = tl.sum(acc, axis=0) / total_p.to(tl.float32)
    clamped = tl.where(mean_v < CMIN, CMIN, tl.where(mean_v > CMAX, CMAX, mean_v))
    tl.store(out_ptr + nc, clamped)


@m1triton.jit
def plane_gap_clamp_kernel(
    x_ptr,
    out_ptr,
    S,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0).to(tl.int64)
    base = x_ptr + nc * S.to(tl.int64)
    ar = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p0 in range(0, S, BLOCK):
        p_idx = p0 + ar
        mask = p_idx < S
        acc = acc + tl.load(base + p_idx.to(tl.int64), mask=mask, other=0.0)
    mean_v = tl.sum(acc, axis=0) / S.to(tl.float32)
    clamped = tl.where(mean_v < CMIN, CMIN, tl.where(mean_v > CMAX, CMAX, mean_v))
    tl.store(out_ptr + nc, clamped)


def maxpool3d_gap_clamp(
    x: torch.Tensor,
    *,
    k: tuple[int, int, int],
    s: tuple[int, int, int],
    p: tuple[int, int, int],
    d: tuple[int, int, int],
    cmin: float,
    cmax: float,
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c, d_in, h_in, w_in = (int(v) for v in x_c.shape)
    if k == (2, 2, 2) and p == (0, 0, 0) and d == (1, 1, 1):
        d_p = pool_output_size(d_in, k[0], s[0], 0, 1, False)
        h_p = pool_output_size(h_in, k[1], s[1], 0, 1, False)
        w_p = pool_output_size(w_in, k[2], s[2], 0, 1, False)
        out = torch.empty((n, c, 1, 1, 1), dtype=x_c.dtype, device=x_c.device)
        maxpool3d_2x2x2_gap_clamp_kernel[(n * c,)](
            x_c, out, d_in, h_in, w_in, d_p, h_p, w_p,
            S_D=s[0], S_H=s[1], S_W=s[2], CMIN=float(cmin), CMAX=float(cmax), BLOCK=256,
            **_launch_opts(),
        )
        return out
    y = pool_nd(x_c, kind="max", ndim=3, kernel_size=k, stride=s, padding=p, dilation=d)
    s_len = int(y.shape[2]) * int(y.shape[3]) * int(y.shape[4])
    out = torch.empty((n, c, 1, 1, 1), dtype=x_c.dtype, device=x_c.device)
    plane_gap_clamp_kernel[(n * c,)](
        y, out, s_len, CMIN=float(cmin), CMAX=float(cmax), BLOCK=256, **_launch_opts()
    )
    return out


@m1triton.jit
def avgpool3d_bias_scale_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    C,
    D_in,
    H_in,
    W_in,
    D_p,
    H_p,
    W_p,
    scale2,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_p = tl.program_id(0)
    nc = tl.program_id(1).to(tl.int64)
    ch = nc % C.to(tl.int64)
    din_i64 = D_in.to(tl.int64)
    hin_i64 = H_in.to(tl.int64)
    win_i64 = W_in.to(tl.int64)
    base_in = x_ptr + nc * (din_i64 * hin_i64 * win_i64)
    hwp = H_p * W_p
    total_p = D_p * hwp
    base_out = out_ptr + nc * total_p.to(tl.int64)

    p_idx = pid_p * BLOCK + tl.arange(0, BLOCK)
    mask = p_idx < total_p
    od = p_idx // hwp
    rem = p_idx - od * hwp
    oh = rem // W_p
    ow = rem - oh * W_p
    id0 = od * S_D
    ih0 = oh * S_H
    iw0 = ow * S_W
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kd in range(0, K_D):
        id_c = (id0 + kd).to(tl.int64)
        for kh in range(0, K_H):
            ih_c = (ih0 + kh).to(tl.int64)
            row_off = (id_c * hin_i64 + ih_c) * win_i64
            for kw in range(0, K_W):
                iw_c = (iw0 + kw).to(tl.int64)
                acc = acc + tl.load(base_in + row_off + iw_c, mask=mask, other=0.0)
    inv_k = 1.0 / float(K_D * K_H * K_W)
    bc = tl.load(bias_ptr + ch)
    y = (acc * inv_k + bc) * scale2
    tl.store(base_out + p_idx.to(tl.int64), y, mask=mask)


def avgpool3d_bias_scale(
    x: torch.Tensor,
    bias: torch.Tensor,
    scale2: float,
    *,
    k: tuple[int, int, int],
    s: tuple[int, int, int],
    p: tuple[int, int, int],
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c, d_in, h_in, w_in = (int(v) for v in x_c.shape)
    b_flat = bias.contiguous().view(-1)
    if p == (0, 0, 0):
        d_p = pool_output_size(d_in, k[0], s[0], 0, 1, False)
        h_p = pool_output_size(h_in, k[1], s[1], 0, 1, False)
        w_p = pool_output_size(w_in, k[2], s[2], 0, 1, False)
        out = torch.empty((n, c, d_p, h_p, w_p), dtype=x_c.dtype, device=x_c.device)
        total_p = d_p * h_p * w_p
        avgpool3d_bias_scale_kernel[(m1triton.cdiv(total_p, 256), n * c)](
            x_c, b_flat, out, c, d_in, h_in, w_in, d_p, h_p, w_p, float(scale2),
            K_D=k[0], K_H=k[1], K_W=k[2], S_D=s[0], S_H=s[1], S_W=s[2], BLOCK=256,
            **_launch_opts(),
        )
        return out
    y = pool_nd(x_c, kind="avg", ndim=3, kernel_size=k, stride=s, padding=p)
    d_p, h_p, w_p = int(y.shape[2]), int(y.shape[3]), int(y.shape[4])
    out = torch.empty_like(y)
    total_p = d_p * h_p * w_p
    avgpool3d_bias_scale_kernel[(m1triton.cdiv(total_p, 256), n * c)](
        y, b_flat, out, c, d_p, h_p, w_p, d_p, h_p, w_p, float(scale2),
        K_D=1, K_H=1, K_W=1, S_D=1, S_H=1, S_W=1, BLOCK=256,
        **_launch_opts(),
    )
    return out


@m1triton.jit
def depth_mean_add_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    C,
    D,
    HW,
    BLOCK_HW: tl.constexpr,
):
    tlx.vec_width(4)
    pid_hw = tl.program_id(0)
    nc = tl.program_id(1).to(tl.int64)
    ch = nc % C.to(tl.int64)
    d_i64 = D.to(tl.int64)
    hw_i64 = HW.to(tl.int64)
    offs = pid_hw.to(tl.int64) * BLOCK_HW + tl.arange(0, BLOCK_HW).to(tl.int64)
    mask = offs < hw_i64
    p = x_ptr + nc * (d_i64 * hw_i64) + offs
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for _ in range(0, D):
        acc = acc + tl.load(p, mask=mask, other=0.0)
        p = p + hw_i64
    bc = tl.load(bias_ptr + ch)
    y = acc / D.to(tl.float32) + bc
    tl.store(out_ptr + nc * hw_i64 + offs, y, mask=mask)


def depth_mean_add_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c, d, h, w = (int(v) for v in x_c.shape)
    hw = h * w
    out = torch.empty((n, c, 1, h, w), dtype=x_c.dtype, device=x_c.device)
    depth_mean_add_bias_kernel[(m1triton.cdiv(hw, 256), n * c)](
        x_c, bias.contiguous().view(-1), out, c, d, hw, BLOCK_HW=256, **_launch_opts()
    )
    return out


@m1triton.jit
def channel_softmax_post_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    C,
    S,
    SCALE: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    tlx.vec_width(4)
    pid_s = tl.program_id(0)
    n_idx = tl.program_id(1).to(tl.int64)
    c_i64 = C.to(tl.int64)
    s_i64 = S.to(tl.int64)
    offs = pid_s.to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
    mask = offs < s_i64
    base = n_idx * (c_i64 * s_i64) + offs

    m = tl.full((BLOCK_S,), -1.0e38, dtype=tl.float32)
    l = tl.zeros((BLOCK_S,), dtype=tl.float32)
    p = x_ptr + base
    for _ in range(0, C):
        v = tl.load(p, mask=mask, other=-1.0e38)
        mn = tl.maximum(m, v)
        l = l * tl.exp(m - mn) + tl.exp(v - mn)
        m = mn
        p = p + s_i64
    inv_l = 1.0 / l

    if MODE == 4:
        best = tl.full((BLOCK_S,), -1.0e38, dtype=tl.float32)
        p = x_ptr + base
        for c in range(0, C):
            v = tl.load(p, mask=mask, other=0.0)
            prob = tl.exp(v - m) * inv_l
            u = prob - tl.load(bias_ptr + c.to(tl.int64))
            sw = u / (1.0 + tl.exp(-u))
            best = tl.maximum(best, sw)
            p = p + s_i64
        tl.store(out_ptr + n_idx * s_i64 + offs, best, mask=mask)
    else:
        p = x_ptr + base
        po = out_ptr + base
        for c in range(0, C):
            v = tl.load(p, mask=mask, other=0.0)
            prob = tl.exp(v - m) * inv_l
            if MODE == 1:
                y = tl.tanh(prob) * SCALE
            elif MODE == 2:
                y = 1.0 / (1.0 + tl.exp(-prob))
            else:
                bc = tl.load(bias_ptr + c.to(tl.int64))
                z = (prob + bc) * SCALE
                y = 1.0 / (1.0 + tl.exp(-z))
            tl.store(po, y, mask=mask)
            p = p + s_i64
            po = po + s_i64


def channel_softmax_post(
    x: torch.Tensor,
    *,
    mode: int,
    bias: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Coalesced dim=1 softmax fused with post-op: 1=tanh*scale, 2=sigmoid, 3=sigmoid((p+b)*scale), 4=max_c(swish(p-b))."""
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(v) for v in x_c.shape[2:])
    s = math.prod(spatial)
    out = torch.empty((n, *spatial), dtype=x_c.dtype, device=x_c.device) if mode == 4 else torch.empty_like(x_c)
    b_arg = bias.contiguous().view(-1) if bias is not None else x_c
    block_s = 256 if s >= 256 else 128
    channel_softmax_post_kernel[(m1triton.cdiv(s, block_s), n)](
        x_c, b_arg, out, c, s, SCALE=float(scale), MODE=int(mode), BLOCK_S=block_s, **_launch_opts()
    )
    return out


@m1triton.jit
def channel_lse_sig_sub_clamp_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    C,
    S,
    BLOCK_S: tl.constexpr,
):
    tlx.vec_width(4)
    pid_s = tl.program_id(0)
    n_idx = tl.program_id(1).to(tl.int64)
    c_i64 = C.to(tl.int64)
    s_i64 = S.to(tl.int64)
    offs = pid_s.to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
    mask = offs < s_i64
    p = x_ptr + n_idx * (c_i64 * s_i64) + offs
    m = tl.full((BLOCK_S,), -1.0e38, dtype=tl.float32)
    l = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for _ in range(0, C):
        v = tl.load(p, mask=mask, other=-1.0e38)
        mn = tl.maximum(m, v)
        l = l * tl.exp(m - mn) + tl.exp(v - mn)
        m = mn
        p = p + s_i64
    lse = m + tl.log(l)
    y = lse * (1.0 / (1.0 + tl.exp(-(lse + 3.0)))) / 6.0
    y = y - tl.load(bias_ptr)
    y = tl.where(y < -1.0, -1.0, tl.where(y > 1.0, 1.0, y))
    tl.store(out_ptr + n_idx * s_i64 + offs, y, mask=mask)


def channel_lse_sig_sub_clamp(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(v) for v in x_c.shape[2:])
    s = math.prod(spatial)
    out = torch.empty((n, 1, *spatial), dtype=x_c.dtype, device=x_c.device)
    channel_lse_sig_sub_clamp_kernel[(m1triton.cdiv(s, 256), n)](
        x_c, bias.contiguous().view(-1), out, c, s, BLOCK_S=256, **_launch_opts()
    )
    return out


@m1triton.jit
def mul_channel_scale_vec4_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    C,
    S,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_s = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    ch = row % C.to(tl.int64)
    s_i64 = S.to(tl.int64)
    offs = pid_s.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < s_i64
    idx = row * s_i64 + offs
    sc = tl.load(scale_ptr + ch)
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, v * sc, mask=mask)


def spatial_softmax_mul_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    s = x_c.numel() // max(n * c, 1)
    sm = torch.empty_like(x_c)
    out = torch.empty_like(x_c)
    opts = _launch_opts()
    online_softmax_kernel[(n * c, 1, 1)](x_c, sm, n * c, s, False, **opts)
    mul_channel_scale_vec4_kernel[(m1triton.cdiv(s, 1024), n * c)](
        sm, scale.contiguous().view(-1), out, c, s, BLOCK=1024, **opts
    )
    return out


def gelu_1d(x: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    gelu_scale_1d_kernel[(m1triton.cdiv(n, 1024),)](x_c, out, n, SCALE=1.0, BLOCK=1024, **_launch_opts())
    return out


@m1triton.jit
def group_norm_apply_hardswish_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    b_ptr,
    partial_ptr,
    C,
    HW,
    CHUNKS: tl.constexpr,
    GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_t = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    c_i64 = C.to(tl.int64)
    hw_i64 = HW.to(tl.int64)
    n_idx = row // c_i64
    ch = row - n_idx * c_i64
    cpg = C // GROUPS
    g_idx = ch // cpg.to(tl.int64)
    first = (n_idx * GROUPS + g_idx) * CHUNKS

    tot_n = 0.0
    tot_mean = 0.0
    tot_m2 = 0.0
    for k in range(0, CHUNKS):
        slot = (first + k) * 3
        kn = tl.load(partial_ptr + slot)
        kmean = tl.load(partial_ptr + slot + 1)
        km2 = tl.load(partial_ptr + slot + 2)
        new_n = tot_n + kn
        delta = kmean - tot_mean
        safe_n = tl.where(new_n > 0.0, new_n, 1.0)
        tot_mean = tot_mean + delta * (kn / safe_n)
        tot_m2 = tot_m2 + km2 + delta * delta * (tot_n * kn / safe_n)
        tot_n = new_n

    inv_std = 1.0 / tl.sqrt(tot_m2 / tot_n + EPS)
    sc = inv_std * tl.load(w_ptr + ch)
    sh = tl.load(b_ptr + ch) - tot_mean * sc

    offs = pid_t.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < hw_i64
    base = row * hw_i64 + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    v = x * sc + sh
    hs = v * tl.where(v <= -3.0, 0.0, tl.where(v >= 3.0, 1.0, (v + 3.0) * 0.1666666716337204))
    tl.store(out_ptr + base, hs, mask=mask)


def group_norm_hardswish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    hw = x_c.numel() // max(n * c, 1)
    group_elems = (c // num_groups) * hw
    chunks = max(1, min(32, _cdiv(group_elems, _STATS_ELEMS_PER_TG)))
    partial = torch.empty(max(n * num_groups * chunks * 3, 1), device=x_c.device, dtype=torch.float32)
    out = torch.empty_like(x_c)
    opts = _launch_opts()
    group_norm_stats_kernel[(chunks, n * num_groups, 1)](
        x_c, partial, n, c, hw, chunks, int(num_groups), **opts
    )
    block = 1024
    group_norm_apply_hardswish_kernel[(m1triton.cdiv(hw, block), n * c)](
        x_c, out, weight.contiguous(), bias.contiguous(), partial,
        c, hw, CHUNKS=chunks, GROUPS=int(num_groups), EPS=float(eps), BLOCK=block, **opts
    )
    return out


@m1triton.jit
def sum_h_gelu_add_bias0_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    H,
    W,
    BLOCK_W: tl.constexpr,
):
    tlx.vec_width(4)
    pid_w = tl.program_id(0)
    n_idx = tl.program_id(1).to(tl.int64)
    h_i64 = H.to(tl.int64)
    w_i64 = W.to(tl.int64)
    offs = pid_w.to(tl.int64) * BLOCK_W + tl.arange(0, BLOCK_W).to(tl.int64)
    mask = offs < w_i64
    p = x_ptr + n_idx * (h_i64 * w_i64) + offs
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for _ in range(0, H):
        acc = acc + tl.load(p, mask=mask, other=0.0)
        p = p + w_i64
    y = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865476)) + tl.load(bias_ptr)
    tl.store(out_ptr + n_idx * w_i64 + offs, y, mask=mask)


def sum_h_gelu_add_bias0(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_c = x if x.is_contiguous() else x.contiguous()
    n, _, h, w = (int(v) for v in x_c.shape)
    out = torch.empty((n, 1, 1, w), dtype=x_c.dtype, device=x_c.device)
    sum_h_gelu_add_bias0_kernel[(m1triton.cdiv(w, 128), n)](
        x_c, bias.contiguous().view(-1), out, h, w, BLOCK_W=128, **_launch_opts()
    )
    return out


@m1triton.jit
def spatial_sum_scaled_planes_kernel(
    x_ptr,
    out_ptr,
    S_in,
    INV_S_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * S_in.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, S_in, BLOCK_K):
        offs = k + ar
        mask = offs < S_in
        acc = acc + tl.load(base + offs.to(tl.int64), mask=mask, other=0.0)
    res = tl.sum(acc, axis=0) * INV_S_OUT
    tl.store(out_ptr + row, res)


@m1triton.jit
def weight_tap_sum_kernel(
    w_ptr,
    out_ptr,
    KT,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = w_ptr + row * KT.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, KT, BLOCK_K):
        offs = k + ar
        mask = offs < KT
        acc = acc + tl.load(base + offs.to(tl.int64), mask=mask, other=0.0)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@m1triton.jit
def gap_proj_affine_kernel(
    xsum_ptr,
    wsum_ptr,
    cb_ptr,
    eb_ptr,
    bn_sc_ptr,
    bn_sh_ptr,
    out_ptr,
    N,
    C_in,
    C_out,
    HAS_CB: tl.constexpr,
    HAS_EB: tl.constexpr,
    HAS_BN: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_co = tl.program_id(0)
    n_idx = tl.program_id(1).to(tl.int64)
    cin_i64 = C_in.to(tl.int64)
    cout_i64 = C_out.to(tl.int64)
    co_offs = pid_co.to(tl.int64) * BLOCK_CO + tl.arange(0, BLOCK_CO).to(tl.int64)
    mask = co_offs < cout_i64
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    x_base = xsum_ptr + n_idx * cin_i64
    for ci in range(0, C_in):
        xv = tl.load(x_base + ci.to(tl.int64))
        wv = tl.load(wsum_ptr + ci.to(tl.int64) * cout_i64 + co_offs, mask=mask, other=0.0)
        acc = acc + xv * wv
    if HAS_CB:
        acc = acc + tl.load(cb_ptr + co_offs, mask=mask, other=0.0)
    if PRE_SCALE != 1.0:
        acc = acc * PRE_SCALE
    if HAS_EB:
        acc = acc + tl.load(eb_ptr + co_offs, mask=mask, other=0.0)
    if HAS_BN:
        sc = tl.load(bn_sc_ptr + co_offs, mask=mask, other=1.0)
        sh = tl.load(bn_sh_ptr + co_offs, mask=mask, other=0.0)
        acc = acc * sc + sh
    tl.store(out_ptr + n_idx * cout_i64 + co_offs, acc, mask=mask)


def zero_pad_conv_transpose_gap(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    *,
    ndim: int,
    stride: tuple[int, ...],
    dilation: tuple[int, ...],
    out_shape: tuple[int, ...],
    extra_bias: torch.Tensor | None = None,
    pre_scale: float = 1.0,
    bn_scale: torch.Tensor | None = None,
    bn_shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact algebraic GlobalAvgPool(ConvTranspose{2,3}d(x)) for padding=0, output_padding=0, groups=1."""
    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = weight if weight.is_contiguous() else weight.contiguous()
    n, c_in = int(x_c.shape[0]), int(x_c.shape[1])
    c_out = int(w_c.shape[1])
    k = tuple(int(v) for v in w_c.shape[2:])
    spatial_in = tuple(int(v) for v in x_c.shape[2:])
    spatial_out = tuple(
        conv_transpose_output_size(spatial_in[i], k[i], stride[i], 0, dilation[i], 0) for i in range(ndim)
    )
    s_in = math.prod(spatial_in)
    s_out = math.prod(spatial_out)
    kt = math.prod(k)

    xsum = torch.empty((n, c_in), dtype=torch.float32, device=x_c.device)
    wsum = torch.empty((c_in, c_out), dtype=torch.float32, device=x_c.device)
    out = torch.empty(out_shape, dtype=x_c.dtype, device=x_c.device)
    opts = _launch_opts()

    bk_x = 256 if s_in <= 256 else 1024
    spatial_sum_scaled_planes_kernel[(n * c_in,)](
        x_c, xsum, s_in, INV_S_OUT=1.0 / float(s_out), BLOCK_K=bk_x, **opts
    )
    bk_w = min(256, max(32, _next_pow2(kt)))
    weight_tap_sum_kernel[(c_in * c_out,)](w_c, wsum, kt, BLOCK_K=bk_w, **opts)

    cb_arg = conv_bias.contiguous().view(-1) if conv_bias is not None else wsum
    eb_arg = extra_bias.contiguous().view(-1) if extra_bias is not None else wsum
    sc_arg = bn_scale.contiguous().view(-1) if bn_scale is not None else wsum
    sh_arg = bn_shift.contiguous().view(-1) if bn_shift is not None else wsum
    block_co = min(128, max(32, _next_pow2(c_out)))
    gap_proj_affine_kernel[(m1triton.cdiv(c_out, block_co), n)](
        xsum, wsum, cb_arg, eb_arg, sc_arg, sh_arg, out,
        n, c_in, c_out,
        HAS_CB=int(conv_bias is not None),
        HAS_EB=int(extra_bias is not None),
        HAS_BN=int(bn_scale is not None),
        PRE_SCALE=float(pre_scale),
        BLOCK_CO=block_co,
        **opts,
    )
    return out


class ConvTransposeL2Base(nn.Module):
    """Parameter-only ConvTranspose{2,3}d base for KernelBench L2 models (zero banned nn modules)."""

    ndim: int = 2

    def _init_conv_transpose(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        nd = self.ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = as_tuple(kernel_size, nd)
        self.stride = as_tuple(stride, nd)
        self.padding = as_tuple(padding, nd)
        self.output_padding = as_tuple(output_padding, nd)
        self.dilation = as_tuple(dilation, nd)
        self.groups = int(groups)
        self.conv_weight = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels // self.groups, *self.kernel_size)
        )
        if bias:
            self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("conv_bias", None)

    def _init_conv_transpose_from_ref(self, conv: nn.Module) -> None:
        self._init_conv_transpose(
            int(conv.in_channels),
            int(conv.out_channels),
            tuple(int(v) for v in conv.kernel_size),
            stride=tuple(int(v) for v in conv.stride),
            padding=tuple(int(v) for v in conv.padding),
            output_padding=tuple(int(v) for v in conv.output_padding),
            dilation=tuple(int(v) for v in conv.dilation),
            groups=int(conv.groups),
            bias=conv.bias is not None,
        )

    def _sync_conv_transpose_from_ref(self, conv: nn.Module) -> None:
        nd = self.ndim
        self.stride = as_tuple(conv.stride, nd)
        self.padding = as_tuple(conv.padding, nd)
        self.output_padding = as_tuple(conv.output_padding, nd)
        self.dilation = as_tuple(conv.dilation, nd)
        self.groups = int(conv.groups)
        with torch.no_grad():
            self.conv_weight.copy_(conv.weight.detach().to(self.conv_weight.device))
            if self.conv_bias is not None and conv.bias is not None:
                self.conv_bias.copy_(conv.bias.detach().to(self.conv_bias.device))

    def _init_bn_buffers(self, num_features: int, eps: float = 1e-5) -> None:
        self.bn_eps = float(eps)
        self.bn_weight = nn.Parameter(torch.ones(num_features))
        self.bn_bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("bn_running_mean", torch.zeros(num_features))
        self.register_buffer("bn_running_var", torch.ones(num_features))

    def _sync_bn_from_ref(self, bn: nn.Module) -> None:
        self.bn_eps = float(bn.eps)
        with torch.no_grad():
            self.bn_weight.copy_(bn.weight.detach().to(self.bn_weight.device))
            self.bn_bias.copy_(bn.bias.detach().to(self.bn_bias.device))
            self.bn_running_mean.copy_(bn.running_mean.detach().to(self.bn_running_mean.device))
            self.bn_running_var.copy_(bn.running_var.detach().to(self.bn_running_var.device))

    def _init_gn_params(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        self.gn_num_groups = int(num_groups)
        self.gn_eps = float(eps)
        self.gn_weight = nn.Parameter(torch.ones(num_channels))
        self.gn_bias = nn.Parameter(torch.zeros(num_channels))

    def _sync_gn_from_ref(self, gn: nn.Module) -> None:
        self.gn_num_groups = int(gn.num_groups)
        self.gn_eps = float(gn.eps)
        with torch.no_grad():
            self.gn_weight.copy_(gn.weight.detach().to(self.gn_weight.device))
            self.gn_bias.copy_(gn.bias.detach().to(self.gn_bias.device))

    def _init_ln_params(self, normalized_shape: int | Sequence[int], eps: float = 1e-5) -> None:
        shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(int(v) for v in normalized_shape)
        self.ln_shape = shape
        self.ln_eps = float(eps)
        self.ln_weight = nn.Parameter(torch.ones(shape))
        self.ln_bias = nn.Parameter(torch.zeros(shape))

    def _sync_ln_from_ref(self, ln: nn.Module) -> None:
        self.ln_eps = float(ln.eps)
        with torch.no_grad():
            self.ln_weight.copy_(ln.weight.detach().to(self.ln_weight.device))
            self.ln_bias.copy_(ln.bias.detach().to(self.ln_bias.device))

    def _prepare_conv(self, *, has_tail: bool = False) -> PreparedConvTranspose:
        return PreparedConvTranspose(
            self.conv_weight.detach(),
            None if self.conv_bias is None else self.conv_bias.detach(),
            ndim=self.ndim,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            has_tail=has_tail,
        )


__all__ = [
    "ConvTransposeL2Base",
    "PreparedConvTranspose",
    "as_tuple",
    "avgpool3d_bias_scale",
    "axis_max",
    "axis_mean",
    "axis_min",
    "axis_sum",
    "bn_fold_affine",
    "channel_lse_sig_sub_clamp",
    "channel_softmax_post",
    "conv_transpose_nd_epilogue",
    "depth_mean_add_bias",
    "finalize_spatial_partials_sum",
    "gelu_1d",
    "group_norm",
    "group_norm_hardswish",
    "launch_conv_transpose_fused",
    "layer_norm",
    "layernorm_avgpool3d_gelu",
    "layernorm_gelu_scale",
    "maxpool2d_hardtanh_mean_tanh",
    "maxpool3d_gap_clamp",
    "pool_nd",
    "row_logsumexp_scale",
    "spatial_softmax_mul_scale",
    "subtract_spatial_mean",
    "sum_h_gelu_add_bias0",
    "zero_pad_conv_transpose_gap",
]
