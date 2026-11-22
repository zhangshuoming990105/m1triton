"""Reusable M1Triton fused Conv2D/Conv3D EpilogueDAG, FusedTail, and post-op kernels for KernelBench Level 2."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import torch.nn as nn

import m1triton
import m1triton.language as tl
import m1triton.runtime as m1_rt
import m1triton.tlx as tlx

_L1_KERNELS_DIR = str(Path(__file__).resolve().parents[1] / "kernels")
if _L1_KERNELS_DIR not in sys.path:
    sys.path.insert(0, _L1_KERNELS_DIR)

from _axis_reduce_common import (
    axis_max,
    axis_min,
    axis_sum,
    last_axis_extrema_kernel,
    last_axis_sum_kernel,
)
from _conv_std_common import (
    _get_wino_workspace,
    as_tuple,
    conv_output_size,
    select_tile,
    select_wino4x3_bmm_tile,
    select_winograd_tile,
    wino4x3_bmm_kernel,
    wino4x3_input_kernel,
    winograd_filter_2x3_kernel,
    winograd_filter_4x3_rm_kernel,
)
from _norm_softmax_common import group_norm, layer_norm_stats_kernel
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
    sync_mps: bool | None = None,
    wait: bool | None = None,
) -> torch.Tensor:
    """Command-batch-safe 1D/2D/3D max and average pooling dispatcher."""
    kind = kind.lower()
    k = _pool_as_tuple(kernel_size, ndim, ())
    s = _pool_as_tuple(stride if stride is not None else k, ndim, k)
    p = _pool_as_tuple(padding, ndim, (0,) * ndim)
    d = _pool_as_tuple(dilation, ndim, (1,) * ndim) if kind == "max" else (1,) * ndim

    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [pool_output_size(spatial_in[i], k[i], s[i], p[i], d[i], ceil_mode) for i in range(ndim)]
    out = torch.empty((n, c, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    in_batch = m1_rt.in_batch()
    do_sync = (not in_batch) if sync_mps is None else bool(sync_mps)
    do_wait = (not in_batch) if wait is None else bool(wait)
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
        sync_mps=do_sync,
        wait=do_wait,
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


def _contig(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return x
    out = x.contiguous()
    if out.device.type == "mps":
        torch.mps.synchronize()
    return out


@m1triton.jit
def conv2d_ep_kernel(
    x_ptr, w_ptr, out_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
        groups=GROUPS,
    )


@m1triton.jit
def conv2d_bias_ep_kernel(
    x_ptr, w_ptr, out_ptr, b_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(K_H, K_W), stride=(S_H, S_W), padding=(P_H, P_W), dilation=(D_H, D_W),
        groups=GROUPS, has_bias=True,
    )


@m1triton.jit
def conv3d_ep_kernel(
    x_ptr, w_ptr, out_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    S_D: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_D: tl.constexpr, P_H: tl.constexpr, P_W: tl.constexpr,
    D_D: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
        dilation=(D_D, D_H, D_W), groups=GROUPS,
    )


@m1triton.jit
def conv3d_bias_ep_kernel(
    x_ptr, w_ptr, out_ptr, b_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    S_D: tl.constexpr, S_H: tl.constexpr, S_W: tl.constexpr,
    P_D: tl.constexpr, P_H: tl.constexpr, P_W: tl.constexpr,
    D_D: tl.constexpr, D_H: tl.constexpr, D_W: tl.constexpr,
    GROUPS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, w_ptr, out_ptr, b_ptr, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        ndim=3, kernel_size=(K_D, K_H, K_W), stride=(S_D, S_H, S_W), padding=(P_D, P_H, P_W),
        dilation=(D_D, D_H, D_W), groups=GROUPS, has_bias=True,
    )


@m1triton.jit
def conv2d_wino2x3_ep_kernel(
    x_ptr, u_ptr, out_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_TH: tl.constexpr, TILE_TW: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
    WINOGRAD: tl.constexpr = 1,
    WINO_PAD: tl.constexpr = 1,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, u_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def conv2d_wino2x3_bias_ep_kernel(
    x_ptr, u_ptr, out_ptr, b_ptr, ep0_ptr, ep1_ptr,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    P_H: tl.constexpr, P_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_TH: tl.constexpr, TILE_TW: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
    WINOGRAD: tl.constexpr = 1,
    WINO_PAD: tl.constexpr = 1,
    EPILOGUE: tl.constexpr = (),
    TAIL_KIND: tl.constexpr = "none",
    TAIL_SCALE: tl.constexpr = 1.0,
):
    tlx.epilogue(EPILOGUE)
    tlx.fused_tail(TAIL_KIND, post_scale=TAIL_SCALE)
    tlx.implicit_gemm_conv(
        x_ptr, u_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(P_H, P_W), dilation=(1, 1),
        groups=1, has_bias=True,
    )


@m1triton.jit
def wino4x3_output_ep_kernel(
    m_ptr, dummy_ptr, out_ptr, ep0_ptr, ep1_ptr,
    row_start, chunk_rows, p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    WINOGRAD: tl.constexpr = 12,
    EPILOGUE: tl.constexpr = (),
):
    tlx.epilogue(EPILOGUE)
    tlx.implicit_gemm_conv(
        m_ptr, dummy_ptr, out_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        groups=1,
    )


@m1triton.jit
def wino4x3_output_bias_ep_kernel(
    m_ptr, dummy_ptr, out_ptr, b_ptr, ep0_ptr, ep1_ptr,
    row_start, chunk_rows, p_ld,
    N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    WINOGRAD: tl.constexpr = 12,
    EPILOGUE: tl.constexpr = (),
):
    tlx.epilogue(EPILOGUE)
    tlx.implicit_gemm_conv(
        m_ptr, dummy_ptr, out_ptr, b_ptr, N, C_in, C_out, H_in, W_in, H_out, W_out,
        ndim=2, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        groups=1, has_bias=True,
    )


_STD_EP_KERNELS = {
    (2, False): conv2d_ep_kernel,
    (2, True): conv2d_bias_ep_kernel,
    (3, False): conv3d_ep_kernel,
    (3, True): conv3d_bias_ep_kernel,
}

_SHAPE_NAMES = {
    2: ("H_in", "W_in", "H_out", "W_out"),
    3: ("D_in", "H_in", "W_in", "D_out", "H_out", "W_out"),
}

_KEY_PREFIX = {2: ("H", "W"), 3: ("D", "H", "W")}


def _finalize_spatial_tail(
    partials: torch.Tensor,
    n: int,
    c_out: int,
    grid_x: int,
    tail: str,
    tail_scale: float,
    *,
    sync_mps: bool,
    wait: bool,
) -> torch.Tensor:
    if grid_x == 1:
        return partials.view(n, c_out)
    out = torch.empty((n, c_out), dtype=partials.dtype, device=partials.device)
    rows = n * c_out
    block_k = 256 if grid_x <= 256 else 1024
    if tail == "spatial_global_max":
        last_axis_extrema_kernel[(rows,)](
            partials, out, grid_x, BLOCK_K=block_k, IS_MAX=1, sync_mps=sync_mps, wait=wait
        )
        if tail_scale != 1.0:
            scaled = torch.empty_like(out)
            scale_const_kernel[(m1triton.cdiv(rows, 256),)](
                out, scaled, rows, float(tail_scale), BLOCK=256, sync_mps=False, wait=wait
            )
            return scaled
        return out
    last_axis_sum_kernel[(rows,)](
        partials, out, grid_x, 1.0, BLOCK_K=block_k, APPLY_SCALE=0, sync_mps=sync_mps, wait=wait
    )
    return out


def conv_nd_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    ndim: int,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    epilogue_steps: Sequence[tuple[Any, ...]] = (),
    extra_bufs: Sequence[torch.Tensor] = (),
    tail: str | None = None,
    tail_scale: float = 1.0,
    use_staged_4x3: bool = False,
    transposed: bool = False,
    tile: tuple[int, int, int, int] | None = None,
    sync_mps: bool = True,
    wait: bool = True,
) -> torch.Tensor:
    """Dispatches fused Conv2D/Conv3D with register-level EpilogueDAG and optional FusedTail."""
    x_c = _contig(x)
    w_c = _contig(weight.detach())
    n, c_in = int(x_c.shape[0]), int(x_c.shape[1])
    c_out = int(w_c.shape[1]) * int(groups) if transposed else int(w_c.shape[0])
    k = tuple(int(v) for v in w_c.shape[2:])
    s = as_tuple(stride, ndim)
    p_raw = as_tuple(padding, ndim)
    p = tuple(k[i] - 1 - p_raw[i] for i in range(ndim)) if transposed else p_raw
    d = as_tuple(dilation, ndim)
    spatial_in = [int(v) for v in x_c.shape[2:]]
    spatial_out = [conv_output_size(spatial_in[i], k[i], s[i], p[i], d[i]) for i in range(ndim)]

    if len(extra_bufs) > 2:
        raise ValueError("conv_nd_epilogue supports up to 2 extra epilogue buffers")
    ep0 = _contig(extra_bufs[0].detach()) if len(extra_bufs) >= 1 else x_c
    ep1 = _contig(extra_bufs[1].detach()) if len(extra_bufs) >= 2 else x_c
    ep_tuple = tuple(tuple(st) for st in epilogue_steps)
    tail_kind = str(tail) if tail is not None else "none"
    b_c = _contig(bias.detach()) if bias is not None else None

    can_wino = (
        tile is None
        and ndim == 2
        and k == (3, 3)
        and s == (1, 1)
        and d == (1, 1)
        and groups == 1
        and (not transposed or (p_raw[0] <= 2 and p_raw[1] <= 2))
        and x_c.dtype == torch.float32
        and w_c.dtype == torch.float32
        and (b_c is None or b_c.dtype == torch.float32)
        and c_in >= 8
        and c_out >= 16
        and spatial_out[0] * spatial_out[1] >= 256
    )

    if can_wino and use_staged_4x3 and tail is None and 64 <= c_in <= 128 and c_out >= 128 and spatial_out[0] * spatial_out[1] >= 15000:
        h_in, w_in = spatial_in
        h_out, w_out = spatial_out
        out = torch.empty((n, c_out, h_out, w_out), dtype=x_c.dtype, device=x_c.device)
        if out.numel() == 0:
            return out
        u = torch.empty((36, c_out, c_in), dtype=torch.float32, device=x_c.device)
        f_grid = (m1triton.cdiv(c_out * c_in, 256),)
        bm, bn, bk, nsg = select_wino4x3_bmm_tile(c_in, c_out)

        def _pad_pld(p_raw_val: int) -> int:
            pv = m1triton.cdiv(p_raw_val, bn) * bn
            if (pv % 256) == 0:
                pv += bn
            return pv

        th = m1triton.cdiv(h_out, 4)
        tw = m1triton.cdiv(w_out, 4)
        total_rows = n * th
        bytes_per_row = 36 * (c_in + c_out) * tw * 4
        max_rows = max(1, min(total_rows, (192 * 1024 * 1024) // max(bytes_per_row, 1)))
        p_alloc = _pad_pld(max_rows * tw)
        v_buf, m_buf = _get_wino_workspace(x_c.device, 36 * (c_in * p_alloc + 64), 36 * (c_out * p_alloc + 64))
        in_gx = m1triton.cdiv(tw, 64)

        def _run_staged(do_sync: bool, do_wait: bool) -> None:
            winograd_filter_4x3_rm_kernel[f_grid](
                w_c, u, C_in=c_in, C_out=c_out, TRANSPOSED=1 if transposed else 0, BLOCK_SIZE=256,
                sync_mps=do_sync, wait=False,
            )
            for r0 in range(0, total_rows, max_rows):
                rc = min(max_rows, total_rows - r0)
                p_cur = _pad_pld(rc * tw)
                is_last = (r0 + rc >= total_rows)
                wino4x3_input_kernel[(in_gx, rc, c_in)](
                    x_c, u, v_buf, r0, rc, p_cur,
                    N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                    P_H=p[0], P_W=p[1], WINOGRAD=10, sync_mps=False, wait=False,
                )
                wino4x3_bmm_kernel[(m1triton.cdiv(p_cur, bn), m1triton.cdiv(c_out, bm), 36)](
                    v_buf, u, m_buf, p_cur,
                    N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=nsg, WINOGRAD=11,
                    sync_mps=False, wait=False,
                )
                if b_c is not None:
                    wino4x3_output_bias_ep_kernel[(in_gx, rc, c_out)](
                        m_buf, u, out, b_c, ep0, ep1, r0, rc, p_cur,
                        N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                        WINOGRAD=12, EPILOGUE=ep_tuple, sync_mps=False, wait=(do_wait and is_last),
                    )
                else:
                    wino4x3_output_ep_kernel[(in_gx, rc, c_out)](
                        m_buf, u, out, ep0, ep1, r0, rc, p_cur,
                        N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                        WINOGRAD=12, EPILOGUE=ep_tuple, sync_mps=False, wait=(do_wait and is_last),
                    )

        if m1_rt.in_batch():
            _run_staged(False, False)
        else:
            with m1_rt.command_batch(wait=wait, sync_mps=sync_mps):
                _run_staged(False, False)
        return out

    if can_wino:
        h_in, w_in = spatial_in
        h_out, w_out = spatial_out
        mode, bm, bn, bk, tth, ttw, nsg, wpad = select_winograd_tile(c_in, c_out, w_in)
        c_in_pad = m1triton.cdiv(c_in, bk) * bk
        c_out_pad = m1triton.cdiv(c_out, bm) * bm
        u = torch.empty((16, c_in_pad // 8, c_out_pad // 8, 8, 8), dtype=torch.float32, device=x_c.device)
        f_grid = (m1triton.cdiv(c_out_pad * c_in_pad, 256),)
        th = m1triton.cdiv(h_out, 2)
        tw = m1triton.cdiv(w_out, 2)
        grid_x = m1triton.cdiv(th, tth) * m1triton.cdiv(tw, ttw)
        grid = (grid_x, m1triton.cdiv(c_out, bm), n)
        if tail is not None:
            out = torch.empty((n * c_out, grid_x), dtype=x_c.dtype, device=x_c.device)
        else:
            out = torch.empty((n, c_out, h_out, w_out), dtype=x_c.dtype, device=x_c.device)
        if out.numel() == 0:
            return out

        def _run_wino2x3() -> torch.Tensor:
            winograd_filter_2x3_kernel[f_grid](
                w_c, u, C_in=c_in, C_out=c_out, C_IN_PAD=c_in_pad, C_OUT_PAD=c_out_pad,
                TRANSPOSED=1 if transposed else 0, BLOCK_SIZE=256, sync_mps=False, wait=False,
            )
            if b_c is not None:
                conv2d_wino2x3_bias_ep_kernel[grid](
                    x_c, u, out, b_c, ep0, ep1,
                    N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                    P_H=p[0], P_W=p[1], BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TILE_TH=tth, TILE_TW=ttw,
                    NUM_SG=nsg, WINOGRAD=mode, WINO_PAD=wpad,
                    EPILOGUE=ep_tuple, TAIL_KIND=tail_kind, TAIL_SCALE=float(tail_scale),
                    sync_mps=False, wait=False,
                )
            else:
                conv2d_wino2x3_ep_kernel[grid](
                    x_c, u, out, ep0, ep1,
                    N=n, C_in=c_in, C_out=c_out, H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                    P_H=p[0], P_W=p[1], BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, TILE_TH=tth, TILE_TW=ttw,
                    NUM_SG=nsg, WINOGRAD=mode, WINO_PAD=wpad,
                    EPILOGUE=ep_tuple, TAIL_KIND=tail_kind, TAIL_SCALE=float(tail_scale),
                    sync_mps=False, wait=False,
                )
            if tail is not None:
                return _finalize_spatial_tail(
                    out, n, c_out, grid_x, tail, tail_scale, sync_mps=False, wait=False
                )
            return out

        if m1_rt.in_batch():
            return _run_wino2x3()
        with m1_rt.command_batch(wait=wait, sync_mps=sync_mps):
            return _run_wino2x3()

    if transposed:
        raise ValueError(
            "conv_nd_epilogue(transposed=True) only supports float32 Winograd 3x3 stride=1 dilation=1 "
            "groups=1 padding<=2 with C_in>=8, C_out>=16, H_out*W_out>=256; use launch_conv_transpose_fused instead."
        )

    p_spatial = math.prod(spatial_out)
    pixels = n * p_spatial
    pointwise = all(v == 1 for v in k) and all(v == 1 for v in s) and all(v == 0 for v in p)
    c_out_g = c_out // groups
    bm, bn, bk, num_sg = (
        tile
        if tile is not None
        else select_tile(c_in // groups, c_out_g, math.prod(k), pointwise, ndim=ndim, stride_w=s[-1])
    )
    if tail is not None:
        grid_x = m1triton.cdiv(p_spatial, bn)
        grid = (grid_x, m1triton.cdiv(c_out_g, bm), n * groups)
        out = torch.empty((n * c_out, grid_x), dtype=x_c.dtype, device=x_c.device)
    else:
        grid_x = m1triton.cdiv(pixels, bn)
        grid = (grid_x, m1triton.cdiv(c_out_g, bm), groups)
        out = torch.empty((n, c_out, *spatial_out), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out

    names = _KEY_PREFIX[ndim]
    consts: dict[str, int] = dict(zip(_SHAPE_NAMES[ndim], (*spatial_in, *spatial_out)))
    for i, nm in enumerate(names):
        consts[f"K_{nm}"] = k[i]
        consts[f"S_{nm}"] = s[i]
        consts[f"P_{nm}"] = p[i]
        consts[f"D_{nm}"] = d[i]
    kernel = _STD_EP_KERNELS[(ndim, b_c is not None)]
    args: list[object] = [x_c, w_c, out]
    if b_c is not None:
        args.append(b_c)
    args.extend([ep0, ep1])

    def _run_std(do_sync: bool, do_wait: bool) -> torch.Tensor:
        kernel[grid](
            *args, N=n, C_in=c_in, C_out=c_out, GROUPS=int(groups),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=num_sg,
            EPILOGUE=ep_tuple, TAIL_KIND=tail_kind, TAIL_SCALE=float(tail_scale),
            sync_mps=do_sync, wait=(do_wait and tail is None), **consts,
        )
        if tail is not None:
            return _finalize_spatial_tail(
                out, n, c_out, grid_x, tail, tail_scale, sync_mps=False, wait=do_wait
            )
        return out

    if tail is None:
        return _run_std(sync_mps and not m1_rt.in_batch(), wait and not m1_rt.in_batch())
    if m1_rt.in_batch():
        return _run_std(False, False)
    with m1_rt.command_batch(wait=wait, sync_mps=sync_mps):
        return _run_std(False, False)


@m1triton.jit
def scale_const_kernel(x_ptr, out_ptr, n, scale, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, v * scale, mask=mask)


@m1triton.jit
def bn_fold_affine_kernel(
    w_ptr, b_ptr, mean_ptr, var_ptr,
    out_scale_ptr, out_shift_ptr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    POST_SCALE: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    mean = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    var = tl.load(var_ptr + offs, mask=mask, other=1.0)
    inv_std = tl.rsqrt(var + EPS)
    if HAS_AFFINE:
        w = tl.load(w_ptr + offs, mask=mask, other=1.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        s = w * inv_std
        sh = b - mean * s
    else:
        s = inv_std
        sh = -mean * inv_std
    tl.store(out_scale_ptr + offs, s * POST_SCALE, mask=mask)
    tl.store(out_shift_ptr + offs, sh * POST_SCALE, mask=mask)


def compute_bn_affine(
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    *,
    eps: float = 1e-5,
    post_scale: float = 1.0,
    sync_mps: bool = False,
    wait: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes live inference BatchNorm per-channel (scale, shift) on GPU with zero caching."""
    rm = _contig(running_mean.detach())
    rv = _contig(running_var.detach())
    c = int(rm.numel())
    has_affine = weight is not None and bias is not None
    w = _contig(weight.detach()) if has_affine else rm
    b = _contig(bias.detach()) if has_affine else rm
    out_scale = torch.empty((c,), dtype=rm.dtype, device=rm.device)
    out_shift = torch.empty((c,), dtype=rm.dtype, device=rm.device)
    block = 64 if c <= 64 else 128
    bn_fold_affine_kernel[(m1triton.cdiv(c, block),)](
        w, b, rm, rv, out_scale, out_shift,
        C=c, EPS=float(eps), POST_SCALE=float(post_scale), HAS_AFFINE=1 if has_affine else 0,
        BLOCK=block, sync_mps=sync_mps, wait=wait,
    )
    return out_scale, out_shift


@m1triton.jit
def channel_softmax_kernel(
    x_ptr, out_ptr, P,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base = n * (C * p_i64) + offs

    m = tl.load(x_ptr + base, mask=mask, other=-float("inf"))
    for c in range(1, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=-float("inf"))
        m = tl.maximum(m, v)

    denom = tl.zeros((BLOCK_P,), dtype=tl.float32)
    for c in range(0, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=0.0)
        denom = denom + tl.exp(v - m)
    inv_denom = 1.0 / denom

    for c in range(0, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=0.0)
        tl.store(out_ptr + (base + c * p_i64), tl.exp(v - m) * inv_denom, mask=mask)


def channel_softmax(x: torch.Tensor, *, sync_mps: bool = False, wait: bool = False) -> torch.Tensor:
    """Coalesced softmax along dim=1 of an (N, C, *spatial) tensor."""
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    p = x_c.numel() // max(n * c, 1)
    out = torch.empty_like(x_c)
    if out.numel() == 0:
        return out
    block_p = 256
    channel_softmax_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, out, p, C=c, BLOCK_P=block_p, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def channel_min_kernel(
    x_ptr, out_ptr, P,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base = n * (C * p_i64) + offs

    acc = tl.load(x_ptr + base, mask=mask, other=float("inf"))
    for c in range(1, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=float("inf"))
        acc = tl.minimum(acc, v)
    tl.store(out_ptr + (n * p_i64 + offs), acc, mask=mask)


def channel_min(x: torch.Tensor, *, sync_mps: bool = False, wait: bool = False) -> torch.Tensor:
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(s) for s in x_c.shape[2:])
    p = math.prod(spatial)
    out = torch.empty((n, 1, *spatial), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out
    block_p = 256
    channel_min_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, out, p, C=c, BLOCK_P=block_p, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def channel_min_tanh_tanh_kernel(
    x_ptr, out_ptr, P,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base = n * (C * p_i64) + offs

    acc = tl.load(x_ptr + base, mask=mask, other=float("inf"))
    for c in range(1, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=float("inf"))
        acc = tl.minimum(acc, v)
    y = tl.tanh(tl.tanh(acc))
    tl.store(out_ptr + (n * p_i64 + offs), y, mask=mask)


def channel_min_tanh_tanh(x: torch.Tensor, *, sync_mps: bool = False, wait: bool = False) -> torch.Tensor:
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(s) for s in x_c.shape[2:])
    p = math.prod(spatial)
    out = torch.empty((n, 1, *spatial), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out
    block_p = 256
    channel_min_tanh_tanh_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, out, p, C=c, BLOCK_P=block_p, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def channel_logsumexp_relu_kernel(
    x_ptr, out_ptr, P,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base = n * (C * p_i64) + offs

    m = tl.load(x_ptr + base, mask=mask, other=-float("inf"))
    s = tl.full((BLOCK_P,), 1.0, dtype=tl.float32)
    for c in range(1, C):
        v = tl.load(x_ptr + (base + c * p_i64), mask=mask, other=-float("inf"))
        nm = tl.maximum(m, v)
        s = s * tl.exp(m - nm) + tl.exp(v - nm)
        m = nm
    lse = m + tl.log(s)
    y = tl.where(lse < 0.0, 0.0, lse)
    tl.store(out_ptr + (n * p_i64 + offs), y, mask=mask)


def channel_logsumexp_relu(x: torch.Tensor, *, sync_mps: bool = False, wait: bool = False) -> torch.Tensor:
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(s) for s in x_c.shape[2:])
    p = math.prod(spatial)
    out = torch.empty((n, 1, *spatial), dtype=x_c.dtype, device=x_c.device)
    if out.numel() == 0:
        return out
    block_p = 256
    channel_logsumexp_relu_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, out, p, C=c, BLOCK_P=block_p, sync_mps=sync_mps, wait=wait
    )
    return out


@m1triton.jit
def maxpool3d_gap_bias_kernel(
    x_ptr, b_ptr, out_ptr,
    C: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_p: tl.constexpr, H_p: tl.constexpr, W_p: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0).to(tl.int64)
    c = nc % C
    in_plane = D_in * H_in * W_in
    base_nc = nc * in_plane
    hw_in = H_in * W_in
    hw_p = H_p * W_p
    total_p = D_p * hw_p

    ar = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for p0 in range(0, total_p, BLOCK):
        idx = p0 + ar
        mask = idx < total_p
        od = idx // hw_p
        rem = idx - od * hw_p
        oh = rem // W_p
        ow = rem - oh * W_p
        id0 = od * KD
        ih0 = oh * KH
        iw0 = ow * KW

        best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
        for dd in tl.static_range(0, KD):
            for dh in tl.static_range(0, KH):
                for dw in tl.static_range(0, KW):
                    off = base_nc + ((id0 + dd).to(tl.int64) * hw_in + (ih0 + dh).to(tl.int64) * W_in + (iw0 + dw).to(tl.int64))
                    v = tl.load(x_ptr + off, mask=mask, other=-float("inf"))
                    best = tl.maximum(best, v)
        acc = acc + tl.where(mask, best, 0.0)

    mean_v = tl.sum(acc, axis=0) * (1.0 / float(total_p))
    if HAS_BIAS:
        mean_v = mean_v + tl.load(b_ptr + c)
    tl.store(out_ptr + nc, mean_v)


def maxpool3d_gap_bias(
    x: torch.Tensor,
    pool_size: int | Sequence[int],
    bias: torch.Tensor | None = None,
    *,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused MaxPool3d(pool_size, stride=pool_size) + GlobalAvgPool3d + optional channel bias."""
    x_c = _contig(x)
    n, c, d_in, h_in, w_in = (int(v) for v in x_c.shape)
    kd, kh, kw = as_tuple(pool_size, 3)
    d_p = pool_output_size(d_in, kd, kd, 0, 1, False)
    h_p = pool_output_size(h_in, kh, kh, 0, 1, False)
    w_p = pool_output_size(w_in, kw, kw, 0, 1, False)
    out = torch.empty((n, c, 1, 1, 1), dtype=x_c.dtype, device=x_c.device)
    b_c = _contig(bias.detach()).view(-1) if bias is not None else x_c
    maxpool3d_gap_bias_kernel[(n * c,)](
        x_c, b_c, out,
        C=c, D_in=d_in, H_in=h_in, W_in=w_in,
        D_p=d_p, H_p=h_p, W_p=w_p,
        KD=kd, KH=kh, KW=kw,
        HAS_BIAS=1 if bias is not None else 0,
        BLOCK=256, sync_mps=sync_mps, wait=wait,
    )
    return out


@m1triton.jit
def fuse_conv2d_avgpool_weight_kernel(
    w_ptr, w_eff_ptr,
    OUT_TOTAL: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    KEH: tl.constexpr, KEW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OUT_TOTAL
    ke_hw = KEH * KEW
    oc_ic = offs // ke_hw
    rem = offs - oc_ic * ke_hw
    p = rem // KEW
    q = rem - p * KEW
    w_base = oc_ic * (KH * KW)
    inv_area = 1.0 / float(PH * PW)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(0, KH):
        u = p - r
        u_ok = (u >= 0) & (u < PH)
        for s in tl.static_range(0, KW):
            v = q - s
            valid = mask & u_ok & (v >= 0) & (v < PW)
            val = tl.load(w_ptr + (w_base + (r * KW + s)), mask=valid, other=0.0)
            acc = acc + val
    tl.store(w_eff_ptr + offs, acc * inv_area, mask=mask)


def conv2d_avgpool_sigmoid_sum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    pool_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Algebraically fuses Conv2d(s=1,p=0,d=1) + AvgPool2d(k,s=k) + Sigmoid + Sum(dim=[1,2,3])."""
    ph, pw = int(pool_kernel_size[0]), int(pool_kernel_size[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if stride == (1, 1) and padding == (0, 0) and dilation == (1, 1) and kh <= 5 and kw <= 5 and ph <= 8 and pw <= 8:
        x_c = _contig(x)
        w_c = _contig(weight.detach())
        b_c = _contig(bias.detach()) if bias is not None else None
        n, c_in, h_in, w_in = (int(v) for v in x_c.shape)
        c_out = int(w_c.shape[0])
        c_in_g = int(w_c.shape[1])
        keh = kh + ph - 1
        kew = kw + pw - 1
        h_out = conv_output_size(h_in, keh, ph, 0, 1)
        w_out = conv_output_size(w_in, kew, pw, 0, 1)
        out_total = c_out * c_in_g * keh * kew
        w_eff = torch.empty((c_out, c_in_g, keh, kew), dtype=w_c.dtype, device=w_c.device)

        c_out_g = c_out // groups
        bm, bn, bk, num_sg = select_tile(c_in_g, c_out_g, keh * kew, False, ndim=2, stride_w=pw)
        p_spatial = h_out * w_out
        grid_x = m1triton.cdiv(p_spatial, bn)
        partials = torch.empty((n, c_out, grid_x), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((n,), dtype=x_c.dtype, device=x_c.device)

        kernel = _STD_EP_KERNELS[(2, b_c is not None)]
        args: list[object] = [x_c, w_eff, partials]
        if b_c is not None:
            args.append(b_c)
        args.extend([x_c, x_c])
        row_len = c_out * grid_x
        block_k = 256 if row_len <= 256 else 1024

        with m1_rt.command_batch(wait=True, sync_mps=True):
            fuse_conv2d_avgpool_weight_kernel[(m1triton.cdiv(out_total, 256),)](
                w_c, w_eff,
                OUT_TOTAL=out_total, KH=kh, KW=kw, PH=ph, PW=pw, KEH=keh, KEW=kew,
                BLOCK=256, sync_mps=False, wait=False,
            )
            kernel[(grid_x, m1triton.cdiv(c_out_g, bm), n * groups)](
                *args,
                N=n, C_in=c_in, C_out=c_out,
                H_in=h_in, W_in=w_in, H_out=h_out, W_out=w_out,
                K_H=keh, K_W=kew, S_H=ph, S_W=pw, P_H=0, P_W=0, D_H=1, D_W=1,
                GROUPS=int(groups), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_SG=num_sg,
                EPILOGUE=(("sigmoid",),), TAIL_KIND="spatial_global_sum", TAIL_SCALE=1.0,
                sync_mps=False, wait=False,
            )
            last_axis_sum_kernel[(n,)](
                partials, out, row_len, 1.0, BLOCK_K=block_k, APPLY_SCALE=0,
                sync_mps=False, wait=False,
            )
        return out

    with m1_rt.command_batch(wait=True, sync_mps=True):
        y = conv_nd_epilogue(
            x, weight, bias, ndim=2, stride=stride, padding=padding, dilation=dilation, groups=groups,
            sync_mps=False, wait=False,
        )
        y = avgpool2d_sigmoid_sum_chw(y, pool_kernel_size, sync_mps=False, wait=False)
    return y


@m1triton.jit
def sigmoid_row_sum_kernel(
    x_ptr, out_ptr,
    ROW_LEN,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    base = row.to(tl.int64) * ROW_LEN.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, ROW_LEN, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < ROW_LEN
        v = tl.load(x_ptr + (base + k_offs.to(tl.int64)), mask=k_mask, other=-100.0)
        acc = acc + 1.0 / (1.0 + tl.exp(-v))
    tl.store(out_ptr + row.to(tl.int64), tl.sum(acc, axis=0))


def avgpool2d_sigmoid_sum_chw(
    x: torch.Tensor,
    pool_size: int | Sequence[int],
    *,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Coalesced vec4 AvgPool2d(pool_size, stride=pool_size) + fused Sigmoid+Sum(dim=[1, 2, 3]) -> (N,)."""
    pooled = pool_nd(
        x,
        kind="avg",
        ndim=2,
        kernel_size=pool_size,
        stride=pool_size,
        sync_mps=sync_mps,
        wait=False,
    )
    n, c = int(pooled.shape[0]), int(pooled.shape[1])
    p = pooled.numel() // max(n * c, 1)
    partial = torch.empty((n, c), dtype=pooled.dtype, device=pooled.device)
    out = torch.empty((n,), dtype=pooled.dtype, device=pooled.device)
    block_k = 256 if p <= 256 else 1024
    sigmoid_row_sum_kernel[(n * c,)](
        pooled, partial, p, BLOCK_K=block_k, sync_mps=False, wait=False
    )
    block_c = 64 if c <= 64 else (256 if c <= 256 else 1024)
    last_axis_sum_kernel[(n,)](
        partial, out, c, 1.0, BLOCK_K=block_c, APPLY_SCALE=0, sync_mps=False, wait=wait
    )
    return out


@m1triton.jit
def maxpool2d_mish_kernel(
    x_ptr, out_ptr,
    H_in: tl.constexpr, W_in: tl.constexpr,
    H_p: tl.constexpr, W_p: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_p = tl.program_id(0).to(tl.int64)
    nc = tl.program_id(1).to(tl.int64)
    total_p = H_p * W_p
    offs = pid_p * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total_p
    oh = offs // W_p
    ow = offs - oh * W_p
    ih0 = oh * KH
    iw0 = ow * KW
    base_nc = nc * (H_in * W_in)

    best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    for dh in tl.static_range(0, KH):
        for dw in tl.static_range(0, KW):
            off = base_nc + (ih0 + dh) * W_in + (iw0 + dw)
            v = tl.load(x_ptr + off, mask=mask, other=-float("inf"))
            best = tl.maximum(best, v)

    sp = tl.where(best > 20.0, best, tl.log(1.0 + tl.exp(best)))
    y = best * tl.tanh(sp)
    tl.store(out_ptr + (nc * total_p + offs), y, mask=mask)


def maxpool2d_mish(
    x: torch.Tensor,
    pool_size: int | Sequence[int],
    *,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused MaxPool2d(pool_size, stride=pool_size) + Mish."""
    x_c = _contig(x)
    n, c, h_in, w_in = (int(v) for v in x_c.shape)
    kh, kw = as_tuple(pool_size, 2)
    h_p = pool_output_size(h_in, kh, kh, 0, 1, False)
    w_p = pool_output_size(w_in, kw, kw, 0, 1, False)
    out = torch.empty((n, c, h_p, w_p), dtype=x_c.dtype, device=x_c.device)
    total_p = h_p * w_p
    block = 256
    maxpool2d_mish_kernel[(m1triton.cdiv(total_p, block), n * c)](
        x_c, out, H_in=h_in, W_in=w_in, H_p=h_p, W_p=w_p, KH=kh, KW=kw,
        BLOCK=block, sync_mps=sync_mps, wait=wait,
    )
    return out


def _welford_row_stats(
    x: torch.Tensor,
    num_rows: int,
    row_len: int,
    *,
    sync_mps: bool = False,
) -> torch.Tensor:
    """Single-pass SIMDgroup shifted-Welford stats returning (num_rows, 3) of [count, mean, m2]."""
    partial = torch.empty((num_rows * 3,), dtype=torch.float32, device=x.device)
    layer_norm_stats_kernel[(1, num_rows, 1)](
        x, partial, num_rows, row_len, 1, sync_mps=sync_mps, wait=False
    )
    return partial


@m1triton.jit
def instance_norm_scale_apply_kernel(
    x_ptr, part_ptr, w_ptr, b_ptr, out_ptr,
    C: tl.constexpr,
    P: tl.constexpr,
    EPS: tl.constexpr,
    INV_DIV: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    nc = tl.program_id(1).to(tl.int64)
    c = nc % C
    offs = pid_p * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < P
    q = nc * 3
    cnt = tl.load(part_ptr + q)
    mean = tl.load(part_ptr + (q + 1))
    m2 = tl.load(part_ptr + (q + 2))
    inv_std = tl.rsqrt(m2 / cnt + EPS)
    if HAS_AFFINE:
        w = tl.load(w_ptr + c)
        b = tl.load(b_ptr + c)
        sc = w * inv_std * INV_DIV
        sh = (b - mean * w * inv_std) * INV_DIV
    else:
        sc = inv_std * INV_DIV
        sh = -mean * inv_std * INV_DIV
    base = nc * P + offs
    v = tl.load(x_ptr + base, mask=mask, other=0.0)
    tl.store(out_ptr + base, v * sc + sh, mask=mask)


def instance_norm_divide(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    divide_by: float = 1.0,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """InstanceNorm over spatial dims fused with division by a constant."""
    x_c = _contig(x)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    p = x_c.numel() // max(n * c, 1)
    partial = _welford_row_stats(x_c, n * c, p, sync_mps=sync_mps)
    out = torch.empty_like(x_c)
    has_affine = weight is not None and bias is not None
    w = _contig(weight.detach()) if has_affine else partial
    b = _contig(bias.detach()) if has_affine else partial
    block_app = 1024
    instance_norm_scale_apply_kernel[(m1triton.cdiv(p, block_app), n * c)](
        x_c, partial, w, b, out,
        C=c, P=p, EPS=float(eps), INV_DIV=1.0 / float(divide_by), HAS_AFFINE=1 if has_affine else 0,
        BLOCK=block_app, sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def in_apply_clamp_mul_max_dim1_kernel(
    x_ptr, part_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    P,
    C: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base_n = n * (C * p_i64) + offs
    nc_base = n * C

    best = tl.full((BLOCK_P,), -float("inf"), dtype=tl.float32)
    for c in range(0, C):
        q = (nc_base + c) * 3
        cnt = tl.load(part_ptr + q)
        mean = tl.load(part_ptr + (q + 1))
        m2 = tl.load(part_ptr + (q + 2))
        inv_std = tl.rsqrt(m2 / cnt + EPS)
        mult = tl.load(mult_ptr + c)
        v = tl.load(x_ptr + (base_n + c * p_i64), mask=mask, other=0.0)
        if HAS_AFFINE:
            w = tl.load(w_ptr + c)
            b = tl.load(b_ptr + c)
            v = (v - mean) * inv_std * w + b
        else:
            v = (v - mean) * inv_std
        v = tl.where(v < CLAMP_MIN, CLAMP_MIN, tl.where(v > CLAMP_MAX, CLAMP_MAX, v))
        v = v * mult
        best = tl.maximum(best, v)
    tl.store(out_ptr + (n * p_i64 + offs), best, mask=mask)


def instance_norm_clamp_mul_max_dim1(
    x: torch.Tensor,
    multiplier: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused InstanceNorm -> Clamp -> Mul(multiplier) -> max(dim=1) eliminating full intermediate tensor."""
    x_c = _contig(x)
    mult_c = _contig(multiplier.detach()).view(-1)
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(s) for s in x_c.shape[2:])
    p = math.prod(spatial)
    partial = _welford_row_stats(x_c, n * c, p, sync_mps=sync_mps)
    out = torch.empty((n, *spatial), dtype=x_c.dtype, device=x_c.device)
    has_affine = weight is not None and bias is not None
    w = _contig(weight.detach()) if has_affine else mult_c
    b = _contig(bias.detach()) if has_affine else mult_c
    block_p = 256
    in_apply_clamp_mul_max_dim1_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, partial, w, b, mult_c, out, p,
        C=c, EPS=float(eps), CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
        HAS_AFFINE=1 if has_affine else 0,
        BLOCK_P=block_p, sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def gn_spatial_mean_finalize_kernel(
    part_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    C: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    EPS: tl.constexpr,
    REDUCE_CHANNELS: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    base_n = n * C
    inv_cg = 1.0 / float(CG)
    total = 0.0
    for g in range(0, G):
        c0 = g * CG
        mu_g = 0.0
        for j in range(0, CG):
            mu_g = mu_g + tl.load(part_ptr + ((base_n + c0 + j) * 3 + 1))
        mu_g = mu_g * inv_cg

        var_g = 0.0
        for j in range(0, CG):
            q = (base_n + c0 + j) * 3
            cnt = tl.load(part_ptr + q)
            mc = tl.load(part_ptr + (q + 1))
            m2c = tl.load(part_ptr + (q + 2))
            d = mc - mu_g
            var_g = var_g + (m2c / cnt) + d * d
        var_g = var_g * inv_cg
        inv_std_g = tl.rsqrt(var_g + EPS)

        for j in range(0, CG):
            c = c0 + j
            mc = tl.load(part_ptr + ((base_n + c) * 3 + 1))
            gw = tl.load(gn_w_ptr + c)
            gb = tl.load(gn_b_ptr + c)
            val = gw * (mc - mu_g) * inv_std_g + gb
            if REDUCE_CHANNELS:
                total = total + val
            else:
                tl.store(out_ptr + (base_n + c), val)
    if REDUCE_CHANNELS:
        tl.store(out_ptr + n, total * (1.0 / float(C)))


def group_norm_spatial_mean(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
    reduce_channels: bool = False,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Exact algebraic GroupNorm + spatial mean (and optional channel mean) in 1 pass over x."""
    x_c = _contig(x)
    w_c = _contig(weight.detach())
    b_c = _contig(bias.detach())
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    p = x_c.numel() // max(n * c, 1)
    partial = _welford_row_stats(x_c, n * c, p, sync_mps=sync_mps)
    out = torch.empty((n,) if reduce_channels else (n, c), dtype=x_c.dtype, device=x_c.device)
    gn_spatial_mean_finalize_kernel[(n,)](
        partial, w_c, b_c, out,
        C=c, G=int(num_groups), CG=c // int(num_groups), EPS=float(eps),
        REDUCE_CHANNELS=1 if reduce_channels else 0,
        sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def gn_apply_min_clamp_kernel(
    x_ptr, part_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    C: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    P: tl.constexpr,
    EPS: tl.constexpr,
    MIN_VAL: tl.constexpr,
    MAX_VAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    nc = tl.program_id(1).to(tl.int64)
    n = nc // C
    c = nc - n * C
    g = c // CG
    q = (n * G + g) * 3
    cnt = tl.load(part_ptr + q)
    mu = tl.load(part_ptr + (q + 1))
    m2 = tl.load(part_ptr + (q + 2))
    inv_std = tl.rsqrt(m2 / cnt + EPS)
    gw = tl.load(gn_w_ptr + c)
    gb = tl.load(gn_b_ptr + c)
    sc = gw * inv_std
    sh = gb - mu * sc

    offs = pid_p * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < P
    base = nc * P + offs
    v = tl.load(x_ptr + base, mask=mask, other=0.0) * sc + sh
    v = tl.minimum(v, MIN_VAL)
    v = tl.where(v < MIN_VAL, MIN_VAL, tl.where(v > MAX_VAL, MAX_VAL, v))
    tl.store(out_ptr + base, v, mask=mask)


def group_norm_min_clamp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
    min_value: float = 0.0,
    max_value: float = 1.0,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused GroupNorm -> min(x, min_value) -> clamp(min_value, max_value)."""
    x_c = _contig(x)
    w_c = _contig(weight.detach())
    b_c = _contig(bias.detach())
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    p = x_c.numel() // max(n * c, 1)
    g = int(num_groups)
    cg = c // g
    partial = _welford_row_stats(x_c, n * g, cg * p, sync_mps=sync_mps)
    out = torch.empty_like(x_c)
    block_app = 1024
    gn_apply_min_clamp_kernel[(m1triton.cdiv(p, block_app), n * c)](
        x_c, partial, w_c, b_c, out,
        C=c, G=g, CG=cg, P=p, EPS=float(eps), MIN_VAL=float(min_value), MAX_VAL=float(max_value),
        BLOCK=block_app, sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def gn_scale_maxpool2d_clamp_kernel(
    x_ptr, part_ptr, gn_w_ptr, gn_b_ptr, scale_ptr, out_ptr,
    C: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    H_in: tl.constexpr, W_in: tl.constexpr,
    H_p: tl.constexpr, W_p: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr, CLAMP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_p = tl.program_id(0).to(tl.int64)
    nc = tl.program_id(1).to(tl.int64)
    n = nc // C
    c = nc - n * C
    g = c // CG
    q = (n * G + g) * 3
    cnt = tl.load(part_ptr + q)
    mu = tl.load(part_ptr + (q + 1))
    m2 = tl.load(part_ptr + (q + 2))
    inv_std = tl.rsqrt(m2 / cnt + EPS)
    gw = tl.load(gn_w_ptr + c)
    gb = tl.load(gn_b_ptr + c)
    sc_param = tl.load(scale_ptr + c)
    eff_sc = gw * inv_std * sc_param
    eff_sh = (gb - mu * gw * inv_std) * sc_param

    total_p = H_p * W_p
    offs = pid_p * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < total_p
    oh = offs // W_p
    ow = offs - oh * W_p
    ih0 = oh * KH
    iw0 = ow * KW
    base_nc = nc * (H_in * W_in)

    best = tl.full((BLOCK,), -float("inf"), dtype=tl.float32)
    for dh in tl.static_range(0, KH):
        for dw in tl.static_range(0, KW):
            off = base_nc + (ih0 + dh) * W_in + (iw0 + dw)
            raw = tl.load(x_ptr + off, mask=mask, other=0.0)
            v = raw * eff_sc + eff_sh
            best = tl.maximum(best, v)
    best = tl.where(best < CLAMP_MIN, CLAMP_MIN, tl.where(best > CLAMP_MAX, CLAMP_MAX, best))
    tl.store(out_ptr + (nc * total_p + offs), best, mask=mask)


def group_norm_scale_maxpool2d_clamp(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    scale: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
    pool_kernel_size: int | Sequence[int] = 4,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused GroupNorm -> * scale -> MaxPool2d -> Clamp eliminating full intermediate tensors."""
    x_c = _contig(x)
    gw = _contig(gn_weight.detach())
    gb = _contig(gn_bias.detach())
    sc = _contig(scale.detach()).view(-1)
    n, c, h_in, w_in = (int(v) for v in x_c.shape)
    g = int(num_groups)
    cg = c // g
    kh, kw = as_tuple(pool_kernel_size, 2)
    h_p = pool_output_size(h_in, kh, kh, 0, 1, False)
    w_p = pool_output_size(w_in, kw, kw, 0, 1, False)

    partial = _welford_row_stats(x_c, n * g, cg * h_in * w_in, sync_mps=sync_mps)
    out = torch.empty((n, c, h_p, w_p), dtype=x_c.dtype, device=x_c.device)
    total_p = h_p * w_p
    block = 256
    gn_scale_maxpool2d_clamp_kernel[(m1triton.cdiv(total_p, block), n * c)](
        x_c, partial, gw, gb, sc, out,
        C=c, G=g, CG=cg, H_in=h_in, W_in=w_in, H_p=h_p, W_p=w_p,
        KH=kh, KW=kw, EPS=float(eps), CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
        BLOCK=block, sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def gn_apply_kernel(
    x_ptr, part_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    C: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    P: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    nc = tl.program_id(1).to(tl.int64)
    n = nc // C
    c = nc - n * C
    g = c // CG
    q = (n * G + g) * 3
    cnt = tl.load(part_ptr + q)
    mu = tl.load(part_ptr + (q + 1))
    m2 = tl.load(part_ptr + (q + 2))
    inv_std = tl.rsqrt(m2 / cnt + EPS)
    gw = tl.load(gn_w_ptr + c)
    gb = tl.load(gn_b_ptr + c)
    sc = gw * inv_std
    sh = gb - mu * sc

    offs = pid_p * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < P
    base = nc * P + offs
    v = tl.load(x_ptr + base, mask=mask, other=0.0) * sc + sh
    tl.store(out_ptr + base, v, mask=mask)


def group_norm_2stage(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Two-stage GroupNorm compatible with command_batch (Welford stats + coalesced affine apply)."""
    x_c = _contig(x)
    w_c = _contig(weight.detach())
    b_c = _contig(bias.detach())
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    p = x_c.numel() // max(n * c, 1)
    g = int(num_groups)
    cg = c // g
    partial = _welford_row_stats(x_c, n * g, cg * p, sync_mps=sync_mps)
    out = torch.empty_like(x_c)
    block_app = 1024
    gn_apply_kernel[(m1triton.cdiv(p, block_app), n * c)](
        x_c, partial, w_c, b_c, out,
        C=c, G=g, CG=cg, P=p, EPS=float(eps), BLOCK=block_app, sync_mps=False, wait=wait,
    )
    return out


@m1triton.jit
def gn_tanh_hardswish_res_logsumexp_kernel(
    x_ptr, part_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    P,
    C: tl.constexpr,
    G: tl.constexpr,
    CG: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    tlx.vec_width(4)
    pid_p = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64)
    p_i64 = P.to(tl.int64)
    offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    mask = offs < p_i64
    base_n = n * (C * p_i64) + offs
    ng_base = n * G

    q0 = ng_base * 3
    cnt0 = tl.load(part_ptr + q0)
    mu0 = tl.load(part_ptr + (q0 + 1))
    m20 = tl.load(part_ptr + (q0 + 2))
    inv_std0 = tl.rsqrt(m20 / cnt0 + EPS)
    gw0 = tl.load(gn_w_ptr)
    gb0 = tl.load(gn_b_ptr)
    xc0 = tl.load(x_ptr + base_n, mask=mask, other=0.0)
    vn0 = (xc0 - mu0) * inv_std0 * gw0 + gb0
    vt0 = tl.tanh(vn0)
    hs0 = vt0 * tl.where(vt0 <= -3.0, 0.0, tl.where(vt0 >= 3.0, 1.0, (vt0 + 3.0) * 0.1666666716337204))
    m = xc0 + hs0
    s = tl.full((BLOCK_P,), 1.0, dtype=tl.float32)

    for c in range(1, C):
        g = c // CG
        q = (ng_base + g) * 3
        cnt = tl.load(part_ptr + q)
        mu = tl.load(part_ptr + (q + 1))
        m2 = tl.load(part_ptr + (q + 2))
        inv_std = tl.rsqrt(m2 / cnt + EPS)
        gw = tl.load(gn_w_ptr + c)
        gb = tl.load(gn_b_ptr + c)
        xc = tl.load(x_ptr + (base_n + c * p_i64), mask=mask, other=0.0)
        vn = (xc - mu) * inv_std * gw + gb
        vt = tl.tanh(vn)
        hs = vt * tl.where(vt <= -3.0, 0.0, tl.where(vt >= 3.0, 1.0, (vt + 3.0) * 0.1666666716337204))
        vr = xc + hs
        nm = tl.maximum(m, vr)
        s = s * tl.exp(m - nm) + tl.exp(vr - nm)
        m = nm
    tl.store(out_ptr + (n * p_i64 + offs), m + tl.log(s), mask=mask)


def group_norm_tanh_hardswish_res_logsumexp(
    x: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float = 1e-5,
    sync_mps: bool = False,
    wait: bool = False,
) -> torch.Tensor:
    """Fused GroupNorm -> Tanh -> HardSwish -> + x -> logsumexp(dim=1, keepdim=True)."""
    x_c = _contig(x)
    gw = _contig(gn_weight.detach())
    gb = _contig(gn_bias.detach())
    n, c = int(x_c.shape[0]), int(x_c.shape[1])
    spatial = tuple(int(s) for s in x_c.shape[2:])
    p = math.prod(spatial)
    g = int(num_groups)
    cg = c // g
    partial = _welford_row_stats(x_c, n * g, cg * p, sync_mps=sync_mps)
    out = torch.empty((n, 1, *spatial), dtype=x_c.dtype, device=x_c.device)
    block_p = 256
    gn_tanh_hardswish_res_logsumexp_kernel[(m1triton.cdiv(p, block_p), n)](
        x_c, partial, gw, gb, out, p,
        C=c, G=g, CG=cg, EPS=float(eps), BLOCK_P=block_p, sync_mps=False, wait=wait,
    )
    return out


class ConvStdL2Base(nn.Module):
    """Parameter-only base class for KernelBench Level 2 Standard Conv2D/Conv3D tasks."""

    _NDIM = 2

    def _init_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
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
        self.conv_weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels // self.groups, *self.kernel_size).uniform_(-bound, bound)
        )
        if bias:
            self.conv_bias = nn.Parameter(torch.empty(self.out_channels).uniform_(-bound, bound))
        else:
            self.register_parameter("conv_bias", None)

    def _sync_conv_from(self, ref_conv: nn.Module) -> None:
        with torch.no_grad():
            self.conv_weight.copy_(ref_conv.weight.detach().to(self.conv_weight.device))
            if self.conv_bias is not None and ref_conv.bias is not None:
                self.conv_bias.copy_(ref_conv.bias.detach().to(self.conv_bias.device))

    def _init_group_norm(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        self.gn_num_groups = int(num_groups)
        self.gn_eps = float(eps)
        self.gn_affine = bool(affine)
        if affine:
            self.gn_weight = nn.Parameter(torch.ones(int(num_channels)))
            self.gn_bias = nn.Parameter(torch.zeros(int(num_channels)))
        else:
            self.register_parameter("gn_weight", None)
            self.register_parameter("gn_bias", None)

    def _sync_group_norm_from(self, ref_gn: nn.Module) -> None:
        self.gn_num_groups = int(ref_gn.num_groups)
        self.gn_eps = float(ref_gn.eps)
        with torch.no_grad():
            if self.gn_weight is not None and ref_gn.weight is not None:
                self.gn_weight.copy_(ref_gn.weight.detach().to(self.gn_weight.device))
            if self.gn_bias is not None and ref_gn.bias is not None:
                self.gn_bias.copy_(ref_gn.bias.detach().to(self.gn_bias.device))

    def _init_batch_norm(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        self.bn_eps = float(eps)
        self.bn_affine = bool(affine)
        if affine:
            self.bn_weight = nn.Parameter(torch.ones(int(num_features)))
            self.bn_bias = nn.Parameter(torch.zeros(int(num_features)))
        else:
            self.register_parameter("bn_weight", None)
            self.register_parameter("bn_bias", None)
        self.register_buffer("bn_running_mean", torch.zeros(int(num_features)))
        self.register_buffer("bn_running_var", torch.ones(int(num_features)))

    def _sync_batch_norm_from(self, ref_bn: nn.Module) -> None:
        self.bn_eps = float(ref_bn.eps)
        with torch.no_grad():
            if self.bn_weight is not None and ref_bn.weight is not None:
                self.bn_weight.copy_(ref_bn.weight.detach().to(self.bn_weight.device))
            if self.bn_bias is not None and ref_bn.bias is not None:
                self.bn_bias.copy_(ref_bn.bias.detach().to(self.bn_bias.device))
            if ref_bn.running_mean is not None:
                self.bn_running_mean.copy_(ref_bn.running_mean.detach().to(self.bn_running_mean.device))
            if ref_bn.running_var is not None:
                self.bn_running_var.copy_(ref_bn.running_var.detach().to(self.bn_running_var.device))

    def _init_instance_norm(self, num_features: int, eps: float = 1e-5, affine: bool = False) -> None:
        self.in_eps = float(eps)
        self.in_affine = bool(affine)
        if affine:
            self.in_weight = nn.Parameter(torch.ones(int(num_features)))
            self.in_bias = nn.Parameter(torch.zeros(int(num_features)))
        else:
            self.register_parameter("in_weight", None)
            self.register_parameter("in_bias", None)

    def _sync_instance_norm_from(self, ref_in: nn.Module) -> None:
        self.in_eps = float(ref_in.eps)
        with torch.no_grad():
            if self.in_weight is not None and ref_in.weight is not None:
                self.in_weight.copy_(ref_in.weight.detach().to(self.in_weight.device))
            if self.in_bias is not None and ref_in.bias is not None:
                self.in_bias.copy_(ref_in.bias.detach().to(self.in_bias.device))


__all__ = [
    "ConvStdL2Base",
    "as_tuple",
    "avgpool2d_sigmoid_sum_chw",
    "axis_max",
    "axis_min",
    "axis_sum",
    "bn_fold_affine_kernel",
    "channel_logsumexp_relu",
    "channel_min_tanh_tanh",
    "channel_softmax",
    "compute_bn_affine",
    "conv2d_avgpool_sigmoid_sum",
    "conv_nd_epilogue",
    "group_norm",
    "group_norm_2stage",
    "group_norm_min_clamp",
    "group_norm_scale_maxpool2d_clamp",
    "group_norm_spatial_mean",
    "group_norm_tanh_hardswish_res_logsumexp",
    "instance_norm_clamp_mul_max_dim1",
    "instance_norm_divide",
    "m1_rt",
    "maxpool2d_mish",
    "maxpool3d_gap_bias",
    "pool_nd",
]
