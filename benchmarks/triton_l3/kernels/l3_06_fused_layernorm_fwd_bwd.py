"""Level 3 Kernel L3-06: Fused LayerNorm Forward + Backward with Atomic Parameter Gradients."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as m1_rt
import m1triton.tlx as tlx


def _prune_row_block_configs(configs, named_args):
    n = int(named_args.get("N", 4096))
    if n <= 4096:
        target = max(512, m1triton.next_power_of_2(n))
        exact = [c for c in configs if int(c.kwargs["BLOCK_N"]) == target]
        if exact:
            return exact
    kept = [c for c in configs if int(c.kwargs["BLOCK_N"]) <= 4096]
    return kept if kept else configs[:1]


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_N": 512}, num_warps=4),
        m1triton.Config({"BLOCK_N": 1024}, num_warps=8),
        m1triton.Config({"BLOCK_N": 2048}, num_warps=8),
        m1triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_row_block_configs},
)
@m1triton.heuristics(
    {
        "SINGLE_PASS": lambda args: args["N"] <= args["BLOCK_N"],
        "NUM_TILES": lambda args: tl.cdiv(args["N"], args["BLOCK_N"]),
        "PAD_F": lambda args: float(tl.cdiv(args["N"], args["BLOCK_N"]) * args["BLOCK_N"] - args["N"]),
    }
)
@m1triton.jit(fast_math=True)
def _ln_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    M,
    N,
    inv_n,
    eps,
    BLOCK_N: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PAD_F: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row.to(tl.int64) * N.to(tl.int64)

    if SINGLE_PASS:
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
        mean = tl.sum(x, axis=0) * inv_n
        diff = x - mean
        sq = tl.sum(diff * diff, axis=0) - PAD_F * (mean * mean)
        rstd = tl.rsqrt(tl.maximum(sq * inv_n, 0.0) + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=1.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        y = diff * rstd * w + b
        tl.store(y_ptr + row_base + offs, y, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)
    else:
        offs0 = tl.arange(0, BLOCK_N)
        acc_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        p_x = x_ptr + row_base + offs0
        col = 0
        for _ in range(0, NUM_TILES):
            mask = (col + offs0) < N
            x = tl.load(p_x, mask=mask, other=0.0)
            acc_sum = acc_sum + x
            p_x = p_x + BLOCK_N
            col = col + BLOCK_N
        mean = tl.sum(acc_sum, axis=0) * inv_n

        acc_var = tl.zeros((BLOCK_N,), dtype=tl.float32)
        p_x2 = x_ptr + row_base + offs0
        col2 = 0
        for _ in range(0, NUM_TILES):
            mask = (col2 + offs0) < N
            x = tl.load(p_x2, mask=mask, other=0.0)
            diff = x - mean
            acc_var = acc_var + diff * diff
            p_x2 = p_x2 + BLOCK_N
            col2 = col2 + BLOCK_N
        sq = tl.sum(acc_var, axis=0) - PAD_F * (mean * mean)
        rstd = tl.rsqrt(tl.maximum(sq * inv_n, 0.0) + eps)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

        p_x3 = x_ptr + row_base + offs0
        p_y = y_ptr + row_base + offs0
        p_w = w_ptr + offs0
        p_b = b_ptr + offs0
        col3 = 0
        for _ in range(0, NUM_TILES):
            mask = (col3 + offs0) < N
            x = tl.load(p_x3, mask=mask, other=0.0)
            w = tl.load(p_w, mask=mask, other=1.0)
            b = tl.load(p_b, mask=mask, other=0.0)
            tl.store(p_y, (x - mean) * rstd * w + b, mask=mask)
            p_x3 = p_x3 + BLOCK_N
            p_y = p_y + BLOCK_N
            p_w = p_w + BLOCK_N
            p_b = p_b + BLOCK_N
            col3 = col3 + BLOCK_N


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_N": 512}, num_warps=4),
        m1triton.Config({"BLOCK_N": 1024}, num_warps=8),
        m1triton.Config({"BLOCK_N": 2048}, num_warps=8),
        m1triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["M", "N"],
    prune_configs_by={"early_config_prune": _prune_row_block_configs},
)
@m1triton.heuristics(
    {
        "SINGLE_PASS": lambda args: args["N"] <= args["BLOCK_N"],
        "NUM_TILES": lambda args: tl.cdiv(args["N"], args["BLOCK_N"]),
    }
)
@m1triton.jit(fast_math=True)
def _ln_bwd_dx_kernel(
    x_ptr,
    dy_ptr,
    w_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    M,
    N,
    inv_n,
    BLOCK_N: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row.to(tl.int64) * N.to(tl.int64)

    if SINGLE_PASS:
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        dy = tl.load(dy_ptr + row_base + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        wdy = dy * w
        c2 = tl.sum(wdy, axis=0) * inv_n
        x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
        xwdy_mean = tl.sum(x * wdy, axis=0) * inv_n
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        c1 = (xwdy_mean - mean * c2) * rstd
        a = c1 * rstd * rstd
        b_s = (c2 - mean * rstd * c1) * rstd
        dx = wdy * rstd - x * a - b_s
        tl.store(dx_ptr + row_base + offs, dx, mask=mask)
    else:
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        offs0 = tl.arange(0, BLOCK_N)
        acc_c1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_c2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        p_x = x_ptr + row_base + offs0
        p_dy = dy_ptr + row_base + offs0
        p_w = w_ptr + offs0
        col = 0
        for _ in range(0, NUM_TILES):
            mask = (col + offs0) < N
            x = tl.load(p_x, mask=mask, other=0.0)
            dy = tl.load(p_dy, mask=mask, other=0.0)
            w = tl.load(p_w, mask=mask, other=0.0)
            xhat = (x - mean) * rstd
            wdy = dy * w
            acc_c1 = acc_c1 + xhat * wdy
            acc_c2 = acc_c2 + wdy
            p_x = p_x + BLOCK_N
            p_dy = p_dy + BLOCK_N
            p_w = p_w + BLOCK_N
            col = col + BLOCK_N
        c1 = tl.sum(acc_c1, axis=0) * inv_n
        c2 = tl.sum(acc_c2, axis=0) * inv_n

        p_x2 = x_ptr + row_base + offs0
        p_dy2 = dy_ptr + row_base + offs0
        p_w2 = w_ptr + offs0
        p_dx = dx_ptr + row_base + offs0
        col2 = 0
        for _ in range(0, NUM_TILES):
            mask = (col2 + offs0) < N
            x = tl.load(p_x2, mask=mask, other=0.0)
            dy = tl.load(p_dy2, mask=mask, other=0.0)
            w = tl.load(p_w2, mask=mask, other=0.0)
            xhat = (x - mean) * rstd
            wdy = dy * w
            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(p_dx, dx, mask=mask)
            p_x2 = p_x2 + BLOCK_N
            p_dy2 = p_dy2 + BLOCK_N
            p_w2 = p_w2 + BLOCK_N
            p_dx = p_dx + BLOCK_N
            col2 = col2 + BLOCK_N


@m1triton.autotune(
    configs=[
        m1triton.Config({"GROUP_SIZE_M": 32, "BLOCK_N": 256}, num_warps=8),
        m1triton.Config({"GROUP_SIZE_M": 64, "BLOCK_N": 256}, num_warps=8),
        m1triton.Config({"GROUP_SIZE_M": 64, "BLOCK_N": 512}, num_warps=8),
        m1triton.Config({"GROUP_SIZE_M": 128, "BLOCK_N": 256}, num_warps=8),
        m1triton.Config({"GROUP_SIZE_M": 128, "BLOCK_N": 512}, num_warps=8),
        m1triton.Config({"GROUP_SIZE_M": 256, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
    reset_to_zero=("dw_ptr", "db_ptr"),
)
@m1triton.heuristics({"EVEN_N": lambda args: (args["N"] % args["BLOCK_N"]) == 0})
@m1triton.jit(fast_math=True)
def _ln_bwd_dwdb_atomic_kernel(
    x_ptr,
    dy_ptr,
    mean_ptr,
    rstd_ptr,
    dw_ptr,
    db_ptr,
    M,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    n_i64 = N.to(tl.int64)
    offs = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask = offs < n_i64

    rows_per_split = (M + GROUP_SIZE_M - 1) // GROUP_SIZE_M
    row_start = pid_s * rows_per_split
    row_end = tl.minimum(row_start + rows_per_split, M)

    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    if row_start < row_end:
        row_end4 = row_start + ((row_end - row_start) // 4) * 4
        p_x = x_ptr + row_start.to(tl.int64) * n_i64 + offs
        p_dy = dy_ptr + row_start.to(tl.int64) * n_i64 + offs

        for r4 in range(row_start, row_end4, 4):
            r0 = r4.to(tl.int64)
            m0 = tl.load(mean_ptr + r0)
            s0 = tl.load(rstd_ptr + r0)
            m1 = tl.load(mean_ptr + (r0 + 1))
            s1 = tl.load(rstd_ptr + (r0 + 1))
            m2 = tl.load(mean_ptr + (r0 + 2))
            s2 = tl.load(rstd_ptr + (r0 + 2))
            m3 = tl.load(mean_ptr + (r0 + 3))
            s3 = tl.load(rstd_ptr + (r0 + 3))

            if EVEN_N:
                x0 = tl.load(p_x)
                d0 = tl.load(p_dy)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x1 = tl.load(p_x)
                d1 = tl.load(p_dy)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x2 = tl.load(p_x)
                d2 = tl.load(p_dy)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x3 = tl.load(p_x)
                d3 = tl.load(p_dy)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
            else:
                x0 = tl.load(p_x, mask=mask, other=0.0)
                d0 = tl.load(p_dy, mask=mask, other=0.0)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x1 = tl.load(p_x, mask=mask, other=0.0)
                d1 = tl.load(p_dy, mask=mask, other=0.0)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x2 = tl.load(p_x, mask=mask, other=0.0)
                d2 = tl.load(p_dy, mask=mask, other=0.0)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64
                x3 = tl.load(p_x, mask=mask, other=0.0)
                d3 = tl.load(p_dy, mask=mask, other=0.0)
                p_x = p_x + n_i64
                p_dy = p_dy + n_i64

            dw_acc = (
                dw_acc
                + d0 * ((x0 - m0) * s0)
                + d1 * ((x1 - m1) * s1)
                + d2 * ((x2 - m2) * s2)
                + d3 * ((x3 - m3) * s3)
            )
            db_acc = db_acc + d0 + d1 + d2 + d3

        for r_rem in range(row_end4, row_end):
            ri = r_rem.to(tl.int64)
            mr = tl.load(mean_ptr + ri)
            sr = tl.load(rstd_ptr + ri)
            if EVEN_N:
                xr = tl.load(p_x)
                dr = tl.load(p_dy)
            else:
                xr = tl.load(p_x, mask=mask, other=0.0)
                dr = tl.load(p_dy, mask=mask, other=0.0)
            dw_acc = dw_acc + dr * ((xr - mr) * sr)
            db_acc = db_acc + dr
            p_x = p_x + n_i64
            p_dy = p_dy + n_i64

        if EVEN_N:
            tl.atomic_add(dw_ptr + offs, dw_acc)
            tl.atomic_add(db_ptr + offs, db_acc)
        else:
            tl.atomic_add(dw_ptr + offs, dw_acc, mask=mask)
            tl.atomic_add(db_ptr + offs, db_acc, mask=mask)


@m1triton.jit
def _ln_bwd_zero_dwdb_kernel(dw_ptr, db_ptr, N, BLOCK_N: tl.constexpr):
    tlx.vec_width(4)
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    zeros = tl.zeros((BLOCK_N,), dtype=tl.float32)
    tl.store(dw_ptr + offs, zeros, mask=mask)
    tl.store(db_ptr + offs, zeros, mask=mask)


_TUNED_FWD_KEYS: set[tuple[int, int, torch.dtype]] = set()
_TUNED_BWD_DX_KEYS: set[tuple[int, int, torch.dtype]] = set()
_TUNED_BWD_DWDB_KEYS: set[tuple[int, int, torch.dtype]] = set()


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


class _FusedLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_c = _contig(x)
        w_c = _contig(weight)
        b_c = _contig(bias)

        n = int(x_c.shape[-1])
        m = int(x_c.numel()) // n
        inv_n = 1.0 / float(n)
        eps_f = float(eps)

        y = torch.empty_like(x_c)
        mean = torch.empty((m,), device=x_c.device, dtype=torch.float32)
        rstd = torch.empty((m,), device=x_c.device, dtype=torch.float32)

        fwd_grid = (m,)
        fwd_key = (m, n, x_c.dtype)
        if fwd_key not in _TUNED_FWD_KEYS:
            _ln_fwd_kernel._select_config(
                fwd_grid,
                (x_c, w_c, b_c, y, mean, rstd, m, n, inv_n, eps_f),
                {},
            )
            _TUNED_FWD_KEYS.add(fwd_key)

        with m1_rt.command_batch(wait=True, sync_mps=True):
            _ln_fwd_kernel[fwd_grid](
                x_c,
                w_c,
                b_c,
                y,
                mean,
                rstd,
                m,
                n,
                inv_n,
                eps_f,
                sync_mps=False,
                wait=False,
            )

        ctx.save_for_backward(x_c, w_c, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c, mean, rstd = ctx.saved_tensors
        dy_c = _contig(dy)

        n = int(x_c.shape[-1])
        m = int(x_c.numel()) // n
        inv_n = 1.0 / float(n)

        dx = torch.empty_like(x_c)
        dweight = torch.empty((n,), device=x_c.device, dtype=torch.float32)
        dbias = torch.empty((n,), device=x_c.device, dtype=torch.float32)

        dx_grid = (m,)
        dwdb_grid = lambda meta: (tl.cdiv(n, meta["BLOCK_N"]), min(m, meta["GROUP_SIZE_M"]))
        bwd_key = (m, n, x_c.dtype)

        if bwd_key not in _TUNED_BWD_DX_KEYS:
            _ln_bwd_dx_kernel._select_config(
                dx_grid,
                (x_c, dy_c, w_c, mean, rstd, dx, m, n, inv_n),
                {},
            )
            _TUNED_BWD_DX_KEYS.add(bwd_key)

        if bwd_key not in _TUNED_BWD_DWDB_KEYS:
            _ln_bwd_dwdb_atomic_kernel._select_config(
                dwdb_grid,
                (x_c, dy_c, mean, rstd, dweight, dbias, m, n),
                {},
            )
            _TUNED_BWD_DWDB_KEYS.add(bwd_key)

        with m1_rt.command_batch(wait=True, sync_mps=True):
            _ln_bwd_zero_dwdb_kernel[(tl.cdiv(n, 256),)](
                dweight,
                dbias,
                n,
                BLOCK_N=256,
                sync_mps=False,
                wait=False,
            )
            _ln_bwd_dx_kernel[dx_grid](
                x_c,
                dy_c,
                w_c,
                mean,
                rstd,
                dx,
                m,
                n,
                inv_n,
                sync_mps=False,
                wait=False,
            )
            _ln_bwd_dwdb_atomic_kernel[dwdb_grid](
                x_c,
                dy_c,
                mean,
                rstd,
                dweight,
                dbias,
                m,
                n,
                sync_mps=False,
                wait=False,
            )

        return dx, dweight, dbias, None


def candidate_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused LayerNorm forward and backward with Metal 3 atomic parameter gradient accumulation."""
    return _FusedLayerNormFn.apply(x, weight, bias, float(eps))
