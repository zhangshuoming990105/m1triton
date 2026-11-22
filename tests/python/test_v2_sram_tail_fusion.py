"""End-to-end Apple M1 Max GPU tests for V2-M3a SRAM/register epilogue DAG fusion
in Norm/Softmax, Pool, and Axis-Reduce Metal emitters."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import m1triton
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.tlx as tlx

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Silicon MPS backend is required for M1Triton GPU tests",
)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@m1triton.jit
def _gn_stats_kernel(
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
def _gn_apply_tanh_hardswish_res_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    res_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    CHUNKS: tl.constexpr,
    GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.epilogue([("tanh",), ("hardswish",), ("add_residual", res_ptr)])
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
def _softmax_bias_scale_sigmoid_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,
    scale,
    NUM_ROWS: tl.constexpr,
    ROW_LEN: tl.constexpr,
):
    tlx.epilogue([("add_bias", bias_ptr), ("mul_scalar", scale), ("sigmoid",)])
    tlx.online_softmax(x_ptr, out_ptr, NUM_ROWS, ROW_LEN, axis=1, log_softmax=False)


@m1triton.jit
def _ln_stats_kernel(x_ptr, partial_ptr, N: tl.constexpr, NORM_SIZE: tl.constexpr, CHUNKS: tl.constexpr):
    tlx.welford_norm("layer_norm", x_ptr, partial_ptr, N, NORM_SIZE, CHUNKS, stage=1)


@m1triton.jit
def _ln_apply_bias_gelu_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    partial_ptr,
    extra_bias_ptr,
    N: tl.constexpr,
    NORM_SIZE: tl.constexpr,
    CHUNKS: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.epilogue([("add_bias", extra_bias_ptr), ("gelu",)])
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


@m1triton.jit
def _rms_bias_tanh_res_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,
    res_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.epilogue([("add_bias", bias_ptr), ("tanh",), ("add_residual", res_ptr)])
    tlx.welford_norm("rms_norm", x_ptr, out_ptr, N, C, HW, eps=EPS, axis=1)


@m1triton.jit
def _maxpool2d_hardtanh_scale_kernel(
    x_ptr,
    out_ptr,
    scale,
    N,
    C,
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
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VEC_W: tl.constexpr = 1,
    ROWS_PER_THREAD: tl.constexpr = 0,
):
    tlx.epilogue([("hardtanh", -1.25, 1.5), ("mul_scalar", scale)])
    tlx.pool(
        x_ptr,
        out_ptr,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        kind="max",
        ndim=2,
        kernel_size=(K_H, K_W),
        stride=(S_H, S_W),
        padding=(P_H, P_W),
        dilation=(1, 1),
        count_include_pad=True,
        ceil_mode=False,
    )


@m1triton.jit
def _avgpool3d_gelu_kernel(
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
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_D: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_D: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tlx.epilogue([("gelu",)])
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
        kind="avg",
        ndim=3,
        kernel_size=(K_D, K_H, K_W),
        stride=(S_D, S_H, S_W),
        padding=(P_D, P_H, P_W),
        dilation=(1, 1, 1),
        count_include_pad=True,
        ceil_mode=False,
    )


@m1triton.jit
def _axis_sum_scale_mish_kernel(
    x_ptr,
    out_ptr,
    R,
    scale,
    BLOCK_K: tl.constexpr,
):
    tlx.epilogue([("mul_scalar", scale), ("mish",)])
    row = tl.program_id(0)
    row_base = x_ptr + row.to(tl.int64) * R.to(tl.int64)
    ar = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, R, BLOCK_K):
        k_offs = k + ar
        k_mask = k_offs < R
        vals = tl.load(row_base + k_offs.to(tl.int64), mask=k_mask, other=0.0)
        acc = acc + vals
    res = tl.sum(acc, axis=0)
    tl.store(out_ptr + row.to(tl.int64), res)


@m1triton.jit
def _axis_sum_2d_bias_mish_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tlx.epilogue([("add_bias", bias_ptr), ("mul_const", 0.5), ("mish",)])
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    rmask = rows < M
    cmask = cols < N
    ptrs = x_ptr + rows[:, None].to(tl.int64) * N.to(tl.int64) + cols[None, :].to(tl.int64)
    vals = tl.load(ptrs, mask=rmask[:, None] & cmask[None, :], other=0.0)
    row_sum = tl.sum(vals, axis=1)
    tl.store(out_ptr + rows.to(tl.int64), row_sum, mask=rmask)


@pytest.mark.parametrize("shape,groups", [((4, 16, 16, 16), 4), ((3, 12, 13, 19), 3)])
def test_fused_groupnorm_tanh_hardswish_residual(shape: tuple[int, int, int, int], groups: int):
    torch.manual_seed(11 + groups)
    n, c, h, w = shape
    hw = h * w
    eps = 1e-5
    x = (torch.randn(shape, device="mps", dtype=torch.float32) * 3.0).contiguous()
    weight = torch.randn((c,), device="mps", dtype=torch.float32)
    bias = torch.randn((c,), device="mps", dtype=torch.float32)
    residual = torch.randn(shape, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    chunks = 2 if (c // groups) * hw >= 512 else 1
    partial = torch.empty((n * groups * chunks * 3,), device="mps", dtype=torch.float32)
    tiles = max(1, _cdiv(hw, 8192))
    torch.mps.synchronize()

    msl = _gn_apply_tanh_hardswish_res_kernel.inspect_msl(
        x, out, weight, bias, partial, residual, n, c, hw, chunks, groups, eps
    )
    assert "m1_tanh(" in msl
    assert "m1p_res_ptr[m1_ep_idx]" in msl

    before = rt.get_dispatch_count()
    _gn_stats_kernel[(chunks, n * groups, 1)](x, partial, n, c, hw, chunks, groups)
    _gn_apply_tanh_hardswish_res_kernel[(tiles, n * c, 1)](
        x, out, weight, bias, partial, residual, n, c, hw, chunks, groups, eps, wait=True
    )
    assert rt.get_dispatch_count() == before + 2

    gn_ref = F.group_norm(x.cpu(), groups, weight.cpu(), bias.cpu(), eps=eps)
    ref = F.hardswish(torch.tanh(gn_ref)) + residual.cpu()
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rows,cols", [(32, 256), (17, 259), (8, 1031)])
def test_fused_softmax_bias_scale_sigmoid(rows: int, cols: int):
    torch.manual_seed(23 + cols)
    x = (torch.randn((rows, cols), device="mps", dtype=torch.float32) * 4.0).contiguous()
    bias = torch.randn((cols,), device="mps", dtype=torch.float32)
    scale = 2.75
    out = torch.empty_like(x)
    torch.mps.synchronize()

    msl = _softmax_bias_scale_sigmoid_kernel.inspect_msl(x, out, bias, scale, rows, cols)
    assert "m1p_bias_ptr[" in msl
    assert "float(m1p_scale)" in msl

    before = rt.get_dispatch_count()
    _softmax_bias_scale_sigmoid_kernel[(rows, 1, 1)](x, out, bias, scale, rows, cols, wait=True)
    assert rt.get_dispatch_count() == before + 1

    ref = torch.sigmoid((torch.softmax(x.cpu(), dim=-1) + bias.cpu()) * scale)
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n,c,hw", [(2, 64, 128), (2, 72, 131)])
def test_fused_rmsnorm_and_layernorm_epilogue(n: int, c: int, hw: int):
    torch.manual_seed(31 + c)
    eps = 1e-5
    x = (torch.randn((n, c, hw), device="mps", dtype=torch.float32) * 2.5).contiguous()
    bias = torch.randn((c,), device="mps", dtype=torch.float32)
    res = torch.randn((n, c, hw), device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    before = rt.get_dispatch_count()
    _rms_bias_tanh_res_kernel[(_cdiv(hw, 64), n, 1)](x, out, bias, res, n, c, hw, eps, wait=True)
    assert rt.get_dispatch_count() == before + 1

    xc = x.cpu()
    rms_ref = xc * torch.rsqrt(xc.pow(2).mean(dim=1, keepdim=True) + eps)
    ref = torch.tanh(rms_ref + bias.cpu().view(1, c, 1)) + res.cpu()
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    ln_n, ln_l = 8, 261
    lx = (torch.randn((ln_n, ln_l), device="mps", dtype=torch.float32) * 2.0).contiguous()
    lw = torch.randn((ln_l,), device="mps", dtype=torch.float32)
    lb = torch.randn((ln_l,), device="mps", dtype=torch.float32)
    lextra = torch.randn((ln_l,), device="mps", dtype=torch.float32)
    lout = torch.empty_like(lx)
    lpart = torch.empty((ln_n * 3,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    _ln_stats_kernel[(1, ln_n, 1)](lx, lpart, ln_n, ln_l, 1)
    _ln_apply_bias_gelu_kernel[(1, ln_n, 1)](lx, lout, lw, lb, lpart, lextra, ln_n, ln_l, 1, eps, wait=True)
    lref = F.gelu(F.layer_norm(lx.cpu(), (ln_l,), lw.cpu(), lb.cpu(), eps=eps) + lextra.cpu())
    torch.testing.assert_close(lout.cpu(), lref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape,k,s,p,bw,bh,vec_w,rh",
    [
        ((2, 8, 32, 32), (4, 4), (1, 1), (1, 1), 128, 32, 4, 8),
        ((2, 6, 27, 29), (3, 3), (1, 1), (1, 1), 128, 16, 4, 4),
        ((2, 8, 32, 32), (2, 2), (2, 2), (0, 0), 32, 8, 4, 0),
        ((2, 6, 23, 25), (3, 3), (2, 2), (1, 1), 32, 32, 1, 0),
    ],
)
def test_fused_maxpool2d_hardtanh_scale(
    shape: tuple[int, int, int, int],
    k: tuple[int, int],
    s: tuple[int, int],
    p: tuple[int, int],
    bw: int,
    bh: int,
    vec_w: int,
    rh: int,
):
    torch.manual_seed(41 + shape[2] + k[0])
    n, c, h_in, w_in = shape
    h_out = (h_in + 2 * p[0] - k[0]) // s[0] + 1
    w_out = (w_in + 2 * p[1] - k[1]) // s[1] + 1
    x = (torch.randn(shape, device="mps", dtype=torch.float32) * 3.0).contiguous()
    scale = 0.65
    out = torch.empty((n, c, h_out, w_out), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    grid = (_cdiv(w_out, bw), _cdiv(h_out, bh), n * c)
    before = rt.get_dispatch_count()
    _maxpool2d_hardtanh_scale_kernel[grid](
        x,
        out,
        scale,
        n,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        K_H=k[0],
        K_W=k[1],
        S_H=s[0],
        S_W=s[1],
        P_H=p[0],
        P_W=p[1],
        BLOCK_W=bw,
        BLOCK_H=bh,
        VEC_W=vec_w,
        ROWS_PER_THREAD=rh,
        wait=True,
    )
    assert rt.get_dispatch_count() == before + 1

    pooled = F.max_pool2d(x.cpu(), kernel_size=k, stride=s, padding=p)
    ref = F.hardtanh(pooled, min_val=-1.25, max_val=1.5) * scale
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape,k,s,p", [((2, 4, 12, 16, 18), (2, 2, 2), (2, 2, 2), (0, 0, 0)), ((2, 3, 9, 11, 13), (3, 3, 3), (2, 2, 2), (1, 1, 1))])
def test_fused_avgpool3d_gelu(
    shape: tuple[int, int, int, int, int],
    k: tuple[int, int, int],
    s: tuple[int, int, int],
    p: tuple[int, int, int],
):
    torch.manual_seed(53 + shape[2])
    n, c, d_in, h_in, w_in = shape
    d_out = (d_in + 2 * p[0] - k[0]) // s[0] + 1
    h_out = (h_in + 2 * p[1] - k[1]) // s[1] + 1
    w_out = (w_in + 2 * p[2] - k[2]) // s[2] + 1
    x = (torch.randn(shape, device="mps", dtype=torch.float32) * 3.0).contiguous()
    out = torch.empty((n, c, d_out, h_out, w_out), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    bw, bh = 32, 8
    msl = _avgpool3d_gelu_kernel.inspect_msl(
        x,
        out,
        n,
        c,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        K_D=k[0],
        K_H=k[1],
        K_W=k[2],
        S_D=s[0],
        S_H=s[1],
        S_W=s[2],
        P_D=p[0],
        P_H=p[1],
        P_W=p[2],
        BLOCK_W=bw,
        BLOCK_H=bh,
    )
    assert "m1_erf(" in msl

    grid = (_cdiv(w_out, bw), _cdiv(h_out, bh), n * c * d_out)
    before = rt.get_dispatch_count()
    _avgpool3d_gelu_kernel[grid](
        x,
        out,
        n,
        c,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        K_D=k[0],
        K_H=k[1],
        K_W=k[2],
        S_D=s[0],
        S_H=s[1],
        S_W=s[2],
        P_D=p[0],
        P_H=p[1],
        P_W=p[2],
        BLOCK_W=bw,
        BLOCK_H=bh,
        wait=True,
    )
    assert rt.get_dispatch_count() == before + 1

    ref = F.gelu(F.avg_pool3d(x.cpu(), kernel_size=k, stride=s, padding=p, count_include_pad=True))
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rows,reduce_len,block_k", [(32, 256, 256), (19, 613, 256), (17, 95, 64)])
def test_fused_axis_sum_scale_mish(rows: int, reduce_len: int, block_k: int):
    torch.manual_seed(67 + reduce_len)
    x = (torch.randn((rows, reduce_len), device="mps", dtype=torch.float32) * 0.25).contiguous()
    scale = 0.75
    out = torch.empty((rows,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    msl = _axis_sum_scale_mish_kernel.inspect_msl(x, out, reduce_len, scale, BLOCK_K=block_k)
    assert "simd_sum(" in msl
    assert "m1_tanh(" in msl

    before = rt.get_dispatch_count()
    _axis_sum_scale_mish_kernel[(rows,)](x, out, reduce_len, scale, BLOCK_K=block_k, wait=True)
    assert rt.get_dispatch_count() == before + 1

    ref = F.mish(torch.sum(x.cpu(), dim=-1) * scale)
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_fused_2d_row_reduce_bias_mish():
    torch.manual_seed(79)
    m, n = 23, 61
    bm, bn = 8, 64
    x = (torch.randn((m, n), device="mps", dtype=torch.float32) * 0.2).contiguous()
    bias = torch.randn((m,), device="mps", dtype=torch.float32)
    out = torch.empty((m,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    before = rt.get_dispatch_count()
    _axis_sum_2d_bias_mish_kernel[(_cdiv(m, bm),)](x, out, bias, m, n, BM=bm, BN=bn, wait=True)
    assert rt.get_dispatch_count() == before + 1

    ref = F.mish((torch.sum(x.cpu(), dim=-1) + bias.cpu()) * 0.5)
    got = out.cpu()
    assert not torch.isnan(got).any() and not torch.isinf(got).any()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_extreme_values_and_nan_safety():
    # Extreme magnitudes (+/- 60) must saturate cleanly without NaN/Inf in Tanh, Mish, Sigmoid, GELU.
    x_ext = torch.tensor(
        [[-60.0, -25.0, -5.0, 0.0, 5.0, 25.0, 60.0, 1.5] * 4],
        device="mps",
        dtype=torch.float32,
    ).contiguous()
    rows, cols = x_ext.shape
    bias = torch.linspace(-20.0, 20.0, cols, device="mps", dtype=torch.float32)
    out_sm = torch.empty_like(x_ext)
    torch.mps.synchronize()
    _softmax_bias_scale_sigmoid_kernel[(rows, 1, 1)](x_ext, out_sm, bias, 3.0, rows, cols, wait=True)
    got_sm = out_sm.cpu()
    assert not torch.isnan(got_sm).any() and not torch.isinf(got_sm).any()
    ref_sm = torch.sigmoid((torch.softmax(x_ext.cpu(), dim=-1) + bias.cpu()) * 3.0)
    torch.testing.assert_close(got_sm, ref_sm, rtol=1e-4, atol=1e-4)

    # Extreme values through AxisSum + Scale + Mish (tests softplus overflow guard > 20.0f and m1_tanh clamp).
    x_mish = torch.full((4, 128), 1.0, device="mps", dtype=torch.float32)
    x_mish[0] = 0.5
    x_mish[1] = -0.5
    x_mish[2] = 0.01
    x_mish[3] = -0.01
    out_mish = torch.empty((4,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    _axis_sum_scale_mish_kernel[(4,)](x_mish, out_mish, 128, 1.0, BLOCK_K=128, wait=True)
    got_mish = out_mish.cpu()
    assert not torch.isnan(got_mish).any() and not torch.isinf(got_mish).any()
    ref_mish = F.mish(torch.sum(x_mish.cpu(), dim=-1))
    torch.testing.assert_close(got_mish, ref_mish, rtol=1e-4, atol=1e-4)

    # Isolated NaN in MaxPool2D propagates to its own output window without corrupting unrelated windows.
    x_pool = torch.randn((1, 2, 16, 16), device="mps", dtype=torch.float32).contiguous()
    x_pool[0, 0, 0, 0] = float("nan")
    out_pool = torch.empty((1, 2, 8, 8), device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    _maxpool2d_hardtanh_scale_kernel[(1, 1, 2)](
        x_pool,
        out_pool,
        1.0,
        1,
        2,
        16,
        16,
        8,
        8,
        K_H=2,
        K_W=2,
        S_H=2,
        S_W=2,
        P_H=0,
        P_W=0,
        BLOCK_W=32,
        BLOCK_H=8,
        VEC_W=4,
        ROWS_PER_THREAD=0,
        wait=True,
    )
    got_pool = out_pool.cpu()
    assert torch.isnan(got_pool[0, 0, 0, 0])
    assert not torch.isnan(got_pool[0, 1]).any()
    assert not torch.isnan(got_pool[0, 0, 1:, :]).any()


@m1triton.jit
def _epilogue_hardswish_minmax_kernel(x_ptr, bias_ptr, out_ptr, N, BLOCK: tl.constexpr):
    tlx.epilogue([("min_bias", bias_ptr), ("hardswish",)])
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    res = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid.to(tl.int64), res)


@m1triton.jit
def _epilogue_nan_const_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    tlx.epilogue([("min_const", float("nan"))])
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    res = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid.to(tl.int64), res)


def test_epilogue_dag_hardswish_inf_and_nan_propagation():
    # HardSwish(-inf) must be 0.0 (not NaN), HardSwish(+inf) == +inf, and NaN in bias must propagate.
    x = torch.tensor(
        [[-float("inf"), 0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        device="mps",
        dtype=torch.float32,
    ).contiguous()
    bias = torch.tensor([10.0, float("inf"), float("nan"), 1.5], device="mps", dtype=torch.float32)
    out = torch.empty((4,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    _epilogue_hardswish_minmax_kernel[(4,)](x, bias, out, 16, BLOCK=4, wait=True)
    got = out.cpu()
    assert got[0].item() == 0.0 and not torch.isnan(got[0])
    assert torch.isinf(got[1]) and got[1].item() > 0.0
    assert torch.isnan(got[2])
    torch.testing.assert_close(got[3:4], F.hardswish(torch.tensor([1.5])), rtol=1e-4, atol=1e-4)

    # NaN constant literal in epilogue must compile as valid MSL `NAN` and produce NaN.
    x2 = torch.ones((4,), device="mps", dtype=torch.float32)
    out2 = torch.empty((1,), device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    _epilogue_nan_const_kernel[(1,)](x2, out2, 4, BLOCK=4, wait=True)
    assert torch.isnan(out2.cpu()[0])

    # tlx.epilogue helper validates param operand & preserves Softplus(1.0, 0.0) threshold=0.
    sp = tlx.epilogue([("softplus", 1.0, 0.0)])
    assert sp.steps[0].param0 == 1.0 and sp.steps[0].param1 == 0.0
    with pytest.raises(ValueError):
        tlx.epilogue([("add_bias",)])
    with pytest.raises(TypeError):
        tlx.epilogue([("add_bias", "not_an_int")])
