"""Unit tests for M4-B coalesced strided and last-axis reduction kernels (#47-#49, #51-#53)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = REPO_ROOT / "benchmarks" / "kernels"
if str(KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(KERNELS_DIR))
if str(REPO_ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evals"))

import m1triton.runtime as rt
from _axis_reduce_common import (
    axis_argmax,
    axis_argmin,
    axis_argextrema_strided_kernel,
    axis_extrema_strided_kernel,
    axis_max,
    axis_mean,
    axis_min,
    axis_sum,
    axis_sum_strided_kernel,
)
from eval_kernelbench import TorchComputeGuard, static_ast_audit

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Silicon MPS backend is required for M1Triton GPU tests",
)


def _load_task_module(task_id: int):
    path = KERNELS_DIR / f"task_{task_id:03d}.py"
    spec = importlib.util.spec_from_file_location(f"_m4_test_task_{task_id}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


@pytest.mark.parametrize("shape", [(4, 17, 300), (3, 1, 130), (2, 33, 5), (5, 19, 1)])
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
def test_all_six_ops_general_dims_and_negative_values(shape: tuple[int, int, int], dim: int):
    torch.manual_seed(100 + sum(shape) + dim * 7)
    x = torch.randn(shape, device="mps", dtype=torch.float32) * 10.0 - 3.5
    torch.mps.synchronize()

    before = rt.get_dispatch_count()
    with TorchComputeGuard():
        out_sum = axis_sum(x, dim, keepdim=True)
        out_mean = axis_mean(x, dim, keepdim=False)
        out_max = axis_max(x, dim, keepdim=False)
        out_min = axis_min(x, dim, keepdim=False)
        out_argmax = axis_argmax(x, dim, keepdim=False)
        out_argmin = axis_argmin(x, dim, keepdim=False)
    rt.synchronize()
    after = rt.get_dispatch_count()
    assert after == before + 6

    ref_sum = torch.sum(x, dim=dim, keepdim=True)
    ref_mean = torch.mean(x, dim=dim)
    ref_max = torch.max(x, dim=dim)[0]
    ref_min = torch.min(x, dim=dim)[0]
    ref_argmax = torch.argmax(x, dim=dim)
    ref_argmin = torch.argmin(x, dim=dim)
    torch.mps.synchronize()

    assert out_sum.shape == ref_sum.shape and out_sum.dtype == torch.float32
    assert out_mean.shape == ref_mean.shape and out_mean.dtype == torch.float32
    assert out_max.shape == ref_max.shape and out_max.dtype == torch.float32
    assert out_min.shape == ref_min.shape and out_min.dtype == torch.float32
    assert out_argmax.shape == ref_argmax.shape and out_argmax.dtype == torch.int64
    assert out_argmin.shape == ref_argmin.shape and out_argmin.dtype == torch.int64

    torch.testing.assert_close(out_sum.cpu(), ref_sum.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_mean.cpu(), ref_mean.cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_max.cpu(), ref_max.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_min.cpu(), ref_min.cpu(), rtol=1e-5, atol=1e-5)
    assert torch.equal(out_argmax.cpu(), ref_argmax.cpu())
    assert torch.equal(out_argmin.cpu(), ref_argmin.cpu())


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_argmax_argmin_tie_breaking_first_index(dim: int):
    # Deliberately create repeated max and min values across multiple positions,
    # including boundary tiles, all-equal rows, and strictly negative tensors.
    shape = (3, 13, 131)
    x_cpu = torch.full(shape, -7.5, dtype=torch.float32)

    if dim == 1:
        # Column 0: all equal (-7.5) -> first index must be 0
        # Column 1: max tie at r=2 and r=11 -> must return 2
        x_cpu[0, 2, 1] = -1.25
        x_cpu[0, 11, 1] = -1.25
        # Column 130 (partial boundary tile): min tie at r=4 and r=9 -> must return 4
        x_cpu[2, 4, 130] = -99.0
        x_cpu[2, 9, 130] = -99.0
        # Column 64: max at r=0 and r=12 -> must return 0
        x_cpu[1, 0, 64] = 42.0
        x_cpu[1, 12, 64] = 42.0
    elif dim == 2:
        # Last-axis tie-breaking within same SIMD group and across SIMD groups
        x_cpu[0, 0, 2] = 50.0
        x_cpu[0, 0, 5] = 50.0
        x_cpu[0, 1, 3] = 50.0
        x_cpu[0, 1, 70] = 50.0
        x_cpu[1, 2, 4] = -50.0
        x_cpu[1, 2, 9] = -50.0
        x_cpu[1, 3, 1] = -50.0
        x_cpu[1, 3, 129] = -50.0
    else:
        x_cpu[1, 3, 129] = 10.0
        x_cpu[2, 3, 129] = 10.0
        x_cpu[1, 5, 0] = -20.0
        x_cpu[2, 5, 0] = -20.0

    x = x_cpu.to("mps")
    torch.mps.synchronize()

    with TorchComputeGuard():
        got_max = axis_argmax(x, dim)
        got_min = axis_argmin(x, dim)
    rt.synchronize()

    ref_max = torch.argmax(x, dim=dim)
    ref_min = torch.argmin(x, dim=dim)
    torch.mps.synchronize()

    assert got_max.dtype == torch.int64
    assert got_min.dtype == torch.int64
    assert torch.equal(got_max.cpu(), ref_max.cpu())
    assert torch.equal(got_min.cpu(), ref_min.cpu())


def test_1d_and_large_last_axis_and_non_contiguous():
    # 1D tensor reducing along dim=0 (produces 0D scalar output)
    v = torch.randn((1537,), device="mps", dtype=torch.float32)
    v[10] = 100.0
    v[1200] = 100.0
    v[7] = -100.0
    v[1500] = -100.0
    torch.mps.synchronize()

    with TorchComputeGuard():
        s = axis_sum(v, 0, keepdim=False)
        m = axis_mean(v, 0, keepdim=False)
        mx = axis_max(v, 0, keepdim=False)
        mn = axis_min(v, 0, keepdim=False)
        amx = axis_argmax(v, 0, keepdim=False)
        amn = axis_argmin(v, 0, keepdim=False)
    rt.synchronize()

    assert s.shape == ()
    assert amx.shape == () and amx.dtype == torch.int64
    assert amn.shape == () and amn.dtype == torch.int64
    torch.testing.assert_close(s.cpu(), torch.sum(v, dim=0).cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(m.cpu(), torch.mean(v, dim=0).cpu(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(mx.cpu(), torch.max(v, dim=0)[0].cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mn.cpu(), torch.min(v, dim=0)[0].cpu(), rtol=1e-5, atol=1e-5)
    assert int(amx.cpu().item()) == 10
    assert int(amn.cpu().item()) == 7

    # Non-contiguous transposed view
    t = torch.randn((8, 25, 130), device="mps", dtype=torch.float32).transpose(0, 1)
    assert not t.is_contiguous()
    torch.mps.synchronize()
    with TorchComputeGuard():
        got = axis_sum(t, 1, keepdim=True)
    rt.synchronize()
    torch.testing.assert_close(got.cpu(), torch.sum(t, dim=1, keepdim=True).cpu(), rtol=1e-4, atol=1e-4)


def test_generated_msl_uses_64bit_pointer_math():
    x = torch.empty((4, 17, 300), device="mps", dtype=torch.float32)
    out_f = torch.empty((4, 300), device="mps", dtype=torch.float32)
    out_i = torch.empty((4, 300), device="mps", dtype=torch.int64)

    msl_sum = axis_sum_strided_kernel.inspect_msl(x, out_f, 17, 300, 1.0, BLOCK_INNER=128, APPLY_SCALE=0)
    msl_max = axis_extrema_strided_kernel.inspect_msl(x, out_f, 17, 300, BLOCK_INNER=128, IS_MAX=1)
    msl_arg = axis_argextrema_strided_kernel.inspect_msl(x, out_i, 17, 300, BLOCK_INNER=128, IS_MAX=1)

    for msl in (msl_sum, msl_max, msl_arg):
        assert "long " in msl
        assert "m1_lane < 128u; m1_lane += 128u" in msl
    assert "device long* out_ptr" in msl_arg


@pytest.mark.parametrize("task_id", [47, 48, 49, 51, 52, 53])
def test_task_wrappers_anticheat_and_small_parity(task_id: int):
    mod, path = _load_task_module(task_id)
    assert static_ast_audit(path) == []

    torch.manual_seed(500 + task_id)
    x = torch.randn((4, 19, 263), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    model = mod.ModelNew(1).to("mps")
    before = rt.get_dispatch_count()
    with torch.no_grad(), TorchComputeGuard():
        out = model(x)
    rt.synchronize()
    assert rt.get_dispatch_count() > before

    if task_id == 47:
        ref = torch.sum(x, dim=1, keepdim=True)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    elif task_id == 48:
        ref = torch.mean(x, dim=1)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)
    elif task_id == 49:
        ref = torch.max(x, dim=1)[0]
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-5, atol=1e-5)
    elif task_id == 51:
        ref = torch.argmax(x, dim=1)
        assert out.dtype == torch.int64
        assert torch.equal(out.cpu(), ref.cpu())
    elif task_id == 52:
        ref = torch.argmin(x, dim=1)
        assert out.dtype == torch.int64
        assert torch.equal(out.cpu(), ref.cpu())
    elif task_id == 53:
        ref = torch.min(x, dim=1)[0]
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-5, atol=1e-5)
