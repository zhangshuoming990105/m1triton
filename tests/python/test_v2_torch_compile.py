"""Tests for V2-M3b `torch.compile(model, backend="m1triton")` FX graph fusion compiler."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import m1triton
import m1triton.compile_backend as cb
import m1triton.runtime as rt

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Silicon MPS backend is required for M1Triton GPU tests",
)


def _run_and_verify(
    fn,
    *args,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    strict: bool = True,
    expect_dispatches: int | None = None,
    use_torch_compile: bool = True,
):
    torch._dynamo.reset()
    with torch.no_grad():
        ref = fn(*args)
        torch.mps.synchronize()
        if use_torch_compile:
            compiled = torch.compile(fn, backend="m1triton", options={"strict": strict})
        else:
            compiled = cb.compile(fn, strict=strict)
        _ = compiled(*args)
        rt.reset_dispatch_count()
        out = compiled(*args)
        dispatches = rt.get_dispatch_count()

    if isinstance(ref, tuple):
        assert isinstance(out, tuple) and len(out) == len(ref)
        for r, o in zip(ref, out):
            torch.testing.assert_close(o.cpu(), r.cpu(), atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(out.cpu(), ref.cpu(), atol=atol, rtol=rtol)

    if expect_dispatches is not None:
        assert dispatches == expect_dispatches, f"Expected {expect_dispatches} dispatches, got {dispatches}"
    else:
        assert dispatches > 0, "Expected at least one M1Triton Metal kernel dispatch"
    return out, dispatches


def test_backend_registered():
    assert "m1triton" in torch._dynamo.list_backends(exclude_tags=())


def test_fused_pointwise_six_ops_single_dispatch_and_vec4():
    torch.manual_seed(1)
    cb.clear_kernel_cache()
    x = torch.randn(16384, device="mps", dtype=torch.float32)

    def six_op_chain(t: torch.Tensor) -> torch.Tensor:
        return F.relu(t * 2.0 + 1.0) * torch.sigmoid(t) - 0.5

    _run_and_verify(six_op_chain, x, expect_dispatches=1)

    assert len(cb._POINTWISE_KERNEL_CACHE) == 1
    kernel = next(iter(cb._POINTWISE_KERNEL_CACHE.values()))
    msl = kernel.inspect_msl(x, x, x.numel(), BLOCK_SIZE=1024)
    assert "float4" in msl


def test_fused_pointwise_activations_math_and_broadcast_single_dispatch():
    torch.manual_seed(2)
    x = torch.randn(64, 128, device="mps", dtype=torch.float32)
    b = torch.randn(1, 128, device="mps", dtype=torch.float32)

    def act_chain(t: torch.Tensor) -> torch.Tensor:
        t = F.leaky_relu(t, 0.1)
        t = F.hardtanh(t, -1.5, 1.5)
        t = F.hardsigmoid(t)
        t = F.hardswish(t)
        t = F.gelu(t)
        t = F.gelu(t, approximate="tanh")
        t = F.silu(t)
        t = F.mish(t)
        return torch.tanh(t)

    _run_and_verify(act_chain, x, expect_dispatches=1)

    def math_bcast_chain(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cond = a > 0.0
        pos = torch.clamp(a ** 2, -0.5, 2.5) - torch.sqrt(torch.abs(a) + 1e-4)
        neg = (
            torch.maximum(a, torch.minimum(v, torch.rsqrt(torch.abs(v) + 1e-4)))
            + torch.exp(-torch.abs(a))
            - torch.log(torch.abs(v) + 1.0)
            + (-v) / 2.0
        )
        return torch.where(cond, pos, neg)

    _run_and_verify(math_bcast_chain, x, b, expect_dispatches=1)


def test_linear_bias_activation_and_matmul_variants():
    torch.manual_seed(3)

    class MLPResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x))) + x

    mlp = MLPResidual().to("mps").eval()
    _run_and_verify(mlp, torch.randn(64, 128, device="mps"), atol=1e-3, rtol=1e-3)

    # 3D linear input
    lin3d = nn.Linear(64, 48).to("mps").eval()
    _run_and_verify(lin3d, torch.randn(4, 16, 64, device="mps"), atol=1e-3, rtol=1e-3)

    # aten.mm + transpose absorption + relu
    a = torch.randn(64, 96, device="mps")
    b = torch.randn(48, 96, device="mps")
    bias = torch.randn(48, device="mps")
    _run_and_verify(lambda x, w: F.relu(torch.mm(x, w.t())), a, b, atol=1e-3, rtol=1e-3)
    _run_and_verify(lambda c, x, w: F.silu(torch.addmm(c, x, w.t())), bias, a, b, atol=1e-3, rtol=1e-3)

    # aten.bmm + activation
    ba = torch.randn(4, 32, 64, device="mps")
    bb = torch.randn(4, 64, 48, device="mps")
    _run_and_verify(lambda x, y: F.relu(torch.bmm(x, y)), ba, bb, atol=1e-3, rtol=1e-3)


def test_conv2d_relu_scale_pool_and_conv3d():
    torch.manual_seed(4)

    class VisionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.scale = nn.Parameter(torch.randn(1, 32, 1, 1))
            self.pool = nn.AvgPool2d(2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(F.relu(self.conv(x)) * self.scale)

    vb = VisionBlock().to("mps").eval()
    _, disp = _run_and_verify(vb, torch.randn(4, 16, 32, 32, device="mps"), atol=1e-3, rtol=1e-3)
    # Cached Winograd filter transform -> Conv2d + fused ReLU*Scale + AvgPool2d = 3 dispatches.
    assert disp == 3

    # Depthwise Conv2d
    dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16).to("mps").eval()
    _run_and_verify(dw, torch.randn(2, 16, 16, 16, device="mps"), atol=1e-3, rtol=1e-3)

    # Conv3d
    c3 = nn.Conv3d(4, 8, kernel_size=3, padding=1).to("mps").eval()
    _run_and_verify(c3, torch.randn(2, 4, 8, 8, 8, device="mps"), atol=1e-3, rtol=1e-3)

    # MaxPool2d (value-only and with indices)
    xp = torch.randn(2, 8, 16, 16, device="mps")
    _run_and_verify(nn.MaxPool2d(2).to("mps"), xp, expect_dispatches=1)
    _run_and_verify(lambda t: F.max_pool2d_with_indices(t, 2), xp, expect_dispatches=1)


def test_softmax_norms_and_reductions():
    torch.manual_seed(5)
    xs = torch.randn(16, 128, device="mps")
    _run_and_verify(lambda t: F.softmax(t, dim=-1), xs, expect_dispatches=1)
    _run_and_verify(lambda t: F.log_softmax(t, dim=-1), xs, expect_dispatches=1)
    _run_and_verify(lambda t: F.softmax(t, dim=1), torch.randn(4, 16, 32, device="mps"), expect_dispatches=1)

    ln = nn.LayerNorm(128).to("mps").eval()
    _run_and_verify(ln, xs)

    gn = nn.GroupNorm(4, 16).to("mps").eval()
    xn = torch.randn(4, 16, 8, 8, device="mps")
    _run_and_verify(gn, xn)

    bn_eval = nn.BatchNorm2d(16).to("mps").eval()
    _run_and_verify(bn_eval, xn, expect_dispatches=1)

    w = torch.ones(16, device="mps")
    b = torch.zeros(16, device="mps")
    _run_and_verify(lambda t, ww, bb: F.batch_norm(t, None, None, ww, bb, training=True), xn, w, b)

    xr = torch.randn(4, 16, 32, device="mps")
    _run_and_verify(lambda t: t.sum(dim=-1), xr, atol=1e-3, rtol=1e-3, expect_dispatches=1)
    _run_and_verify(lambda t: t.mean(dim=1, keepdim=True), xr, atol=1e-3, rtol=1e-3, expect_dispatches=1)
    _run_and_verify(lambda t: t.amax(dim=-1), xr, expect_dispatches=1)
    _run_and_verify(lambda t: t.amin(dim=1), xr, expect_dispatches=1)
    _run_and_verify(lambda t: t.sum(), xr, atol=1e-2, rtol=1e-3)


def test_strict_mode_raises_and_default_mode_partitions_gracefully(monkeypatch):
    torch.manual_seed(6)
    x = torch.randn(16, 64, device="mps")

    def mixed_fn(t: torch.Tensor) -> torch.Tensor:
        y = F.relu(t * 2.0 + 1.0)
        sorted_vals, _ = torch.sort(y, dim=-1)
        return torch.sigmoid(sorted_vals) - 0.5

    # Strict via options={"strict": True}
    torch._dynamo.reset()
    with pytest.raises(Exception) as exc_info:
        c_strict = torch.compile(mixed_fn, backend="m1triton", options={"strict": True})
        c_strict(x)
    assert "NotImplementedError" in repr(exc_info.value) or "does not support" in str(exc_info.value)

    # Strict via M1TRITON_STRICT_COMPILE=1 env var
    monkeypatch.setenv("M1TRITON_STRICT_COMPILE", "1")
    torch._dynamo.reset()
    with pytest.raises(Exception) as env_exc:
        c_env = torch.compile(mixed_fn, backend="m1triton")
        c_env(x)
    assert "NotImplementedError" in repr(env_exc.value) or "does not support" in str(env_exc.value)
    monkeypatch.delenv("M1TRITON_STRICT_COMPILE", raising=False)

    # Default strict=False cleanly partitions: M1Triton (pre) -> Eager (sort) -> M1Triton (post)
    _, disp = _run_and_verify(mixed_fn, x, strict=False, expect_dispatches=2)
    assert disp == 2


def test_kernel_compile_cache_avoids_recompilation():
    torch.manual_seed(7)
    cb.clear_kernel_cache()

    def f1(t: torch.Tensor) -> torch.Tensor:
        return F.relu(t * 3.0 - 1.0) + torch.sigmoid(t)

    def f2(t: torch.Tensor) -> torch.Tensor:
        return F.relu(t * 3.0 - 1.0) + torch.sigmoid(t)

    x1 = torch.randn(2048, device="mps")
    x2 = torch.randn(4096, device="mps")

    with torch.no_grad():
        torch._dynamo.reset()
        c1 = cb.compile(f1, strict=True)
        out1 = c1(x1)
        stats_after_first = cb.get_cache_stats()
        assert stats_after_first["misses"] == 1
        assert stats_after_first["hits"] == 0

        # Repeated calls incur zero cache misses or recompilation.
        for _ in range(3):
            out1_rep = c1(x1)
        assert cb.get_cache_stats()["misses"] == 1
        torch.testing.assert_close(out1_rep.cpu(), out1.cpu(), atol=1e-5, rtol=1e-5)

        # Structurally identical function on a different tensor length hits the synthesized kernel cache.
        torch._dynamo.reset()
        c2 = cb.compile(f2, strict=True)
        out2 = c2(x2)
        stats_after_second = cb.get_cache_stats()
        assert stats_after_second["misses"] == 1
        assert stats_after_second["hits"] >= 1
        torch.testing.assert_close(out2.cpu(), f2(x2).cpu(), atol=1e-4, rtol=1e-4)


def test_autograd_backward_with_torch_compile():
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32)).to("mps")
    x = torch.randn(16, 64, device="mps", requires_grad=True)

    ref_loss = model(x).sum()
    ref_loss.backward()
    ref_grad = x.grad.detach().clone()
    x.grad = None
    model.zero_grad()

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="m1triton")
    rt.reset_dispatch_count()
    loss = compiled(x).sum()
    loss.backward()
    assert rt.get_dispatch_count() > 0
    torch.testing.assert_close(loss.detach().cpu(), ref_loss.detach().cpu(), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(x.grad.detach().cpu(), ref_grad.cpu(), atol=1e-2, rtol=1e-3)


def test_example_03_torch_compile_fusion_runs_cleanly():
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'python'}:{REPO_ROOT / 'build'}"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "03_torch_compile_fusion.py")],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Pointwise 6-op chain" in proc.stdout
    assert "MLP" in proc.stdout
    assert "Vision" in proc.stdout
    assert "vs Eager" in proc.stdout
    assert "vs Inductor" in proc.stdout


def test_hardswish_special_values():
    torch._dynamo.reset()
    x = torch.tensor(
        [-float("inf"), -4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0, float("inf"), float("nan")],
        device="mps",
        dtype=torch.float32,
    )
    compiled = torch.compile(F.hardswish, backend="m1triton", options={"strict": True})
    with torch.no_grad():
        _ = compiled(x)
        rt.reset_dispatch_count()
        out = compiled(x).cpu()
        assert rt.get_dispatch_count() == 1

    assert out[0].item() == 0.0
    assert torch.isposinf(out[8])
    assert torch.isnan(out[9])
    ref_finite = F.hardswish(x[1:8].cpu())
    torch.testing.assert_close(out[1:8], ref_finite, atol=1e-5, rtol=1e-5)


def test_max_pool2d_with_indices_nan_propagation():
    cases = (
        ([[[[1.0, float("nan")], [2.0, 3.0]]]], 1),
        ([[[[float("nan"), float("nan")], [2.0, 3.0]]]], 0),
        ([[[[1.0, 2.0], [float("nan"), 4.0]]]], 2),
    )
    for raw, expected_idx in cases:
        x_mps = torch.tensor(raw, dtype=torch.float32, device="mps")
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda t: F.max_pool2d_with_indices(t, 2), backend="m1triton", options={"strict": True}
        )
        with torch.no_grad():
            _ = compiled(x_mps)
            rt.reset_dispatch_count()
            out_val, out_idx = compiled(x_mps)
            assert rt.get_dispatch_count() == 1
        assert torch.isnan(out_val.cpu()).all()
        assert out_idx.cpu().item() == expected_idx

    # All -inf windows must select each window's first valid spatial index, not default 0.
    x_inf = torch.full((1, 1, 4, 4), -float("inf"), dtype=torch.float32, device="mps")
    torch._dynamo.reset()
    c_inf = torch.compile(
        lambda t: F.max_pool2d_with_indices(t, 2, stride=2),
        backend="m1triton",
        options={"strict": True},
    )
    with torch.no_grad():
        val_inf, idx_inf = c_inf(x_inf)
    assert torch.isneginf(val_inf.cpu()).all()
    assert idx_inf.cpu().flatten().tolist() == [0, 2, 8, 10]


def test_logical_ops_non_bool_inputs():
    for dt in (torch.bool, torch.int32, torch.int64, torch.float32):
        if dt == torch.float32:
            a = torch.tensor([0.0, -0.0, 1.5, -2.0, float("nan"), 0.0], dtype=dt, device="mps")
            b = torch.tensor([0.0, 2.0, 3.0, 0.0, 1.0, float("nan")], dtype=dt, device="mps")
        elif dt == torch.bool:
            a = torch.tensor([False, False, True, True], dtype=dt, device="mps")
            b = torch.tensor([False, True, False, True], dtype=dt, device="mps")
        else:
            a = torch.tensor([0, 1, 2, -3, 0], dtype=dt, device="mps")
            b = torch.tensor([0, 2, 0, 4, 5], dtype=dt, device="mps")

        out_and, _ = _run_and_verify(torch.logical_and, a, b, strict=True, expect_dispatches=1)
        out_or, _ = _run_and_verify(torch.logical_or, a, b, strict=True, expect_dispatches=1)
        out_not, _ = _run_and_verify(torch.logical_not, a, strict=True, expect_dispatches=1)
        assert out_and.dtype == torch.bool
        assert out_or.dtype == torch.bool
        assert out_not.dtype == torch.bool

    # Fused float32 pointwise chain using logical_and/logical_not intermediates (vec4 path).
    af = torch.tensor([0.0, 1.5, -2.0, 0.0, 3.0, -0.0, 4.0, 5.0], dtype=torch.float32, device="mps")
    bf = torch.tensor([1.0, 2.0, 0.0, 0.0, -1.0, 2.0, 0.0, 1.0], dtype=torch.float32, device="mps")
    _run_and_verify(
        lambda x, y: torch.where(torch.logical_and(x, y), x + y, torch.where(torch.logical_not(x), y, -x)),
        af,
        bf,
        strict=True,
        expect_dispatches=1,
    )


def test_duplicate_2d_transpose_in_addmm_and_mm():
    torch.manual_seed(11)
    a = torch.randn(32, 32, device="mps")
    b = torch.randn(32, 32, device="mps")

    def addmm_dup_bias_and_b(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = y.t()
        return torch.addmm(t, x, t)

    def addmm_dup_all(x: torch.Tensor) -> torch.Tensor:
        t = x.t()
        return torch.addmm(t, t, t)

    def mm_dup(x: torch.Tensor) -> torch.Tensor:
        t = x.t()
        return torch.mm(t, t)

    _run_and_verify(addmm_dup_bias_and_b, a, b, atol=1e-3, rtol=1e-3, strict=True)
    _run_and_verify(addmm_dup_all, a, atol=1e-3, rtol=1e-3, strict=True)
    _run_and_verify(mm_dup, a, atol=1e-3, rtol=1e-3, strict=True)


def test_batch_norm_training_updates_running_stats_and_eval_grad():
    torch.manual_seed(12)
    bn_ref = nn.BatchNorm2d(16).to("mps").train()
    bn_cmp = nn.BatchNorm2d(16).to("mps").train()
    bn_cmp.load_state_dict(bn_ref.state_dict())

    torch._dynamo.reset()
    compiled_bn = torch.compile(bn_cmp, backend="m1triton")

    for _ in range(2):
        x = torch.randn(4, 16, 8, 8, device="mps") * 2.0 + 1.5
        ref_y = bn_ref(x)
        cmp_y = compiled_bn(x)
        torch.testing.assert_close(cmp_y.cpu(), ref_y.cpu(), atol=1e-4, rtol=1e-4)

    assert not torch.allclose(bn_cmp.running_mean.cpu(), torch.zeros(16))
    torch.testing.assert_close(bn_cmp.running_mean.cpu(), bn_ref.running_mean.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(bn_cmp.running_var.cpu(), bn_ref.running_var.cpu(), atol=1e-4, rtol=1e-4)

    # Eval BN with grad enabled requires saved stats for backward and must not return None stats.
    bn_eval = nn.BatchNorm2d(16).to("mps").eval()
    x_req = torch.randn(4, 16, 8, 8, device="mps", requires_grad=True)
    ref_out = bn_eval(x_req)
    ref_out.sum().backward()
    ref_grad = x_req.grad.detach().clone()
    x_req.grad = None

    torch._dynamo.reset()
    compiled_eval = torch.compile(bn_eval, backend="m1triton")
    cmp_out = compiled_eval(x_req)
    cmp_out.sum().backward()
    torch.testing.assert_close(cmp_out.detach().cpu(), ref_out.detach().cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x_req.grad.detach().cpu(), ref_grad.cpu(), atol=1e-4, rtol=1e-4)


def test_empty_tensor_reductions():
    x0 = torch.empty((0,), device="mps", dtype=torch.float32)
    x_mid = torch.empty((2, 0, 3), device="mps", dtype=torch.float32)

    out_sum0, _ = _run_and_verify(lambda t: t.sum(), x0, strict=True)
    assert out_sum0.cpu().item() == 0.0

    torch._dynamo.reset()
    c_mean0 = torch.compile(lambda t: t.mean(), backend="m1triton", options={"strict": True})
    with torch.no_grad():
        out_mean0 = c_mean0(x0).cpu()
    assert torch.isnan(out_mean0)

    out_sum_mid, _ = _run_and_verify(lambda t: t.sum(dim=1), x_mid, strict=True)
    torch.testing.assert_close(out_sum_mid.cpu(), torch.zeros(2, 3))

    torch._dynamo.reset()
    c_mean_mid = torch.compile(lambda t: t.mean(dim=1), backend="m1triton", options={"strict": True})
    with torch.no_grad():
        out_mean_mid = c_mean_mid(x_mid).cpu()
    assert out_mean_mid.shape == (2, 3) and torch.isnan(out_mean_mid).all()

    out_sum_multi, _ = _run_and_verify(lambda t: t.sum(dim=(1, 2)), x_mid, strict=True)
    torch.testing.assert_close(out_sum_multi.cpu(), torch.zeros(2))

    torch._dynamo.reset()
    c_amax_empty = torch.compile(lambda t: t.amax(dim=1), backend="m1triton")
    with pytest.raises((IndexError, RuntimeError)):
        c_amax_empty(x_mid)

    torch._dynamo.reset()
    c_amin_empty = torch.compile(lambda t: t.amin(), backend="m1triton")
    with pytest.raises((IndexError, RuntimeError)):
        c_amin_empty(x0)


def test_cross_device_to_copy_and_no_inplace_relu():
    assert torch.ops.aten.relu_.default not in cb._POINTWISE_TARGETS
    torch.manual_seed(13)
    x_mps = torch.randn(1024, device="mps", dtype=torch.float32)

    def mps_to_cpu(t: torch.Tensor) -> torch.Tensor:
        return (t * 2.0 + 1.0).to("cpu")

    out_cpu, disp = _run_and_verify(mps_to_cpu, x_mps, strict=False)
    assert out_cpu.device.type == "cpu"
    assert disp >= 1

    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)

    def cpu_to_mps(t: torch.Tensor) -> torch.Tensor:
        return t.to("mps") * 3.0 - 0.5

    out_mps, disp_mps = _run_and_verify(cpu_to_mps, x_cpu, strict=False)
    assert out_mps.device.type == "mps"
    assert disp_mps >= 1


def test_sgmm_int64_msl_pointwise_int64_and_wino_cache_safety():
    # Verify SGMM kernel with int64 stride arithmetic still lowers via SGMM emitter.
    a = torch.randn(64, 64, device="mps", dtype=torch.float32)
    b = torch.randn(64, 64, device="mps", dtype=torch.float32)
    c = torch.empty(64, 64, device="mps", dtype=torch.float32)
    sgmm_msl = cb._sgmm_gemm_kernel.inspect_msl(
        a, b, c, 64, 64, 64, 0, 64, 1, 0, 64, 1, 0, 64, 1,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, NUM_SG=4, SWIZZLE_W=8,
        TRANS_A=0, TRANS_B=0, N_MAJOR=0, IS_BATCHED=0, ALIGNED_MNK=1,
    )
    assert "simdgroup_multiply_accumulate" in sgmm_msl

    # Verify synthesized pointwise kernel compiles cleanly with USE_INT64=1.
    cb.clear_kernel_cache()
    x = torch.randn(1024, device="mps", dtype=torch.float32)
    _run_and_verify(lambda t: F.relu(t + 1.0), x, strict=True, expect_dispatches=1)
    pw_kernel = next(iter(cb._POINTWISE_KERNEL_CACHE.values()))
    msl_i64 = pw_kernel.inspect_msl(x, x, 3_000_000_000, BLOCK_SIZE=1024, USE_INT64=1)
    assert "long" in msl_i64

    # Verify Winograd cache defers insertion until flush, drops on abort, and uses LRU eviction at 64.
    cb.clear_kernel_cache()
    runner_abort = cb._BatchRunner()
    runner_abort.begin_m1(True)
    w_abort = torch.randn(16, 8, 3, 3, device="mps")
    _ = cb._get_or_transform_wino2x3_filter(w_abort, w_abort, 8, 16, 8, 16, runner_abort)
    assert len(cb._WINO_FILTER_CACHE) == 0
    runner_abort.abort()
    assert len(cb._WINO_FILTER_CACHE) == 0

    weights: list[torch.Tensor] = []
    runner_lru = cb._BatchRunner()
    runner_lru.begin_m1(True)
    for _ in range(65):
        w = torch.randn(16, 8, 3, 3, device="mps")
        weights.append(w)
        _ = cb._get_or_transform_wino2x3_filter(w, w, 8, 16, 8, 16, runner_lru)
    assert len(cb._WINO_FILTER_CACHE) == 0
    runner_lru.flush()
    assert len(cb._WINO_FILTER_CACHE) == 64
    oldest_key = (id(weights[0]), 8, 16, 8, 16, str(weights[0].device))
    newest_key = (id(weights[-1]), 8, 16, 8, 16, str(weights[-1].device))
    assert oldest_key not in cb._WINO_FILTER_CACHE
    assert newest_key in cb._WINO_FILTER_CACHE

    # Verify single-affine GroupNorm (weight-only and bias-only) through unified welford kernel.
    xn = torch.randn(2, 8, 4, 4, device="mps")
    gw = torch.randn(8, device="mps")
    gb = torch.randn(8, device="mps")
    _run_and_verify(lambda t, ww: F.group_norm(t, 4, weight=ww, bias=None), xn, gw, strict=True)
    _run_and_verify(lambda t, bb: F.group_norm(t, 4, weight=None, bias=bb), xn, gb, strict=True)
