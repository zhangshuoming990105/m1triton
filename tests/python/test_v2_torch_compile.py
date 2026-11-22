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
