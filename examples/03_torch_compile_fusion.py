"""Example 03: End-to-end `torch.compile(model, backend="m1triton")` FX graph fusion on Apple Silicon."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    import m1triton
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import m1triton

import torch
import torch.nn as nn
import torch.nn.functional as F

import m1triton.compile_backend as m1_compile
import m1triton.runtime as rt


def pointwise_chain(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x * 2.0 + 1.0) * torch.sigmoid(x) - 0.5


class MLPBlock(nn.Module):
    """Transformer-style MLP block (`Linear -> GELU -> Linear + Residual`)."""

    def __init__(self, d_model: int = 1024, d_ff: int = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x))) + x


class VisionBlock(nn.Module):
    """Vision conv block (`Conv2d -> ReLU -> Scale -> AvgPool2d`)."""

    def __init__(self, in_ch: int = 64, out_ch: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.full((1, out_ch, 1, 1), 0.75))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv(x)) * self.scale)


def _measure_ms(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int = 5,
    rep: int = 20,
    sync_mps: bool = True,
) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            out = fn(x)
        if sync_mps:
            torch.mps.synchronize()
        times_ms: list[float] = []
        for _ in range(rep):
            t0 = time.perf_counter()
            out = fn(x)
            if sync_mps:
                torch.mps.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times_ms), out


def _run_benchmark(
    name: str,
    model: Any,
    x: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    torch._dynamo.reset()
    compiled_m1 = torch.compile(model, backend="m1triton")
    compiled_ind = torch.compile(model, backend="inductor")

    # Warm up and verify strict numerical accuracy plus M1Triton kernel dispatch count.
    with torch.no_grad():
        ref = model(x)
        torch.mps.synchronize()
        _ = compiled_m1(x)
        rt.reset_dispatch_count()
        out_m1 = compiled_m1(x)
        dispatches = rt.get_dispatch_count()
        assert dispatches > 0, f"{name}: expected M1Triton dispatch_count > 0, got {dispatches}"
        torch.testing.assert_close(out_m1.cpu(), ref.cpu(), atol=atol, rtol=rtol)
        max_abs_err = float((out_m1.cpu() - ref.cpu()).abs().max().item())

    eager_ms, _ = _measure_ms(model, x, sync_mps=True)
    inductor_ms, out_ind = _measure_ms(compiled_ind, x, sync_mps=True)
    torch.testing.assert_close(out_ind.cpu(), ref.cpu(), atol=atol, rtol=rtol)
    m1_ms, _ = _measure_ms(compiled_m1, x, sync_mps=False)

    spd_eager = eager_ms / m1_ms
    spd_inductor = inductor_ms / m1_ms
    print(
        f"{name:32s} | disp={dispatches:2d} | err={max_abs_err:.2e} | "
        f"Eager={eager_ms:.3f}ms | Inductor={inductor_ms:.3f}ms | "
        f"M1Triton={m1_ms:.3f}ms ({spd_eager:.2f}x vs Eager, {spd_inductor:.2f}x vs Inductor)"
    )
    return {
        "name": name,
        "dispatches": dispatches,
        "max_abs_err": max_abs_err,
        "eager_ms": eager_ms,
        "inductor_ms": inductor_ms,
        "m1triton_ms": m1_ms,
        "speedup_vs_eager": spd_eager,
        "speedup_vs_inductor": spd_inductor,
    }


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple Silicon MPS backend is required to run this example.")

    torch.manual_seed(42)
    dev = rt.device_info()
    print(f"Device       : {dev.name}")
    print(f"PyTorch      : {torch.__version__} (backends include 'm1triton': {'m1triton' in torch._dynamo.list_backends(exclude_tags=())})")
    print("-" * 124)

    x_pw = torch.randn(16 * 1024 * 1024, device="mps", dtype=torch.float32)
    _run_benchmark("1. Pointwise 6-op chain (16M)", pointwise_chain, x_pw, atol=1e-4, rtol=1e-4)

    mlp = MLPBlock(d_model=1024, d_ff=4096).to("mps").eval()
    x_mlp = torch.randn(512, 1024, device="mps", dtype=torch.float32)
    _run_benchmark("2. MLP (512x1024->4096+Res)", mlp, x_mlp, atol=1e-3, rtol=1e-3)

    vision = VisionBlock(in_ch=64, out_ch=128).to("mps").eval()
    x_vis = torch.randn(8, 64, 64, 64, device="mps", dtype=torch.float32)
    _run_benchmark("3. Vision (Conv+ReLU+Scale+Pool)", vision, x_vis, atol=1e-3, rtol=1e-3)

    stats = m1_compile.get_cache_stats()
    print("-" * 124)
    print(f"Kernel Cache : {stats}")


if __name__ == "__main__":
    main()
