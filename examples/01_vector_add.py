"""Example 01: Vectorized Apple Silicon Metal vector addition with M1Triton."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

try:
    import m1triton
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    import m1triton

import torch

import m1triton.language as tl
import m1triton.runtime as rt


@m1triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, *, block_size: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (m1triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple Silicon MPS backend is required to run this example.")

    n_elements = 16 * 1024 * 1024
    block_size = 1024
    torch.manual_seed(42)
    x = torch.randn(n_elements, device="mps", dtype=torch.float32)
    y = torch.randn(n_elements, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    grid = (m1triton.cdiv(n_elements, block_size),)
    for _ in range(3):
        vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size, wait=True, sync_mps=False)

    expected = x + y
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
    max_abs_err = float((out.cpu() - expected.cpu()).abs().max().item())

    m1_times_ms: list[float] = []
    for _ in range(10):
        res = vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size, wait=True, sync_mps=False)
        m1_times_ms.append(res.last_gpu_ms)

    torch_times_ms: list[float] = []
    for _ in range(3):
        _ = x + y
    torch.mps.synchronize()
    for _ in range(10):
        t0 = time.perf_counter()
        _ = x + y
        torch.mps.synchronize()
        torch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    total_bytes = 3.0 * float(n_elements) * 4.0
    m1_med_ms = statistics.median(m1_times_ms)
    m1_gbs = (total_bytes / (m1_med_ms * 1e-3)) / 1e9
    torch_med_ms = statistics.median(torch_times_ms)
    torch_gbs = (total_bytes / (torch_med_ms * 1e-3)) / 1e9

    dev = rt.device_info()
    print(f"Device       : {dev.name}")
    print(f"Elements     : {n_elements:,} float32 ({total_bytes / 1e6:.1f} MB traffic)")
    print(f"Correctness  : PASS (max_abs_err={max_abs_err:.2e})")
    print(f"M1Triton     : {m1_med_ms:.3f} ms ({m1_gbs:.1f} GB/s)")
    print(f"PyTorch MPS  : {torch_med_ms:.3f} ms ({torch_gbs:.1f} GB/s)")


if __name__ == "__main__":
    main()
