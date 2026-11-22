"""Example 02: Autotuned SIMD-group matrix multiplication with `@m1triton.autotune` and `@m1triton.heuristics`."""

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
import m1triton.tlx as tlx


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 8}, num_warps=8),
    ],
    key=["M", "N", "K"],
    warmup=2,
    rep=5,
)
@m1triton.heuristics(
    {
        "ALIGNED_MNK": lambda args: int(
            (args["M"] % args["BLOCK_M"] == 0)
            and (args["N"] % args["BLOCK_N"] == 0)
            and (args["K"] % args["BLOCK_K"] == 0)
        ),
    }
)
@m1triton.jit
def autotuned_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr = 4,
    ALIGNED_MNK: tl.constexpr = 0,
):
    tlx.tile_spec(BLOCK_M, BLOCK_N, BLOCK_K, num_simdgroups=NUM_SG, double_buffer=False, split_k=1, swizzle_w=8)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        if ALIGNED_MNK == 1:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
            b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if ALIGNED_MNK == 1:
        tl.store(c_ptrs, acc)
    else:
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, *, wait: bool = False, sync_mps: bool = True):
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = lambda meta: (m1triton.cdiv(m, meta["BLOCK_M"]), m1triton.cdiv(n, meta["BLOCK_N"]))
    compiled = autotuned_matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        wait=wait,
        sync_mps=sync_mps,
    )
    return c, compiled


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple Silicon MPS backend is required to run this example.")

    m = n = k = 1024
    torch.manual_seed(7)
    a = torch.randn((m, k), device="mps", dtype=torch.float32)
    b = torch.randn((k, n), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    c, _ = matmul(a, b, wait=True, sync_mps=True)
    ref = torch.matmul(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
    max_abs_err = float((c.cpu() - ref.cpu()).abs().max().item())

    grid = lambda meta: (m1triton.cdiv(m, meta["BLOCK_M"]), m1triton.cdiv(n, meta["BLOCK_N"]))
    m1_times_ms: list[float] = []
    for _ in range(10):
        res = autotuned_matmul_kernel[grid](
            a, b, c,
            m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            wait=True,
            sync_mps=False,
        )
        m1_times_ms.append(res.last_gpu_ms)

    torch_times_ms: list[float] = []
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.mps.synchronize()
    for _ in range(10):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        torch.mps.synchronize()
        torch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    flops = 2.0 * float(m) * float(n) * float(k)
    m1_med_ms = statistics.median(m1_times_ms)
    m1_tflops = (flops / (m1_med_ms * 1e-3)) / 1e12
    torch_med_ms = statistics.median(torch_times_ms)
    torch_tflops = (flops / (torch_med_ms * 1e-3)) / 1e12

    dev = rt.device_info()
    print(f"Device       : {dev.name}")
    print(f"MatMul Shape : ({m}, {k}) @ ({k}, {n}) float32")
    print(f"Best Config  : {autotuned_matmul_kernel.best_config}")
    print(f"Correctness  : PASS (max_abs_err={max_abs_err:.2e})")
    print(f"M1Triton     : {m1_med_ms:.3f} ms ({m1_tflops:.2f} TFLOPS)")
    print(f"PyTorch MPS  : {torch_med_ms:.3f} ms ({torch_tflops:.2f} TFLOPS)")


if __name__ == "__main__":
    main()
