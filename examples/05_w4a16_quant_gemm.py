"""Example 05: W4A16 4-bit Group-Quantized GEMM with in-register dequantization on Apple M1 Max."""

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
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_SG": 8}, num_warps=8),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4}, num_warps=4),
    ],
    key=["M", "N", "K", "GROUP_SIZE"],
    warmup=2,
    rep=5,
)
@m1triton.jit
def w4a16_gemm_kernel(
    a_ptr,
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    out_ptr,
    M,
    N,
    K,
    GROUP_SIZE: tl.constexpr = 128,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 4,
):
    tlx.quant_dot(
        a_ptr,
        qweight_ptr,
        scales_ptr,
        zeros_ptr,
        out_ptr,
        M,
        N,
        K,
        bits=4,
        group_size=GROUP_SIZE,
        bm=BLOCK_M,
        bn=BLOCK_N,
        bk=BLOCK_K,
        num_simdgroups=NUM_SG,
        symmetric=False,
    )


def pack_w4_groupwise(
    weight_fp16: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetrically quantize a [K, N] FP16 weight matrix into packed 4-bit int32 + per-group scales/zeros."""
    k, n = weight_fp16.shape
    if k % group_size != 0 or k % 8 != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size} and 8")
    w_cpu = weight_fp16.detach().to("cpu", dtype=torch.float32)
    num_groups = k // group_size
    grouped = w_cpu.view(num_groups, group_size, n)
    w_min = grouped.amin(dim=1)
    w_max = grouped.amax(dim=1)
    scales = ((w_max - w_min) / 15.0).clamp_min(1e-6)
    zeros = (-w_min / scales).round().clamp(0.0, 15.0)
    q = (grouped / scales.unsqueeze(1) + zeros.unsqueeze(1)).round().clamp(0.0, 15.0).to(torch.int64)
    q_words = q.view(k // 8, 8, n)
    packed = torch.zeros((k // 8, n), dtype=torch.int64)
    for j in range(8):
        packed |= q_words[:, j, :] << (4 * j)
    return (
        packed.to(torch.int32).to(weight_fp16.device),
        scales.to(torch.float16).to(weight_fp16.device),
        zeros.to(torch.float16).to(weight_fp16.device),
    )


def torch_w4a16_reference(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    k_packed, n = qweight.shape
    k = k_packed * 8
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
    unpacked = ((qweight.unsqueeze(1) >> shifts) & 0xF).reshape(k, n).to(a.dtype)
    num_groups = k // group_size
    w_fp16 = ((unpacked.view(num_groups, group_size, n) - zeros.unsqueeze(1)) * scales.unsqueeze(1)).reshape(k, n)
    return torch.matmul(a, w_fp16)


def w4a16_matmul(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    group_size: int = 128,
    wait: bool = True,
) -> torch.Tensor:
    m, k = a.shape
    n = qweight.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (m1triton.cdiv(m, meta["BLOCK_M"]) * m1triton.cdiv(n, meta["BLOCK_N"]),)
    w4a16_gemm_kernel[grid](
        a,
        qweight,
        scales,
        zeros,
        out,
        m,
        n,
        k,
        GROUP_SIZE=int(group_size),
        wait=wait,
    )
    return out


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple Silicon MPS backend is required to run this example.")

    m, k, n, group_size = 256, 4096, 4096, 128
    torch.manual_seed(42)
    a = (torch.randn((m, k), dtype=torch.float32) * 0.25).to(torch.float16).to("mps")
    w_dense = (torch.randn((k, n), dtype=torch.float32) * 0.05).to(torch.float16).to("mps")
    qweight, scales, zeros = pack_w4_groupwise(w_dense, group_size=group_size)
    torch.mps.synchronize()

    out = w4a16_matmul(a, qweight, scales, zeros, group_size=group_size, wait=True)
    ref = torch_w4a16_reference(a, qweight, scales, zeros, group_size=group_size)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=5e-3, atol=5e-3)
    max_abs_err = float((out.cpu().float() - ref.cpu().float()).abs().max().item())

    for _ in range(3):
        _ = torch_w4a16_reference(a, qweight, scales, zeros, group_size=group_size)
        _ = w4a16_matmul(a, qweight, scales, zeros, group_size=group_size, wait=True)
    torch.mps.synchronize()

    torch_times_ms: list[float] = []
    m1_times_ms: list[float] = []
    for _ in range(15):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = torch_w4a16_reference(a, qweight, scales, zeros, group_size=group_size)
        torch.mps.synchronize()
        torch_times_ms.append((time.perf_counter() - t0) * 1000.0)

        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = w4a16_matmul(a, qweight, scales, zeros, group_size=group_size, wait=True)
        rt.synchronize()
        m1_times_ms.append((time.perf_counter() - t0) * 1000.0)

    fp32_weight_mb = (w_dense.numel() * 4) / (1024.0 * 1024.0)
    fp16_weight_mb = (w_dense.numel() * w_dense.element_size()) / (1024.0 * 1024.0)
    w4_weight_mb = (qweight.numel() * qweight.element_size()) / (1024.0 * 1024.0)
    w4_meta_mb = (
        scales.numel() * scales.element_size() + zeros.numel() * zeros.element_size()
    ) / (1024.0 * 1024.0)
    m1_med_ms = statistics.median(m1_times_ms)
    torch_med_ms = statistics.median(torch_times_ms)
    speedup = torch_med_ms / m1_med_ms if m1_med_ms > 0 else 0.0

    dev = rt.device_info()
    print(f"Device            : {dev.name}")
    print(f"W4A16 Shape       : ({m}, {k}) @ ({k}, {n}) group_size={group_size}")
    print(
        f"Weight Footprint  : FP32 {fp32_weight_mb:.2f} MB / FP16 {fp16_weight_mb:.2f} MB -> "
        f"W4 {w4_weight_mb:.2f} MB ({fp32_weight_mb / w4_weight_mb:.1f}x vs FP32, "
        f"{fp16_weight_mb / w4_weight_mb:.1f}x vs FP16; +{w4_meta_mb:.2f} MB scales/zeros)"
    )
    print(f"Best Config       : {w4a16_gemm_kernel.best_config}")
    print(f"Correctness       : PASS (max_abs_err={max_abs_err:.2e})")
    print(f"M1Triton W4A16    : {m1_med_ms:.3f} ms")
    print(f"PyTorch MPS Ref   : {torch_med_ms:.3f} ms")
    print(f"Speedup           : {speedup:.2f}x")


if __name__ == "__main__":
    main()
