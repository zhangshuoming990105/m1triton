"""L3-10 Candidate Kernel: Single-dispatch W4A16 register-dequant SGMM on Apple M1 Max."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "NUM_SG": 8}, num_warps=8),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4}, num_warps=4),
        m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 2}, num_warps=2),
        m1triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "NUM_SG": 8}, num_warps=8),
    ],
    key=["M", "N", "K", "GROUP_SIZE"],
    warmup=2,
    rep=5,
)
@m1triton.jit
def _w4a16_dequant_gemm_kernel(
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
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 8,
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


def candidate_forward(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    a_c = a if a.is_contiguous() else a.contiguous()
    qw_c = qweight if qweight.is_contiguous() else qweight.contiguous()
    sc_c = scales if scales.is_contiguous() else scales.contiguous()
    zp_c = zeros if zeros.is_contiguous() else zeros.contiguous()
    m, k = a_c.shape
    n = qw_c.shape[1]
    out = torch.empty((m, n), device=a_c.device, dtype=a_c.dtype)
    grid = lambda meta: (m1triton.cdiv(m, meta["BLOCK_M"]) * m1triton.cdiv(n, meta["BLOCK_N"]),)
    _w4a16_dequant_gemm_kernel[grid](
        a_c,
        qw_c,
        sc_c,
        zp_c,
        out,
        m,
        n,
        k,
        GROUP_SIZE=int(group_size),
        wait=True,
    )
    return out
