"""L3-09 Candidate Kernel: Persistent single-dispatch M1 Max MoE Grouped GEMM."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


@m1triton.autotune(
    configs=[
        m1triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "MAX_TGS_PER_CORE": 4},
            num_warps=4,
        ),
        m1triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "MAX_TGS_PER_CORE": 6},
            num_warps=4,
        ),
        m1triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "MAX_TGS_PER_CORE": 3},
            num_warps=4,
        ),
        m1triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "NUM_SG": 4, "MAX_TGS_PER_CORE": 6},
            num_warps=4,
        ),
        m1triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_SG": 4, "MAX_TGS_PER_CORE": 3},
            num_warps=4,
        ),
        m1triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "NUM_SG": 8, "MAX_TGS_PER_CORE": 2},
            num_warps=8,
        ),
    ],
    key=["E", "N", "K"],
    warmup=2,
    rep=5,
)
@m1triton.jit
def _moe_grouped_gemm_kernel(
    a_ptr,
    w_ptr,
    offsets_ptr,
    out_ptr,
    E,
    N,
    K,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 16,
    NUM_SG: tl.constexpr = 4,
    MAX_TGS_PER_CORE: tl.constexpr = 4,
):
    tlx.grouped_gemm(
        a_ptr,
        w_ptr,
        offsets_ptr,
        out_ptr,
        E,
        N,
        K,
        bm=BLOCK_M,
        bn=BLOCK_N,
        bk=BLOCK_K,
        num_simdgroups=NUM_SG,
        transpose_b=True,
        persistent=True,
        max_tgs_per_core=MAX_TGS_PER_CORE,
    )


def candidate_forward(a: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    a_c = a if a.is_contiguous() else a.contiguous()
    w_c = w if w.is_contiguous() else w.contiguous()
    off_c = offsets if offsets.is_contiguous() else offsets.contiguous()
    total_tokens, k = a_c.shape
    num_experts, n, _ = w_c.shape
    out = torch.empty((total_tokens, n), device=a_c.device, dtype=a_c.dtype)
    grid = lambda meta: (32 * meta["MAX_TGS_PER_CORE"],)
    _moe_grouped_gemm_kernel[grid](
        a_c,
        w_c,
        off_c,
        out,
        num_experts,
        n,
        k,
        wait=True,
    )
    return out
