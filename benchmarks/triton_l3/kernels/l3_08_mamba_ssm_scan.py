"""Level 3 Kernel L3-08: Mamba / Selective State-Space Scan (SSM Forward).

Single-dispatch register-resident M1Triton kernel for Apple M1 Max:
  - Grid `(cdiv(D, BLOCK_D), B)`: each thread owns one channel `d` with coalesced
    `D`-major loads and stores across every 32-thread SIMD group.
  - Preloads `A[d, 0..15]`, `dt_bias[d]`, and `D_skip[d]` once before the `L` loop.
  - Keeps all 16 recurrent state slots `h0..h15` purely in FP32 thread registers
    (zero `[B, L, D, N]` DRAM allocation, zero SRAM spills).
  - Uniform scalar loads of `B_proj[b, t, n]` and `C_proj[b, t, n]` broadcast across
    each threadgroup from L1 cache.
  - Autotunes `(BLOCK_D, UNROLL_T)` with persistent disk caching and derives
    unmasked `EVEN_D` / tail-free `EVEN_L` specializations via `@m1triton.heuristics`.
"""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_D": 32, "UNROLL_T": 1}, num_warps=1),
        m1triton.Config({"BLOCK_D": 32, "UNROLL_T": 2}, num_warps=1),
        m1triton.Config({"BLOCK_D": 32, "UNROLL_T": 4}, num_warps=1),
        m1triton.Config({"BLOCK_D": 64, "UNROLL_T": 2}, num_warps=2),
        m1triton.Config({"BLOCK_D": 64, "UNROLL_T": 4}, num_warps=2),
        m1triton.Config({"BLOCK_D": 128, "UNROLL_T": 2}, num_warps=4),
        m1triton.Config({"BLOCK_D": 256, "UNROLL_T": 2}, num_warps=8),
    ],
    key=["B", "L", "D"],
)
@m1triton.heuristics(
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "EVEN_L": lambda args: (args["L"] % args["UNROLL_T"]) == 0,
    }
)
@m1triton.jit(fast_math=True)
def _mamba_ssm_scan_n16_kernel(
    u_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_skip_ptr,
    dt_bias_ptr,
    y_ptr,
    B,
    L,
    D,
    stride_u_b,
    stride_u_l,
    stride_u_d,
    stride_bc_b,
    BLOCK_D: tl.constexpr,
    UNROLL_T: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_L: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    d_idx = d_offs * stride_u_d

    b_u_off = pid_b * stride_u_b
    u_base = u_ptr + b_u_off + d_idx
    dt_base = dt_ptr + b_u_off + d_idx
    y_base = y_ptr + b_u_off + d_idx
    bc_b_off = pid_b * stride_bc_b

    a_base = a_ptr + d_offs * 16
    if EVEN_D:
        d_skip = tl.load(d_skip_ptr + d_idx)
        dt_b = tl.load(dt_bias_ptr + d_idx)
        a0 = tl.load(a_base + 0)
        a1 = tl.load(a_base + 1)
        a2 = tl.load(a_base + 2)
        a3 = tl.load(a_base + 3)
        a4 = tl.load(a_base + 4)
        a5 = tl.load(a_base + 5)
        a6 = tl.load(a_base + 6)
        a7 = tl.load(a_base + 7)
        a8 = tl.load(a_base + 8)
        a9 = tl.load(a_base + 9)
        a10 = tl.load(a_base + 10)
        a11 = tl.load(a_base + 11)
        a12 = tl.load(a_base + 12)
        a13 = tl.load(a_base + 13)
        a14 = tl.load(a_base + 14)
        a15 = tl.load(a_base + 15)
    else:
        d_skip = tl.load(d_skip_ptr + d_idx, mask=d_mask, other=0.0)
        dt_b = tl.load(dt_bias_ptr + d_idx, mask=d_mask, other=0.0)
        a0 = tl.load(a_base + 0, mask=d_mask, other=0.0)
        a1 = tl.load(a_base + 1, mask=d_mask, other=0.0)
        a2 = tl.load(a_base + 2, mask=d_mask, other=0.0)
        a3 = tl.load(a_base + 3, mask=d_mask, other=0.0)
        a4 = tl.load(a_base + 4, mask=d_mask, other=0.0)
        a5 = tl.load(a_base + 5, mask=d_mask, other=0.0)
        a6 = tl.load(a_base + 6, mask=d_mask, other=0.0)
        a7 = tl.load(a_base + 7, mask=d_mask, other=0.0)
        a8 = tl.load(a_base + 8, mask=d_mask, other=0.0)
        a9 = tl.load(a_base + 9, mask=d_mask, other=0.0)
        a10 = tl.load(a_base + 10, mask=d_mask, other=0.0)
        a11 = tl.load(a_base + 11, mask=d_mask, other=0.0)
        a12 = tl.load(a_base + 12, mask=d_mask, other=0.0)
        a13 = tl.load(a_base + 13, mask=d_mask, other=0.0)
        a14 = tl.load(a_base + 14, mask=d_mask, other=0.0)
        a15 = tl.load(a_base + 15, mask=d_mask, other=0.0)

    h0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h8 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h9 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h11 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h14 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h15 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    l_main = L if EVEN_L else (L // UNROLL_T) * UNROLL_T

    for t0 in range(0, l_main, UNROLL_T):
        for s in tl.static_range(0, UNROLL_T):
            t = t0 + s
            t_u_off = t * stride_u_l
            t_bc_off = bc_b_off + t * 16

            if EVEN_D:
                u_t = tl.load(u_base + t_u_off)
                dt_t = tl.load(dt_base + t_u_off)
            else:
                u_t = tl.load(u_base + t_u_off, mask=d_mask, other=0.0)
                dt_t = tl.load(dt_base + t_u_off, mask=d_mask, other=0.0)

            x = dt_t + dt_b
            x_safe = tl.where(x > 20.0, 0.0, x)
            delta = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
            du = delta * u_t

            h0 = tl.exp(delta * a0) * h0 + du * tl.load(b_ptr + t_bc_off + 0)
            y_t = d_skip * u_t + h0 * tl.load(c_ptr + t_bc_off + 0)

            h1 = tl.exp(delta * a1) * h1 + du * tl.load(b_ptr + t_bc_off + 1)
            y_t = y_t + h1 * tl.load(c_ptr + t_bc_off + 1)

            h2 = tl.exp(delta * a2) * h2 + du * tl.load(b_ptr + t_bc_off + 2)
            y_t = y_t + h2 * tl.load(c_ptr + t_bc_off + 2)

            h3 = tl.exp(delta * a3) * h3 + du * tl.load(b_ptr + t_bc_off + 3)
            y_t = y_t + h3 * tl.load(c_ptr + t_bc_off + 3)

            h4 = tl.exp(delta * a4) * h4 + du * tl.load(b_ptr + t_bc_off + 4)
            y_t = y_t + h4 * tl.load(c_ptr + t_bc_off + 4)

            h5 = tl.exp(delta * a5) * h5 + du * tl.load(b_ptr + t_bc_off + 5)
            y_t = y_t + h5 * tl.load(c_ptr + t_bc_off + 5)

            h6 = tl.exp(delta * a6) * h6 + du * tl.load(b_ptr + t_bc_off + 6)
            y_t = y_t + h6 * tl.load(c_ptr + t_bc_off + 6)

            h7 = tl.exp(delta * a7) * h7 + du * tl.load(b_ptr + t_bc_off + 7)
            y_t = y_t + h7 * tl.load(c_ptr + t_bc_off + 7)

            h8 = tl.exp(delta * a8) * h8 + du * tl.load(b_ptr + t_bc_off + 8)
            y_t = y_t + h8 * tl.load(c_ptr + t_bc_off + 8)

            h9 = tl.exp(delta * a9) * h9 + du * tl.load(b_ptr + t_bc_off + 9)
            y_t = y_t + h9 * tl.load(c_ptr + t_bc_off + 9)

            h10 = tl.exp(delta * a10) * h10 + du * tl.load(b_ptr + t_bc_off + 10)
            y_t = y_t + h10 * tl.load(c_ptr + t_bc_off + 10)

            h11 = tl.exp(delta * a11) * h11 + du * tl.load(b_ptr + t_bc_off + 11)
            y_t = y_t + h11 * tl.load(c_ptr + t_bc_off + 11)

            h12 = tl.exp(delta * a12) * h12 + du * tl.load(b_ptr + t_bc_off + 12)
            y_t = y_t + h12 * tl.load(c_ptr + t_bc_off + 12)

            h13 = tl.exp(delta * a13) * h13 + du * tl.load(b_ptr + t_bc_off + 13)
            y_t = y_t + h13 * tl.load(c_ptr + t_bc_off + 13)

            h14 = tl.exp(delta * a14) * h14 + du * tl.load(b_ptr + t_bc_off + 14)
            y_t = y_t + h14 * tl.load(c_ptr + t_bc_off + 14)

            h15 = tl.exp(delta * a15) * h15 + du * tl.load(b_ptr + t_bc_off + 15)
            y_t = y_t + h15 * tl.load(c_ptr + t_bc_off + 15)

            if EVEN_D:
                tl.store(y_base + t_u_off, y_t)
            else:
                tl.store(y_base + t_u_off, y_t, mask=d_mask)

    if not EVEN_L:
        for t in range(l_main, L):
            t_u_off = t * stride_u_l
            t_bc_off = bc_b_off + t * 16

            if EVEN_D:
                u_t = tl.load(u_base + t_u_off)
                dt_t = tl.load(dt_base + t_u_off)
            else:
                u_t = tl.load(u_base + t_u_off, mask=d_mask, other=0.0)
                dt_t = tl.load(dt_base + t_u_off, mask=d_mask, other=0.0)

            x = dt_t + dt_b
            x_safe = tl.where(x > 20.0, 0.0, x)
            delta = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
            du = delta * u_t

            h0 = tl.exp(delta * a0) * h0 + du * tl.load(b_ptr + t_bc_off + 0)
            y_t = d_skip * u_t + h0 * tl.load(c_ptr + t_bc_off + 0)

            h1 = tl.exp(delta * a1) * h1 + du * tl.load(b_ptr + t_bc_off + 1)
            y_t = y_t + h1 * tl.load(c_ptr + t_bc_off + 1)

            h2 = tl.exp(delta * a2) * h2 + du * tl.load(b_ptr + t_bc_off + 2)
            y_t = y_t + h2 * tl.load(c_ptr + t_bc_off + 2)

            h3 = tl.exp(delta * a3) * h3 + du * tl.load(b_ptr + t_bc_off + 3)
            y_t = y_t + h3 * tl.load(c_ptr + t_bc_off + 3)

            h4 = tl.exp(delta * a4) * h4 + du * tl.load(b_ptr + t_bc_off + 4)
            y_t = y_t + h4 * tl.load(c_ptr + t_bc_off + 4)

            h5 = tl.exp(delta * a5) * h5 + du * tl.load(b_ptr + t_bc_off + 5)
            y_t = y_t + h5 * tl.load(c_ptr + t_bc_off + 5)

            h6 = tl.exp(delta * a6) * h6 + du * tl.load(b_ptr + t_bc_off + 6)
            y_t = y_t + h6 * tl.load(c_ptr + t_bc_off + 6)

            h7 = tl.exp(delta * a7) * h7 + du * tl.load(b_ptr + t_bc_off + 7)
            y_t = y_t + h7 * tl.load(c_ptr + t_bc_off + 7)

            h8 = tl.exp(delta * a8) * h8 + du * tl.load(b_ptr + t_bc_off + 8)
            y_t = y_t + h8 * tl.load(c_ptr + t_bc_off + 8)

            h9 = tl.exp(delta * a9) * h9 + du * tl.load(b_ptr + t_bc_off + 9)
            y_t = y_t + h9 * tl.load(c_ptr + t_bc_off + 9)

            h10 = tl.exp(delta * a10) * h10 + du * tl.load(b_ptr + t_bc_off + 10)
            y_t = y_t + h10 * tl.load(c_ptr + t_bc_off + 10)

            h11 = tl.exp(delta * a11) * h11 + du * tl.load(b_ptr + t_bc_off + 11)
            y_t = y_t + h11 * tl.load(c_ptr + t_bc_off + 11)

            h12 = tl.exp(delta * a12) * h12 + du * tl.load(b_ptr + t_bc_off + 12)
            y_t = y_t + h12 * tl.load(c_ptr + t_bc_off + 12)

            h13 = tl.exp(delta * a13) * h13 + du * tl.load(b_ptr + t_bc_off + 13)
            y_t = y_t + h13 * tl.load(c_ptr + t_bc_off + 13)

            h14 = tl.exp(delta * a14) * h14 + du * tl.load(b_ptr + t_bc_off + 14)
            y_t = y_t + h14 * tl.load(c_ptr + t_bc_off + 14)

            h15 = tl.exp(delta * a15) * h15 + du * tl.load(b_ptr + t_bc_off + 15)
            y_t = y_t + h15 * tl.load(c_ptr + t_bc_off + 15)

            if EVEN_D:
                tl.store(y_base + t_u_off, y_t)
            else:
                tl.store(y_base + t_u_off, y_t, mask=d_mask)


@m1triton.jit(fast_math=True)
def _mamba_ssm_scan_chunk16_kernel(
    u_ptr,
    dt_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_skip_ptr,
    dt_bias_ptr,
    y_ptr,
    B,
    L,
    D,
    N_STATE,
    n_base,
    stride_u_b,
    stride_u_l,
    stride_u_d,
    stride_bc_b,
    stride_bc_l,
    IS_FIRST_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    d_idx = d_offs * stride_u_d

    b_u_off = pid_b * stride_u_b
    u_base = u_ptr + b_u_off + d_idx
    dt_base = dt_ptr + b_u_off + d_idx
    y_base = y_ptr + b_u_off + d_idx
    bc_b_off = pid_b * stride_bc_b

    d_skip = tl.load(d_skip_ptr + d_idx, mask=d_mask, other=0.0)
    dt_b = tl.load(dt_bias_ptr + d_idx, mask=d_mask, other=0.0)
    a_base = a_ptr + d_offs * N_STATE + n_base

    n0_ok = (n_base + 0) < N_STATE
    n1_ok = (n_base + 1) < N_STATE
    n2_ok = (n_base + 2) < N_STATE
    n3_ok = (n_base + 3) < N_STATE
    n4_ok = (n_base + 4) < N_STATE
    n5_ok = (n_base + 5) < N_STATE
    n6_ok = (n_base + 6) < N_STATE
    n7_ok = (n_base + 7) < N_STATE
    n8_ok = (n_base + 8) < N_STATE
    n9_ok = (n_base + 9) < N_STATE
    n10_ok = (n_base + 10) < N_STATE
    n11_ok = (n_base + 11) < N_STATE
    n12_ok = (n_base + 12) < N_STATE
    n13_ok = (n_base + 13) < N_STATE
    n14_ok = (n_base + 14) < N_STATE
    n15_ok = (n_base + 15) < N_STATE

    a0 = tl.load(a_base + 0, mask=d_mask & n0_ok, other=0.0)
    a1 = tl.load(a_base + 1, mask=d_mask & n1_ok, other=0.0)
    a2 = tl.load(a_base + 2, mask=d_mask & n2_ok, other=0.0)
    a3 = tl.load(a_base + 3, mask=d_mask & n3_ok, other=0.0)
    a4 = tl.load(a_base + 4, mask=d_mask & n4_ok, other=0.0)
    a5 = tl.load(a_base + 5, mask=d_mask & n5_ok, other=0.0)
    a6 = tl.load(a_base + 6, mask=d_mask & n6_ok, other=0.0)
    a7 = tl.load(a_base + 7, mask=d_mask & n7_ok, other=0.0)
    a8 = tl.load(a_base + 8, mask=d_mask & n8_ok, other=0.0)
    a9 = tl.load(a_base + 9, mask=d_mask & n9_ok, other=0.0)
    a10 = tl.load(a_base + 10, mask=d_mask & n10_ok, other=0.0)
    a11 = tl.load(a_base + 11, mask=d_mask & n11_ok, other=0.0)
    a12 = tl.load(a_base + 12, mask=d_mask & n12_ok, other=0.0)
    a13 = tl.load(a_base + 13, mask=d_mask & n13_ok, other=0.0)
    a14 = tl.load(a_base + 14, mask=d_mask & n14_ok, other=0.0)
    a15 = tl.load(a_base + 15, mask=d_mask & n15_ok, other=0.0)

    h0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h8 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h9 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h11 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h14 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h15 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(0, L):
        t_u_off = t * stride_u_l
        t_bc_off = bc_b_off + t * stride_bc_l + n_base

        u_t = tl.load(u_base + t_u_off, mask=d_mask, other=0.0)
        dt_t = tl.load(dt_base + t_u_off, mask=d_mask, other=0.0)

        x = dt_t + dt_b
        x_safe = tl.where(x > 20.0, 0.0, x)
        delta = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
        du = delta * u_t

        if IS_FIRST_CHUNK:
            y_t = d_skip * u_t
        else:
            y_t = tl.load(y_base + t_u_off, mask=d_mask, other=0.0)

        h0 = tl.exp(delta * a0) * h0 + du * tl.load(b_ptr + t_bc_off + 0, mask=n0_ok, other=0.0)
        y_t = y_t + h0 * tl.load(c_ptr + t_bc_off + 0, mask=n0_ok, other=0.0)

        h1 = tl.exp(delta * a1) * h1 + du * tl.load(b_ptr + t_bc_off + 1, mask=n1_ok, other=0.0)
        y_t = y_t + h1 * tl.load(c_ptr + t_bc_off + 1, mask=n1_ok, other=0.0)

        h2 = tl.exp(delta * a2) * h2 + du * tl.load(b_ptr + t_bc_off + 2, mask=n2_ok, other=0.0)
        y_t = y_t + h2 * tl.load(c_ptr + t_bc_off + 2, mask=n2_ok, other=0.0)

        h3 = tl.exp(delta * a3) * h3 + du * tl.load(b_ptr + t_bc_off + 3, mask=n3_ok, other=0.0)
        y_t = y_t + h3 * tl.load(c_ptr + t_bc_off + 3, mask=n3_ok, other=0.0)

        h4 = tl.exp(delta * a4) * h4 + du * tl.load(b_ptr + t_bc_off + 4, mask=n4_ok, other=0.0)
        y_t = y_t + h4 * tl.load(c_ptr + t_bc_off + 4, mask=n4_ok, other=0.0)

        h5 = tl.exp(delta * a5) * h5 + du * tl.load(b_ptr + t_bc_off + 5, mask=n5_ok, other=0.0)
        y_t = y_t + h5 * tl.load(c_ptr + t_bc_off + 5, mask=n5_ok, other=0.0)

        h6 = tl.exp(delta * a6) * h6 + du * tl.load(b_ptr + t_bc_off + 6, mask=n6_ok, other=0.0)
        y_t = y_t + h6 * tl.load(c_ptr + t_bc_off + 6, mask=n6_ok, other=0.0)

        h7 = tl.exp(delta * a7) * h7 + du * tl.load(b_ptr + t_bc_off + 7, mask=n7_ok, other=0.0)
        y_t = y_t + h7 * tl.load(c_ptr + t_bc_off + 7, mask=n7_ok, other=0.0)

        h8 = tl.exp(delta * a8) * h8 + du * tl.load(b_ptr + t_bc_off + 8, mask=n8_ok, other=0.0)
        y_t = y_t + h8 * tl.load(c_ptr + t_bc_off + 8, mask=n8_ok, other=0.0)

        h9 = tl.exp(delta * a9) * h9 + du * tl.load(b_ptr + t_bc_off + 9, mask=n9_ok, other=0.0)
        y_t = y_t + h9 * tl.load(c_ptr + t_bc_off + 9, mask=n9_ok, other=0.0)

        h10 = tl.exp(delta * a10) * h10 + du * tl.load(b_ptr + t_bc_off + 10, mask=n10_ok, other=0.0)
        y_t = y_t + h10 * tl.load(c_ptr + t_bc_off + 10, mask=n10_ok, other=0.0)

        h11 = tl.exp(delta * a11) * h11 + du * tl.load(b_ptr + t_bc_off + 11, mask=n11_ok, other=0.0)
        y_t = y_t + h11 * tl.load(c_ptr + t_bc_off + 11, mask=n11_ok, other=0.0)

        h12 = tl.exp(delta * a12) * h12 + du * tl.load(b_ptr + t_bc_off + 12, mask=n12_ok, other=0.0)
        y_t = y_t + h12 * tl.load(c_ptr + t_bc_off + 12, mask=n12_ok, other=0.0)

        h13 = tl.exp(delta * a13) * h13 + du * tl.load(b_ptr + t_bc_off + 13, mask=n13_ok, other=0.0)
        y_t = y_t + h13 * tl.load(c_ptr + t_bc_off + 13, mask=n13_ok, other=0.0)

        h14 = tl.exp(delta * a14) * h14 + du * tl.load(b_ptr + t_bc_off + 14, mask=n14_ok, other=0.0)
        y_t = y_t + h14 * tl.load(c_ptr + t_bc_off + 14, mask=n14_ok, other=0.0)

        h15 = tl.exp(delta * a15) * h15 + du * tl.load(b_ptr + t_bc_off + 15, mask=n15_ok, other=0.0)
        y_t = y_t + h15 * tl.load(c_ptr + t_bc_off + 15, mask=n15_ok, other=0.0)

        tl.store(y_base + t_u_off, y_t, mask=d_mask)


def candidate_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    D_skip: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Fused register-resident Mamba selective state-space scan (M1Triton Metal)."""
    u_c = u if u.is_contiguous() else u.contiguous()
    dt_c = dt if dt.is_contiguous() else dt.contiguous()
    a_c = A if A.is_contiguous() else A.contiguous()
    b_c = B_proj if B_proj.is_contiguous() else B_proj.contiguous()
    c_c = C_proj if C_proj.is_contiguous() else C_proj.contiguous()
    d_skip_c = D_skip if D_skip.is_contiguous() else D_skip.contiguous()
    dt_bias_c = dt_bias if dt_bias.is_contiguous() else dt_bias.contiguous()

    B, L, D = u_c.shape
    N_STATE = a_c.shape[1]
    y = torch.empty_like(u_c)

    if B == 0 or L == 0 or D == 0:
        return y

    stride_u_b = L * D
    stride_u_l = D
    stride_u_d = 1
    stride_bc_b = L * N_STATE

    if N_STATE == 16:
        _mamba_ssm_scan_n16_kernel[lambda meta: (m1triton.cdiv(D, meta["BLOCK_D"]), B)](
            u_c,
            dt_c,
            a_c,
            b_c,
            c_c,
            d_skip_c,
            dt_bias_c,
            y,
            B,
            L,
            D,
            stride_u_b,
            stride_u_l,
            stride_u_d,
            stride_bc_b,
            wait=True,
        )
        return y

    block_d = 64
    num_chunks = m1triton.cdiv(N_STATE, 16)
    with m1triton.runtime.command_batch(wait=True):
        for chunk_idx in range(num_chunks):
            _mamba_ssm_scan_chunk16_kernel[(m1triton.cdiv(D, block_d), B)](
                u_c,
                dt_c,
                a_c,
                b_c,
                c_c,
                d_skip_c,
                dt_bias_c,
                y,
                B,
                L,
                D,
                N_STATE,
                chunk_idx * 16,
                stride_u_b,
                stride_u_l,
                stride_u_d,
                stride_bc_b,
                N_STATE,
                IS_FIRST_CHUNK=(chunk_idx == 0),
                BLOCK_D=block_d,
                wait=False,
                sync_mps=False,
            )
    return y


__all__ = ["candidate_forward"]
