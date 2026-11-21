"""Reusable M1Triton GEMM, Split-K, GEMV, diagonal, and matrix-scalar kernels."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx


def _is_16b_aligned(t: torch.Tensor) -> bool:
    if t.device.type == "mps":
        return (int(t.storage_offset()) * int(t.element_size())) % 16 == 0
    return (int(t.data_ptr()) % 16) == 0


@m1triton.jit
def sgmm_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SG: tl.constexpr = 4,
    SWIZZLE_W: tl.constexpr = 8,
    TRANS_A: tl.constexpr = 0,
    TRANS_B: tl.constexpr = 0,
    STRUCTURE: tl.constexpr = "dense",
):
    tlx.tile_spec(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_simdgroups=NUM_SG,
        double_buffer=False,
        split_k=1,
        swizzle_w=SWIZZLE_W,
    )
    if STRUCTURE != "dense":
        tlx.structure(STRUCTURE)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + pid_b.to(tl.int64) * stride_ab
    b_base = b_ptr + pid_b.to(tl.int64) * stride_bb
    c_base = c_ptr + pid_b.to(tl.int64) * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        if TRANS_A == 0:
            a_ptrs = a_base + offs_m[:, None] * stride_am + k_idx[None, :]
        else:
            a_ptrs = a_base + offs_m[:, None] + k_idx[None, :] * stride_ak
        if TRANS_B == 0:
            b_ptrs = b_base + k_idx[:, None] * stride_bk + offs_n[None, :]
        else:
            b_ptrs = b_base + k_idx[:, None] + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@m1triton.jit
def sgmm_splitk_kernel(
    a_ptr,
    b_ptr,
    w_ptr,
    M,
    N,
    K,
    k_chunk,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_wb,
    stride_wm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_SG: tl.constexpr = 8,
):
    tlx.tile_spec(
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_simdgroups=NUM_SG,
        double_buffer=False,
        split_k=SPLIT_K,
        swizzle_w=8,
    )

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    w_base = w_ptr + pid_z.to(tl.int64) * stride_wb
    k_start = pid_z * k_chunk
    k_end = min((pid_z + 1) * k_chunk, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        k_idx = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idx[None, :]
        b_ptrs = b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :]
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    w_ptrs = w_base + offs_m[:, None] * stride_wm + offs_n[None, :]
    w_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(w_ptrs, acc, mask=w_mask)


@m1triton.jit
def splitk_reduce_kernel(
    w_ptr,
    c_ptr,
    n_elements,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_VEC4: tl.constexpr = 1,
):
    if USE_VEC4:
        tlx.vec_width(4)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    acc = tl.load(w_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc = acc + tl.load(w_ptr + (s * n_elements + offs), mask=mask, other=0.0)

    tl.store(c_ptr + offs, acc, mask=mask)


@m1triton.jit
def gemv_4row_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    stride_am,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    r0 = pid * 4
    r1 = r0 + 1
    r2 = r0 + 2
    r3 = r0 + 3

    m0 = r0 < M
    m1 = r1 < M
    m2 = r2 < M
    m3 = r3 < M

    a_base0 = a_ptr + r0.to(tl.int64) * stride_am
    a_base1 = a_ptr + r1.to(tl.int64) * stride_am
    a_base2 = a_ptr + r2.to(tl.int64) * stride_am
    a_base3 = a_ptr + r3.to(tl.int64) * stride_am

    offs_k = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        b = tl.load(b_ptr + k_offs, mask=k_mask, other=0.0)
        a0 = tl.load(a_base0 + k_offs, mask=k_mask & m0, other=0.0)
        a1 = tl.load(a_base1 + k_offs, mask=k_mask & m1, other=0.0)
        a2 = tl.load(a_base2 + k_offs, mask=k_mask & m2, other=0.0)
        a3 = tl.load(a_base3 + k_offs, mask=k_mask & m3, other=0.0)
        acc0 += a0 * b
        acc1 += a1 * b
        acc2 += a2 * b
        acc3 += a3 * b

    out0 = tl.sum(acc0, axis=0)
    out1 = tl.sum(acc1, axis=0)
    out2 = tl.sum(acc2, axis=0)
    out3 = tl.sum(acc3, axis=0)

    tl.store(c_ptr + r0, out0, mask=m0)
    tl.store(c_ptr + r1, out1, mask=m1)
    tl.store(c_ptr + r2, out2, mask=m2)
    tl.store(c_ptr + r3, out3, mask=m3)


@m1triton.jit
def mat_scalar_mul_kernel(
    a_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a * scalar, mask=mask)


@m1triton.jit(hints=[tlx.structure("diagonal")])
def diag_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    M,
    BLOCK_M: tl.constexpr,
    USE_VEC4: tl.constexpr = 1,
):
    if USE_VEC4:
        tlx.vec_width(4)

    pid_col = tl.program_id(0)
    row = tl.program_id(1)
    col_offs = pid_col * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = col_offs < M
    a_val = tl.load(a_ptr + row, mask=row < N, other=0.0)
    row_off = row.to(tl.int64) * M
    b_vec = tl.load(b_ptr + (row_off + col_offs), mask=mask)
    tl.store(c_ptr + (row_off + col_offs), a_val * b_vec, mask=mask)


def _select_sgmm_schedule(m: int, n: int, k: int, structure: str) -> tuple[int, int, int, int, int]:
    if structure != "dense":
        return 64, 64, 32, 4, 8
    if k <= 64:
        return 64, 64, 32, 4, 1
    if (m % 64 != 0) or (n % 64 != 0) or (k % 4 != 0):
        return 64, 128, 32, 8, 8
    if m >= 128 and n >= 64:
        return 128, 64, 32, 8, 8
    return 64, 64, 32, 4, 8


def gemm_2d(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    structure: str = "dense",
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_sg: int | None = None,
    swizzle_w: int | None = None,
) -> torch.Tensor:
    a_c = A.contiguous()
    b_c = B.contiguous()

    if trans_a:
        k, m = a_c.shape
        stride_am = a_c.stride(1)
        stride_ak = a_c.stride(0)
    else:
        m, k = a_c.shape
        stride_am = a_c.stride(0)
        stride_ak = a_c.stride(1)

    if trans_b:
        n, k_b = b_c.shape
        stride_bk = b_c.stride(1)
        stride_bn = b_c.stride(0)
    else:
        k_b, n = b_c.shape
        stride_bk = b_c.stride(0)
        stride_bn = b_c.stride(1)

    if k != k_b:
        raise ValueError(f"Incompatible GEMM K dimensions: {k} vs {k_b}")

    bm, bn, bk, sg, sw = _select_sgmm_schedule(m, n, k, structure)
    bm = bm if block_m is None else block_m
    bn = bn if block_n is None else block_n
    bk = bk if block_k is None else block_k
    sg = sg if num_sg is None else num_sg
    sw = sw if swizzle_w is None else swizzle_w

    out = torch.empty((m, n), dtype=a_c.dtype, device=a_c.device)
    grid = (m1triton.cdiv(m, bm), m1triton.cdiv(n, bn), 1)
    sgmm_gemm_kernel[grid](
        a_c,
        b_c,
        out,
        m,
        n,
        k,
        0,
        stride_am,
        stride_ak,
        0,
        stride_bk,
        stride_bn,
        0,
        out.stride(0),
        out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        NUM_SG=sg,
        SWIZZLE_W=sw,
        TRANS_A=1 if trans_a else 0,
        TRANS_B=1 if trans_b else 0,
        STRUCTURE=structure,
        fast_math=True,
    )
    return out


def bmm_3d(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 32,
    num_sg: int = 8,
    swizzle_w: int = 8,
) -> torch.Tensor:
    a_c = A.contiguous()
    b_c = B.contiguous()
    batch, m, k = a_c.shape
    b_batch, k_b, n = b_c.shape
    if batch != b_batch or k != k_b:
        raise ValueError(f"Incompatible BMM shapes: {tuple(a_c.shape)} vs {tuple(b_c.shape)}")

    out = torch.empty((batch, m, n), dtype=a_c.dtype, device=a_c.device)
    grid = (m1triton.cdiv(m, block_m), m1triton.cdiv(n, block_n), batch)
    sgmm_gemm_kernel[grid](
        a_c,
        b_c,
        out,
        m,
        n,
        k,
        a_c.stride(0),
        a_c.stride(1),
        a_c.stride(2),
        b_c.stride(0),
        b_c.stride(1),
        b_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_SG=num_sg,
        SWIZZLE_W=swizzle_w,
        TRANS_A=0,
        TRANS_B=0,
        STRUCTURE="dense",
        fast_math=True,
    )
    return out


def gemm_splitk(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    split_k: int = 32,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
    num_sg: int = 8,
    reduce_block: int = 1024,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_c = A.contiguous()
    b_c = B.contiguous()
    m, k = a_c.shape
    k_b, n = b_c.shape
    if k != k_b:
        raise ValueError(f"Incompatible Split-K GEMM K dimensions: {k} vs {k_b}")

    if (
        workspace is None
        or workspace.shape != (split_k, m, n)
        or workspace.dtype != a_c.dtype
        or workspace.device != a_c.device
    ):
        workspace = torch.empty((split_k, m, n), dtype=a_c.dtype, device=a_c.device)

    out = torch.empty((m, n), dtype=a_c.dtype, device=a_c.device)
    k_tiles = m1triton.cdiv(k, block_k)
    k_chunk = max(block_k, m1triton.cdiv(k_tiles, split_k) * block_k)

    grid_mm = (m1triton.cdiv(m, block_m), m1triton.cdiv(n, block_n), split_k)
    sgmm_splitk_kernel[grid_mm](
        a_c,
        b_c,
        workspace,
        m,
        n,
        k,
        k_chunk,
        a_c.stride(0),
        a_c.stride(1),
        b_c.stride(0),
        b_c.stride(1),
        workspace.stride(0),
        workspace.stride(1),
        workspace.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        NUM_SG=num_sg,
        fast_math=True,
    )

    n_elements = m * n
    aligned = _is_16b_aligned(workspace) and _is_16b_aligned(out)
    grid_red = (m1triton.cdiv(n_elements, reduce_block),)
    splitk_reduce_kernel[grid_red](
        workspace,
        out,
        n_elements,
        SPLIT_K=split_k,
        BLOCK_SIZE=reduce_block,
        USE_VEC4=1 if aligned else 0,
        sync_mps=False,
    )
    return out, workspace


def gemv(A: torch.Tensor, B: torch.Tensor, *, block_k: int = 1024) -> torch.Tensor:
    a_c = A.contiguous()
    b_c = B.contiguous()
    m, k = a_c.shape
    out = torch.empty((m, 1), dtype=a_c.dtype, device=a_c.device)
    grid = (m1triton.cdiv(m, 4),)
    gemv_4row_kernel[grid](
        a_c,
        b_c,
        out,
        m,
        k,
        a_c.stride(0),
        BLOCK_K=block_k,
    )
    return out


def mat_scalar_mul(A: torch.Tensor, s: float, *, block_size: int = 1024) -> torch.Tensor:
    a_c = A.contiguous()
    out = torch.empty_like(a_c)
    n_elements = a_c.numel()
    grid = (m1triton.cdiv(n_elements, block_size),)
    mat_scalar_mul_kernel[grid](
        a_c,
        out,
        float(s),
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


def diag_matmul(A: torch.Tensor, B: torch.Tensor, *, block_m: int = 1024) -> torch.Tensor:
    a_c = A.contiguous()
    b_c = B.contiguous()
    n, m = b_c.shape
    out = torch.empty_like(b_c)
    aligned = _is_16b_aligned(b_c) and _is_16b_aligned(out)
    grid = (m1triton.cdiv(m, block_m), n)
    diag_matmul_kernel[grid](
        a_c,
        b_c,
        out,
        n,
        m,
        BLOCK_M=block_m,
        USE_VEC4=1 if aligned else 0,
    )
    return out


__all__ = [
    "bmm_3d",
    "diag_matmul",
    "diag_matmul_kernel",
    "gemm_2d",
    "gemm_splitk",
    "gemv",
    "gemv_4row_kernel",
    "mat_scalar_mul",
    "mat_scalar_mul_kernel",
    "sgmm_gemm_kernel",
    "sgmm_splitk_kernel",
    "splitk_reduce_kernel",
]
