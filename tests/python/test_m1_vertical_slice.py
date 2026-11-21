"""End-to-end M1 vertical slice tests: Python @jit -> C++ SSA IR -> passes -> MSL -> Metal GPU."""

from __future__ import annotations

import statistics

import pytest
import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.tlx as tlx
import m1triton.triton_compat as triton
from m1triton.autotune import roofline_filter_configs

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Silicon MPS backend is required for M1Triton GPU tests",
)

MEASURED_M1_MAX_STREAM_GBS = 373.83
MIN_VECTOR_ADD_BW_GBS = 0.75 * MEASURED_M1_MAX_STREAM_GBS


@m1triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


@m1triton.jit
def saxpy_kernel(x_ptr, y_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = alpha * x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@m1triton.jit
def fused_add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@m1triton.jit
def naive_matmul_kernel(
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@pytest.mark.parametrize("n_elements", [262144, 1000003])
def test_vector_add_aligned_and_odd_tail(n_elements: int):
    torch.manual_seed(101 + n_elements)
    x = torch.randn(n_elements, device="mps", dtype=torch.float32)
    y = torch.randn(n_elements, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    block_size = 1024
    grid = lambda meta: (m1triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    before = rt.get_dispatch_count()
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    rt.synchronize()
    after = rt.get_dispatch_count()

    assert after == before + 1
    expected = x + y
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


def test_saxpy_float_scalar():
    torch.manual_seed(202)
    n_elements = 500007
    alpha = 2.75
    x = torch.randn(n_elements, device="mps", dtype=torch.float32)
    y = torch.randn(n_elements, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    block_size = 1024
    grid = (m1triton.cdiv(n_elements, block_size),)

    before = rt.get_dispatch_count()
    saxpy_kernel[grid](x, y, out, alpha, n_elements, BLOCK_SIZE=block_size)
    rt.synchronize()
    after = rt.get_dispatch_count()

    assert after == before + 1
    expected = alpha * x + y
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_elements", [262144, 100003])
def test_fused_add_relu(n_elements: int):
    torch.manual_seed(303 + n_elements)
    x = torch.randn(n_elements, device="mps", dtype=torch.float32)
    y = torch.randn(n_elements, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    block_size = 1024
    grid = (m1triton.cdiv(n_elements, block_size),)

    before = rt.get_dispatch_count()
    fused_add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    rt.synchronize()
    after = rt.get_dispatch_count()

    assert after == before + 1
    expected = torch.relu(x + y)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "m,k,n",
    [
        (256, 256, 256),
        (130, 195, 140),
    ],
)
def test_naive_matmul(m: int, k: int, n: int):
    torch.manual_seed(404 + m + n + k)
    a = torch.randn((m, k), device="mps", dtype=torch.float32)
    b = torch.randn((k, n), device="mps", dtype=torch.float32)
    c = torch.empty((m, n), device="mps", dtype=torch.float32)
    torch.mps.synchronize()

    block_m, block_n, block_k = 32, 32, 16
    grid = (m1triton.cdiv(m, block_m), m1triton.cdiv(n, block_n))

    before = rt.get_dispatch_count()
    naive_matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    rt.synchronize()
    after = rt.get_dispatch_count()

    assert after == before + 1
    expected = torch.matmul(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(c.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)


def test_ir_and_msl_dump_inspection_vectorized_float4(capsys, monkeypatch):
    n_elements = 65536
    x = torch.randn(n_elements, device="mps", dtype=torch.float32)
    y = torch.randn(n_elements, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)

    pre_ir, post_ir = vector_add_kernel.inspect_ir(x, y, out, n_elements, BLOCK_SIZE=1024)
    msl = vector_add_kernel.inspect_msl(x, y, out, n_elements, BLOCK_SIZE=1024)

    assert "vector_add_kernel" in pre_ir
    assert "vector_add_kernel" in post_ir
    assert "kernel void" in msl
    assert "float4" in msl, f"Expected vectorized float4 loads/stores in generated MSL:\n{msl}"

    monkeypatch.setenv("M1TRITON_DUMP_IR", "1")
    monkeypatch.setenv("M1TRITON_DUMP_MSL", "1")
    vector_add_kernel.clear_cache()
    vector_add_kernel[(m1triton.cdiv(n_elements, 1024),)](x, y, out, n_elements, BLOCK_SIZE=1024)
    rt.synchronize()
    captured = capsys.readouterr()
    assert "Pre-Pass IR" in captured.err
    assert "Post-Pass IR" in captured.err
    assert "Generated MSL" in captured.err
    assert "float4" in captured.err


def test_triton_compat_and_specialization_cache_and_tlx_hints():
    @triton.jit
    def hinted_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ADD_BIAS: tl.constexpr):
        tlx.vec_width(4)
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        res = x + y
        if ADD_BIAS:
            res = res + 1.0
        tl.store(out_ptr + offs, res, mask=mask)

    n = 4096
    x = torch.randn(n, device="mps", dtype=torch.float32)
    y = torch.randn(n, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    grid = (triton.cdiv(n, 1024),)
    hinted_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, ADD_BIAS=True)
    rt.synchronize()
    assert hinted_add_kernel.cache_size == 1
    torch.testing.assert_close(out.cpu(), (x + y + 1.0).cpu(), rtol=1e-5, atol=1e-5)

    # Identical specialization hits cache without recompiling.
    hinted_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, ADD_BIAS=True)
    rt.synchronize()
    assert hinted_add_kernel.cache_size == 1

    # Different constexpr branch compiles a second specialization.
    hinted_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024, ADD_BIAS=False)
    rt.synchronize()
    assert hinted_add_kernel.cache_size == 2
    torch.testing.assert_close(out.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-5)


def test_autotune_roofline_filter_and_dispatch():
    oversized = m1triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2)
    valid_a = m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=1)
    too_many_threads = m1triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=64, num_stages=1)
    filtered = roofline_filter_configs([oversized, valid_a, too_many_threads], elem_bytes=4)
    assert filtered == [valid_a]

    @m1triton.autotune(
        configs=[
            m1triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
            m1triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
        warmup=2,
        rep=5,
    )
    @m1triton.jit
    def tuned_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 131072
    x = torch.randn(n, device="mps", dtype=torch.float32)
    y = torch.randn(n, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)
    tuned_add[grid](x, y, out, n)
    rt.synchronize()
    assert tuned_add.best_config is not None
    torch.testing.assert_close(out.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-5)


def test_vector_add_memory_bandwidth_64m():
    n_floats = 64 * 1024 * 1024
    x = torch.randn(n_floats, device="mps", dtype=torch.float32)
    y = torch.randn(n_floats, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    block_size = 1024
    grid = (m1triton.cdiv(n_floats, block_size),)

    try:
        for _ in range(5):
            vector_add_kernel[grid](x, y, out, n_floats, BLOCK_SIZE=block_size, wait=True, sync_mps=False)

        torch.testing.assert_close(out[:4096].cpu(), (x[:4096] + y[:4096]).cpu(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out[-4096:].cpu(), (x[-4096:] + y[-4096:]).cpu(), rtol=1e-5, atol=1e-5)

        times_ms: list[float] = []
        for _ in range(15):
            res = vector_add_kernel[grid](x, y, out, n_floats, BLOCK_SIZE=block_size, wait=True, sync_mps=False)
            assert res.last_gpu_ms > 0.0
            times_ms.append(res.last_gpu_ms)

        median_ms = statistics.median(times_ms)
        total_bytes = 3.0 * float(n_floats) * 4.0
        achieved_gbs = (total_bytes / (median_ms * 1e-3)) / 1e9
        assert achieved_gbs >= MIN_VECTOR_ADD_BW_GBS, (
            f"vector_add achieved {achieved_gbs:.2f} GB/s (median {median_ms:.3f} ms), "
            f"below 75% M1 Max STREAM target ({MIN_VECTOR_ADD_BW_GBS:.2f} GB/s)"
        )
    finally:
        del x, y, out
        torch.mps.empty_cache()
