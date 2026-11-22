"""Tests for V2-M0 PSO introspection, single-command-buffer batching, Metal ICB graphs, and m1triton-opt CLI."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import m1triton
from m1triton import language as tl
from m1triton import runtime as rt

_ROOT_DIR = Path(__file__).resolve().parents[2]


@m1triton.jit
def _affine_kernel(x_ptr, out_ptr, scale, bias, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * scale + bias
    tl.store(out_ptr + offs, y, mask=mask)


@m1triton.jit
def _add_two_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def test_device_info_and_pso_introspection():
    info = rt.device_info()
    assert info.available is True
    assert info.has_unified_memory is True
    assert isinstance(info.supports_metal3, bool)
    assert isinstance(info.supports_indirect_command_buffers, bool)
    assert info.supports_metal3 is True
    assert info.supports_indirect_command_buffers is True

    kid = rt.compile_kernel(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void pso_probe_kernel(
            device const float* x [[buffer(0)]],
            device float* y [[buffer(1)]],
            constant int& n [[buffer(2)]],
            uint gid [[thread_position_in_grid]]) {
            if ((int)gid < n) y[gid] = x[gid] + 1.0f;
        }
        """,
        "pso_probe_kernel",
    )
    max_tg = rt.kernel_max_threads_per_threadgroup(kid)
    assert max_tg > 0
    assert max_tg <= info.max_threads_per_threadgroup
    assert (max_tg % 32) == 0
    assert rt.kernel_supports_icb(kid) is True

    assert rt.kernel_max_threads_per_threadgroup(0) == 0
    assert rt.kernel_max_threads_per_threadgroup(999_999) == 0


def test_command_batch_dependent_chain_mps():
    assert torch.backends.mps.is_available()
    n = 65536
    block = 256
    grid = (m1triton.cdiv(n, block),)

    x = torch.randn(n, device="mps", dtype=torch.float32)
    t1 = torch.empty_like(x)
    t2 = torch.empty_like(x)
    t3 = torch.empty_like(x)
    out = torch.empty_like(x)

    # Warm up compilation before counting dispatches.
    _affine_kernel.compile(x, t1, 2.0, 1.0, n, BLOCK_SIZE=block)
    _add_two_kernel.compile(t3, x, out, n, BLOCK_SIZE=block)

    rt.reset_dispatch_count()
    assert rt.in_batch() is False

    with rt.command_batch(wait=True, sync_mps=True) as batch:
        assert rt.in_batch() is True
        _affine_kernel[grid](x, t1, 2.0, 1.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](t1, t2, 3.0, -2.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](t2, t3, 0.5, 4.0, n, BLOCK_SIZE=block, sync_mps=False)
        _add_two_kernel[grid](t3, x, out, n, BLOCK_SIZE=block, sync_mps=False)

    assert rt.in_batch() is False
    assert rt.get_dispatch_count() == 4
    assert batch.gpu_ms > 0.0

    ref = ((x * 2.0 + 1.0) * 3.0 - 2.0) * 0.5 + 4.0 + x
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_command_batch_cpu_buffers_and_error_handling():
    n = 1024
    block = 128
    grid = (m1triton.cdiv(n, block),)

    x = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    t1 = np.zeros(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)

    with rt.command_batch(wait=False, sync_mps=False) as batch:
        _affine_kernel[grid](x, t1, 2.5, -1.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](t1, out, -0.5, 3.0, n, BLOCK_SIZE=block, sync_mps=False)

    assert rt.in_batch() is False
    assert batch.gpu_ms > 0.0
    ref = (x * 2.5 - 1.0) * (-0.5) + 3.0
    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)

    # Empty batch succeeds cleanly.
    with rt.command_batch(wait=True, sync_mps=False) as empty_batch:
        pass
    assert empty_batch.gpu_ms == 0.0
    assert rt.in_batch() is False

    # Requesting wait=True inside an active batch raises cleanly and aborts batch state.
    with pytest.raises(RuntimeError, match="wait=True"):
        with rt.command_batch(wait=True, sync_mps=False):
            _affine_kernel[grid](x, out, 1.0, 0.0, n, BLOCK_SIZE=block, wait=True, sync_mps=False)
    assert rt.in_batch() is False


def test_command_batch_cpu_slice_aliasing_waw_and_strided_mps_span():
    # 1. CPU slice aliasing inside command_batch: write full buffer, then read overlapping sub-slice.
    buf = np.zeros(512, dtype=np.float32)
    out = np.zeros(256, dtype=np.float32)
    with rt.command_batch(wait=True, sync_mps=False):
        _affine_kernel[(4,)](buf, buf, 1.0, 7.0, 512, BLOCK_SIZE=128, sync_mps=False)
        _affine_kernel[(2,)](buf[64:320], out, 1.0, 1.0, 256, BLOCK_SIZE=128, sync_mps=False)
    np.testing.assert_allclose(out, np.full(256, 8.0, dtype=np.float32), atol=1e-5, rtol=1e-5)

    # 2. Disjoint-nbytes WAW on same base pointer: write 512 elems then overwrite first 256 elems.
    waw = np.zeros(512, dtype=np.float32)
    with rt.command_batch(wait=True, sync_mps=False):
        _affine_kernel[(4,)](waw, waw, 0.0, 5.0, 512, BLOCK_SIZE=128, sync_mps=False)
        _affine_kernel[(2,)](waw[:256], waw[:256], 0.0, 9.0, 256, BLOCK_SIZE=128, sync_mps=False)
    expected_waw = np.concatenate([np.full(256, 9.0, dtype=np.float32), np.full(256, 5.0, dtype=np.float32)])
    np.testing.assert_allclose(waw, expected_waw, atol=1e-5, rtol=1e-5)

    # 3. Strided MPS view reports full spanned byte extent so overlapping views trigger barriers.
    base = torch.zeros((64, 64), device="mps", dtype=torch.float32)
    col0 = base[:, 0]
    row32 = base[32, :]
    arg_col = rt.make_buffer_arg(col0, 0, writable=True)
    arg_row = rt.make_buffer_arg(row32, 1, writable=False)
    assert arg_col.byte_offset == 0
    assert arg_col.nbytes == (63 * 64 + 1) * 4
    assert arg_row.byte_offset == 32 * 64 * 4
    assert arg_col.byte_offset + arg_col.nbytes > arg_row.byte_offset


def test_metal_graph_icb_dependent_chain_and_in_place_mutation():
    assert torch.backends.mps.is_available()
    n = 65536
    block = 256
    grid = (m1triton.cdiv(n, block),)

    x = torch.randn(n, device="mps", dtype=torch.float32)
    t1 = torch.empty_like(x)
    t2 = torch.empty_like(x)
    t3 = torch.empty_like(x)
    out = torch.empty_like(x)

    def ref_fn(inp: torch.Tensor) -> torch.Tensor:
        s1 = inp * 2.0 + 1.0
        s2 = s1 * 3.0 - 2.0
        s3 = s2 * 0.5 + 4.0
        return s3 + inp

    _affine_kernel.compile(x, t1, 2.0, 1.0, n, BLOCK_SIZE=block)
    _add_two_kernel.compile(t3, x, out, n, BLOCK_SIZE=block)

    rt.reset_dispatch_count()
    with rt.MetalGraph() as graph:
        assert rt.in_graph_capture() is True
        _affine_kernel[grid](x, t1, 2.0, 1.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](t1, t2, 3.0, -2.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](t2, t3, 0.5, 4.0, n, BLOCK_SIZE=block, sync_mps=False)
        _add_two_kernel[grid](t3, x, out, n, BLOCK_SIZE=block, sync_mps=False)

    assert rt.in_graph_capture() is False
    assert graph.command_count == 4
    assert rt.get_dispatch_count() == 0

    # First replay on initial x.
    ms1 = graph.replay(wait=True)
    assert ms1 > 0.0
    assert graph.last_gpu_ms == ms1
    assert rt.get_dispatch_count() == 4
    torch.testing.assert_close(out, ref_fn(x), atol=1e-5, rtol=1e-5)

    # Mutate x in place and replay again.
    x2 = torch.randn(n, device="mps", dtype=torch.float32) * 3.0 - 1.5
    x.copy_(x2)
    ms2 = graph.replay(wait=True)
    assert ms2 > 0.0
    assert rt.get_dispatch_count() == 8
    torch.testing.assert_close(out, ref_fn(x2), atol=1e-5, rtol=1e-5)

    # Repeated deterministic replays verify RAW barrier safety.
    for step in range(5):
        x_next = torch.randn(n, device="mps", dtype=torch.float32) + float(step)
        x.copy_(x_next)
        graph.replay(wait=True)
        torch.testing.assert_close(out, ref_fn(x_next), atol=1e-5, rtol=1e-5)

    graph.close()
    assert graph.command_count == 0
    with pytest.raises(RuntimeError, match="closed|not captured"):
        graph.replay(wait=True)


def test_metal_graph_parallel_branches_and_validation_errors():
    assert torch.backends.mps.is_available()
    n = 16384
    block = 256
    grid = (m1triton.cdiv(n, block),)

    x = torch.randn(n, device="mps", dtype=torch.float32)
    y1 = torch.empty_like(x)
    y2 = torch.empty_like(x)
    out = torch.empty_like(x)

    # K1(x->y1) and K2(x->y2) are independent (same concurrent range); K3(y1,y2->out) waits on a barrier.
    with rt.MetalGraph() as graph:
        _affine_kernel[grid](x, y1, 1.5, 2.0, n, BLOCK_SIZE=block, sync_mps=False)
        _affine_kernel[grid](x, y2, -0.75, 0.5, n, BLOCK_SIZE=block, sync_mps=False)
        _add_two_kernel[grid](y1, y2, out, n, BLOCK_SIZE=block, sync_mps=False)

    assert graph.command_count == 3
    graph.replay(wait=True)
    ref = (x * 1.5 + 2.0) + (x * (-0.75) + 0.5)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    graph.close()

    # CPU buffer must be rejected during graph capture and capture state must clean up.
    cpu_x = np.ones(1024, dtype=np.float32)
    cpu_y = np.zeros(1024, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Metal GPU buffers"):
        with rt.MetalGraph():
            _affine_kernel[(4,)](cpu_x, cpu_y, 2.0, 1.0, 1024, BLOCK_SIZE=256, sync_mps=False)
    assert rt.in_graph_capture() is False

    # Empty graph must raise and clean up.
    with pytest.raises(RuntimeError, match="empty MetalGraph"):
        with rt.MetalGraph():
            pass
    assert rt.in_graph_capture() is False


def test_metal_graph_cross_threadgroup_raw_barrier_and_threadgroup_memory():
    assert torch.backends.mps.is_available()

    # Cross-threadgroup shifted-read kernel ensures RAW barrier spans all threadgroups.
    shift_kid = rt.compile_kernel(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void shift_affine_barrier_test(
            device const float* x [[buffer(0)]],
            device float* y [[buffer(1)]],
            constant uint& n [[buffer(2)]],
            constant uint& shift [[buffer(3)]],
            constant float& scale [[buffer(4)]],
            constant float& bias [[buffer(5)]],
            uint gid [[thread_position_in_grid]]) {
            if (gid < n) {
                uint src = (gid + shift) % n;
                y[gid] = x[src] * scale + bias;
            }
        }
        """,
        "shift_affine_barrier_test",
    )
    n = 65536
    shift = 32771
    chain_len = 6
    x = torch.randn(n, device="mps", dtype=torch.float32)
    bufs = [x] + [torch.empty_like(x) for _ in range(chain_len)]
    torch.mps.synchronize()

    with rt.MetalGraph(sync_mps=False) as g:
        for i in range(chain_len):
            rt.launch_kernel(
                shift_kid,
                [rt.make_buffer_arg(bufs[i], 0, writable=False), rt.make_buffer_arg(bufs[i + 1], 1, writable=True)],
                [
                    rt.make_scalar_arg(n, 2, "u32"),
                    rt.make_scalar_arg(shift, 3, "u32"),
                    rt.make_scalar_arg(1.01, 4, "f32"),
                    rt.make_scalar_arg(0.1, 5, "f32"),
                ],
                grid=(n // 256, 1, 1),
                threads_per_threadgroup=(256, 1, 1),
                wait=False,
            )

    for trial in range(5):
        x_in = torch.randn(n, device="mps", dtype=torch.float32) + float(trial)
        x.copy_(x_in)
        bufs[-1].zero_()
        torch.mps.synchronize()
        g.replay(wait=True, sync_mps=False)
        ref = x_in.clone()
        for _ in range(chain_len):
            ref = torch.roll(ref, shifts=-shift, dims=0) * 1.01 + 0.1
        torch.testing.assert_close(bufs[-1], ref, atol=1e-4, rtol=1e-4)
    g.close()

    # Dynamic threadgroup memory binding works inside both command_batch and MetalGraph.
    smem_kid = rt.compile_kernel(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void smem_scale_test(
            device const float* x [[buffer(0)]],
            device float* y [[buffer(1)]],
            constant float& k [[buffer(2)]],
            threadgroup float* scratch [[threadgroup(0)]],
            uint lid [[thread_position_in_threadgroup]],
            uint gid [[thread_position_in_grid]]) {
            scratch[lid] = x[gid] * k;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            y[gid] = scratch[lid] + 1.0f;
        }
        """,
        "smem_scale_test",
    )
    sx = torch.randn(1024, device="mps", dtype=torch.float32)
    sy = torch.empty_like(sx)
    torch.mps.synchronize()

    with rt.command_batch(wait=True, sync_mps=False):
        rt.launch_kernel(
            smem_kid,
            [rt.make_buffer_arg(sx, 0, writable=False), rt.make_buffer_arg(sy, 1, writable=True)],
            [rt.make_scalar_arg(2.5, 2, "f32")],
            grid=(4, 1, 1),
            threads_per_threadgroup=(256, 1, 1),
            threadgroup_mem_bytes=256 * 4,
            wait=False,
        )
    torch.testing.assert_close(sy, sx * 2.5 + 1.0, atol=1e-5, rtol=1e-5)

    sy.zero_()
    torch.mps.synchronize()
    with rt.MetalGraph(sync_mps=False) as sg:
        rt.launch_kernel(
            smem_kid,
            [rt.make_buffer_arg(sx, 0, writable=False), rt.make_buffer_arg(sy, 1, writable=True)],
            [rt.make_scalar_arg(2.5, 2, "f32")],
            grid=(4, 1, 1),
            threads_per_threadgroup=(256, 1, 1),
            threadgroup_mem_bytes=256 * 4,
            wait=False,
        )
    sg.replay(wait=True, sync_mps=False)
    torch.testing.assert_close(sy, sx * 2.5 + 1.0, atol=1e-5, rtol=1e-5)
    sg.close()


def test_metal_graph_and_batch_overhead_vs_unbatched_10_kernel_chain():
    assert torch.backends.mps.is_available()
    n = 8192
    block = 256
    grid = (m1triton.cdiv(n, block),)
    chain_len = 10

    x = torch.randn(n, device="mps", dtype=torch.float32)
    bufs = [x] + [torch.empty_like(x) for _ in range(chain_len)]

    # Compile once and extract raw kernel_id + LaunchSpec for minimal Python launch overhead.
    compiled = _affine_kernel.compile(x, bufs[1], 1.01, 0.1, n, BLOCK_SIZE=block)
    spec = compiled.launch_spec
    tg = (int(spec.tg_x), int(spec.tg_y), int(spec.tg_z))
    smem = int(spec.threadgroup_mem_bytes)

    step_args = []
    for i in range(chain_len):
        b_args = [
            rt.make_buffer_arg(bufs[i], 0, writable=False),
            rt.make_buffer_arg(bufs[i + 1], 1, writable=True),
        ]
        s_args = [
            rt.make_scalar_arg(1.01, 2, "f32"),
            rt.make_scalar_arg(0.1, 3, "f32"),
            rt.make_scalar_arg(n, 4, "i32"),
        ]
        step_args.append((b_args, s_args))

    torch.mps.synchronize()

    with rt.MetalGraph(sync_mps=False) as graph:
        for b_args, s_args in step_args:
            rt.launch_kernel(
                compiled.kernel_id,
                b_args,
                s_args,
                grid=(grid[0], 1, 1),
                threads_per_threadgroup=tg,
                threadgroup_mem_bytes=smem,
                wait=False,
            )
    assert graph.command_count == chain_len

    def run_unbatched(wait_end: bool = True):
        for i, (b_args, s_args) in enumerate(step_args):
            rt.launch_kernel(
                compiled.kernel_id,
                b_args,
                s_args,
                grid=(grid[0], 1, 1),
                threads_per_threadgroup=tg,
                threadgroup_mem_bytes=smem,
                wait=(wait_end and i == chain_len - 1),
            )

    def run_batched(wait_end: bool = True):
        with rt.command_batch(wait=wait_end, sync_mps=False):
            for b_args, s_args in step_args:
                rt.launch_kernel(
                    compiled.kernel_id,
                    b_args,
                    s_args,
                    grid=(grid[0], 1, 1),
                    threads_per_threadgroup=tg,
                    threadgroup_mem_bytes=smem,
                    wait=False,
                )

    def run_graph(wait_end: bool = True):
        graph.replay(wait=wait_end, sync_mps=False)

    ref = x.clone()
    for _ in range(chain_len):
        ref = ref * 1.01 + 0.1

    run_unbatched(wait_end=True)
    torch.testing.assert_close(bufs[-1], ref, atol=1e-4, rtol=1e-4)
    bufs[-1].zero_()
    torch.mps.synchronize()

    run_batched(wait_end=True)
    torch.testing.assert_close(bufs[-1], ref, atol=1e-4, rtol=1e-4)
    bufs[-1].zero_()
    torch.mps.synchronize()

    run_graph(wait_end=True)
    torch.testing.assert_close(bufs[-1], ref, atol=1e-4, rtol=1e-4)

    # Interleaved warm rounds comparing both end-to-end minimum latency and CPU submit overhead.
    unb_wall, bat_wall, grp_wall = [], [], []
    unb_cpu, grp_cpu = [], []
    for r in range(45):
        t0 = time.perf_counter()
        run_unbatched(wait_end=True)
        dt_unb = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        run_batched(wait_end=True)
        dt_bat = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        run_graph(wait_end=True)
        dt_grp = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        run_unbatched(wait_end=False)
        dt_unb_cpu = (time.perf_counter() - t0) * 1000.0
        rt.synchronize()

        t0 = time.perf_counter()
        run_graph(wait_end=False)
        dt_grp_cpu = (time.perf_counter() - t0) * 1000.0
        rt.synchronize()

        if r >= 5:
            unb_wall.append(dt_unb)
            bat_wall.append(dt_bat)
            grp_wall.append(dt_grp)
            unb_cpu.append(dt_unb_cpu)
            grp_cpu.append(dt_grp_cpu)

    graph.close()
    assert min(bat_wall) < min(unb_wall)
    assert min(grp_wall) < min(unb_wall)
    assert min(grp_cpu) < min(unb_cpu)


def test_m1triton_opt_cli_passes_stats_and_dumps():
    opt_bin = _ROOT_DIR / "build" / "m1triton-opt"
    assert opt_bin.is_file(), f"Missing m1triton-opt binary at {opt_bin}"

    x = np.zeros(256, dtype=np.float32)
    y = np.zeros(256, dtype=np.float32)
    pre_ir, _ = _affine_kernel.inspect_ir(x, y, 2.0, 1.0, 256, BLOCK_SIZE=256)

    res = subprocess.run(
        [str(opt_bin), "--passes=m2", "--stats", "--dump-ir-before-all"],
        input=pre_ir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "kernel @_affine_kernel" in res.stdout
    assert "// --- Before pass: canonicalize ---" in res.stderr
    assert "// --- Before pass: dce ---" in res.stderr
    assert "[stats] kernel=@_affine_kernel" in res.stderr
    assert "ops=" in res.stderr
    assert "blocks=" in res.stderr
    assert "params=" in res.stderr

    # Verify --dump-ir-after-all alias and --emit-msl LaunchSpec stats.
    res_msl = subprocess.run(
        [str(opt_bin), "--passes=m2", "--emit-msl", "--stats", "--dump-ir-after-all"],
        input=pre_ir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "kernel void" in res_msl.stdout
    assert "// --- After pass: canonicalize ---" in res_msl.stderr
    assert "// --- After pass: dce ---" in res_msl.stderr
    assert "[stats] kernel=@_affine_kernel" in res_msl.stderr
    assert "tg=(" in res_msl.stderr
    assert "smem=" in res_msl.stderr
    assert "vec=" in res_msl.stderr

    # Verify all 15 individual named passes are registered and runnable.
    all_15 = ",".join(
        [
            "canonicalize",
            "cse",
            "licm",
            "epilogue-fusion",
            "structure-pruning",
            "online-softmax-welford",
            "pool-lowering",
            "conv-lowering",
            "loss-lowering",
            "flash-attn-lowering",
            "contiguity",
            "tile-layout",
            "threadgroup-staging",
            "simdgroup-matmul",
            "dce",
        ]
    )
    res_15 = subprocess.run(
        [str(opt_bin), f"--passes={all_15}", "--stats"],
        input=pre_ir,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "kernel @_affine_kernel" in res_15.stdout
    assert "[stats] kernel=@_affine_kernel" in res_15.stderr
