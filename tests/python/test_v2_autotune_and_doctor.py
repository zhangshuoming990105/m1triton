"""Tests for V2-M0 Autotuner v2 (@heuristics, disk cache, restore_value, PSO pruning), doctor CLI, and examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import m1triton
import m1triton.doctor as doctor
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.triton_compat as triton
from m1triton.autotune import Autotuner, Config, Heuristics, get_autotune_cache_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple Silicon MPS backend is required for M1Triton GPU tests",
)


def test_version_and_exports():
    assert m1triton.__version__ == "0.2.0"
    assert m1triton.Heuristics is Heuristics
    assert m1triton.heuristics is triton.heuristics
    assert triton.Heuristics is Heuristics


def test_heuristics_standalone_and_stacked_with_autotune(tmp_path, monkeypatch):
    monkeypatch.setenv("M1TRITON_CACHE_DIR", str(tmp_path))

    @m1triton.heuristics(
        {
            "BLOCK_SIZE": lambda args: min(1024, m1triton.next_power_of_2(args["n_elements"])),
            "ADD_ONE": lambda args: int(args["n_elements"] % 2 == 1),
        }
    )
    @m1triton.jit
    def heur_only_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ADD_ONE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        res = tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask)
        if ADD_ONE == 1:
            res = res + 1.0
        tl.store(out_ptr + offs, res, mask=mask)

    assert isinstance(heur_only_add, Heuristics)

    n_odd = 513
    x = torch.randn(n_odd, device="mps", dtype=torch.float32)
    y = torch.randn(n_odd, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    grid = lambda meta: (m1triton.cdiv(n_odd, meta["BLOCK_SIZE"]),)
    heur_only_add[grid](x, y, out, n_odd)
    rt.synchronize()
    torch.testing.assert_close(out.cpu(), (x + y + 1.0).cpu(), rtol=1e-5, atol=1e-5)

    @m1triton.autotune(
        configs=[
            Config({"BLOCK_SIZE": 256}, num_warps=4),
            Config({"BLOCK_SIZE": 512}, num_warps=4),
        ],
        key=["n_elements"],
        warmup=1,
        rep=2,
    )
    @m1triton.heuristics({"IS_DIVISIBLE": lambda args: int(args["n_elements"] % args["BLOCK_SIZE"] == 0)})
    @m1triton.jit
    def autotune_above_heuristics(
        x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, IS_DIVISIBLE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if IS_DIVISIBLE == 1:
            x_v = tl.load(x_ptr + offs)
            y_v = tl.load(y_ptr + offs)
            tl.store(out_ptr + offs, x_v + y_v)
        else:
            mask = offs < n_elements
            x_v = tl.load(x_ptr + offs, mask=mask)
            y_v = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x_v + y_v, mask=mask)

    n_div = 4096
    x2 = torch.randn(n_div, device="mps", dtype=torch.float32)
    y2 = torch.randn(n_div, device="mps", dtype=torch.float32)
    out2 = torch.empty_like(x2)
    torch.mps.synchronize()

    grid2 = lambda meta: (m1triton.cdiv(n_div, meta["BLOCK_SIZE"]),)
    autotune_above_heuristics[grid2](x2, y2, out2, n_div)
    rt.synchronize()
    assert autotune_above_heuristics.best_config is not None
    torch.testing.assert_close(out2.cpu(), (x2 + y2).cpu(), rtol=1e-5, atol=1e-5)

    @triton.heuristics({"SCALE_TWO": lambda args: int(args["n_elements"] >= 1024)})
    @triton.autotune(
        configs=[
            Config({"BLOCK_SIZE": 256}, num_warps=4),
            Config({"BLOCK_SIZE": 512}, num_warps=4),
        ],
        key=["n_elements"],
        warmup=1,
        rep=2,
    )
    @triton.jit
    def heuristics_above_autotune(
        x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SCALE_TWO: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        val = tl.load(x_ptr + offs, mask=mask)
        if SCALE_TWO == 1:
            val = val * 2.0
        tl.store(out_ptr + offs, val, mask=mask)

    out3 = torch.empty_like(x2)
    heuristics_above_autotune[grid2](x2, out3, n_div)
    rt.synchronize()
    assert heuristics_above_autotune.best_config is not None
    torch.testing.assert_close(out3.cpu(), (x2 * 2.0).cpu(), rtol=1e-5, atol=1e-5)


def test_autotune_persistent_disk_cache_hit_zero_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setenv("M1TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("M1TRITON_DISABLE_AUTOTUNE_CACHE", raising=False)

    @m1triton.jit
    def disk_cached_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

    configs = [
        Config({"BLOCK_SIZE": 256}, num_warps=4),
        Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ]
    n = 16384
    x = torch.randn(n, device="mps", dtype=torch.float32)
    y = torch.randn(n, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()
    grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)

    tuner1 = Autotuner(disk_cached_add, configs=configs, key=["n_elements"], warmup=1, rep=2)
    tuner1[grid](x, y, out, n)
    rt.synchronize()
    assert tuner1.bench_count == len(configs)
    assert tuner1.disk_cache_hits == 0
    assert tuner1.best_config is not None
    cache_files = list(get_autotune_cache_dir().glob("*.json"))
    assert len(cache_files) == 1

    tuner2 = Autotuner(disk_cached_add, configs=configs, key=["n_elements"], warmup=1, rep=2)

    def _fail_if_benchmarked(*args, **kwargs):
        raise AssertionError("Second Autotuner instance should hit disk cache with 0 benchmarks")

    monkeypatch.setattr(tuner2, "_bench_config", _fail_if_benchmarked)
    out2 = torch.empty_like(x)
    tuner2[grid](x, y, out2, n)
    rt.synchronize()

    assert tuner2.bench_count == 0
    assert tuner2.disk_cache_hits == 1
    assert tuner2.best_config is not None
    assert tuner2.best_config.signature() == tuner1.best_config.signature()
    torch.testing.assert_close(out2.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-5)

    monkeypatch.setenv("M1TRITON_DISABLE_AUTOTUNE_CACHE", "1")
    tuner3 = Autotuner(disk_cached_add, configs=configs, key=["n_elements"], warmup=1, rep=2)
    tuner3[grid](x, y, out2, n)
    rt.synchronize()
    assert tuner3.disk_cache_hits == 0
    assert tuner3.bench_count == len(configs)


def test_autotune_restore_value_and_post_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("M1TRITON_CACHE_DIR", str(tmp_path))

    hook_events: list[bool] = []

    def _post_hook(named_args: dict, had_exception: bool) -> None:
        assert "x_ptr" in named_args
        hook_events.append(had_exception)

    @m1triton.autotune(
        configs=[
            Config({"BLOCK_SIZE": 256}, num_warps=4),
            Config({"BLOCK_SIZE": 512}, num_warps=4),
        ],
        key=["n_elements"],
        warmup=2,
        rep=3,
        restore_value=["x_ptr"],
        post_hook=_post_hook,
        use_disk_cache=False,
    )
    @m1triton.jit
    def inplace_increment_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(x_ptr + offs, v + 1.0, mask=mask)

    n = 4096
    x_init = torch.arange(n, device="mps", dtype=torch.float32)
    x = x_init.clone()
    torch.mps.synchronize()

    grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)
    inplace_increment_kernel[grid](x, n)
    rt.synchronize()

    assert inplace_increment_kernel.bench_count == 2
    assert len(hook_events) == 2 * (2 + 3)
    assert all(exc is False for exc in hook_events)
    torch.testing.assert_close(x.cpu(), (x_init + 1.0).cpu(), rtol=0.0, atol=0.0)


def test_autotune_pso_aware_pruning(tmp_path, monkeypatch):
    monkeypatch.setenv("M1TRITON_CACHE_DIR", str(tmp_path))

    @m1triton.jit
    def pso_probe_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 3.0, mask=mask)

    n = 4096
    x = torch.randn(n, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    torch.mps.synchronize()

    c_small = pso_probe_kernel.compile(x, out, n, BLOCK_SIZE=256)
    c_large = pso_probe_kernel.compile(x, out, n, BLOCK_SIZE=1024)
    large_tg = int(c_large.launch_spec.tg_x) * int(c_large.launch_spec.tg_y) * int(c_large.launch_spec.tg_z)

    def _fake_max_tg(kernel_id: int) -> int:
        if kernel_id == c_large.kernel_id:
            return max(1, large_tg - 1)
        return 1024

    monkeypatch.setattr(rt, "kernel_max_threads_per_threadgroup", _fake_max_tg, raising=False)

    cfg_small = Config({"BLOCK_SIZE": 256}, num_warps=4)
    cfg_large = Config({"BLOCK_SIZE": 1024}, num_warps=8)
    tuner = Autotuner(
        pso_probe_kernel,
        configs=[cfg_large, cfg_small],
        key=["n_elements"],
        warmup=1,
        rep=2,
        use_disk_cache=False,
    )
    grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)
    tuner[grid](x, out, n)
    rt.synchronize()

    assert tuner.best_config is cfg_small
    assert tuner.bench_count == 1
    assert c_small.kernel_id != c_large.kernel_id
    torch.testing.assert_close(out.cpu(), (x * 3.0).cpu(), rtol=1e-5, atol=1e-5)


def test_doctor_api_and_cli_json():
    report = doctor.run_doctor()
    assert report["healthy"] is True
    assert report["m1triton_version"] == "0.2.0"
    assert report["gpu"]["available"] is True
    assert report["gpu"]["has_unified_memory"] is True
    assert report["gpu"]["max_threadgroup_memory_bytes"] >= 32768
    assert report["pytorch"]["mps_available"] is True
    assert report["smoke_tests"]["vector_add"]["ok"] is True
    assert report["smoke_tests"]["vector_add"]["bandwidth_gbs"] > 0.0
    assert report["smoke_tests"]["matmul"]["ok"] is True

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'python'}:{REPO_ROOT / 'build'}"
    proc = subprocess.run(
        [sys.executable, "-m", "m1triton", "doctor", "--json"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"doctor --json failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    parsed = json.loads(proc.stdout)
    assert parsed["healthy"] is True
    assert parsed["smoke_tests"]["vector_add"]["ok"] is True
    assert parsed["smoke_tests"]["matmul"]["ok"] is True


@pytest.mark.parametrize(
    "script_rel",
    [
        "examples/01_vector_add.py",
        "examples/02_autotuned_matmul.py",
    ],
)
def test_examples_run_end_to_end(script_rel: str, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'python'}:{REPO_ROOT / 'build'}"
    env["M1TRITON_CACHE_DIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_rel)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script_rel} failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "Correctness  : PASS" in proc.stdout
