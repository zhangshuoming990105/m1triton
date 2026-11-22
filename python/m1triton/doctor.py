"""End-user health diagnostics CLI (`python -m m1triton doctor` / `m1triton-doctor`)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import m1triton
import m1triton.language as tl
from m1triton.autotune import get_autotune_cache_dir


def _check_clang() -> dict[str, Any]:
    preferred = Path("/usr/bin/clang++")
    clang_path = str(preferred) if preferred.is_file() and os.access(preferred, os.X_OK) else shutil.which("clang++")
    if not clang_path:
        return {
            "available": False,
            "path": None,
            "version": None,
            "is_apple_clang": False,
        }
    try:
        proc = subprocess.run(
            [clang_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        first_line = (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr).strip() else ""
        return {
            "available": proc.returncode == 0,
            "path": clang_path,
            "version": first_line,
            "is_apple_clang": "Apple clang" in first_line,
        }
    except Exception as exc:
        return {
            "available": False,
            "path": clang_path,
            "version": str(exc),
            "is_apple_clang": False,
        }


def _check_gpu() -> dict[str, Any]:
    try:
        from m1triton import runtime as rt

        info = rt.device_info()
        spec = rt.target_spec()
        result: dict[str, Any] = {
            "available": bool(info.available),
            "name": str(info.name),
            "has_unified_memory": bool(info.has_unified_memory),
            "max_threadgroup_memory_bytes": int(info.max_threadgroup_memory_bytes),
            "max_threadgroup_memory_kb": round(int(info.max_threadgroup_memory_bytes) / 1024.0, 1),
            "max_threads_per_threadgroup": int(info.max_threads_per_threadgroup),
            "recommended_max_working_set_gb": round(
                int(info.recommended_max_working_set_bytes) / (1024.0 ** 3), 2
            ),
            "target_sram_bytes": int(spec.max_threadgroup_memory_bytes),
            "target_simd_width": int(spec.simd_width),
        }
        if hasattr(info, "supports_metal3"):
            result["supports_metal3"] = bool(getattr(info, "supports_metal3"))
        if hasattr(info, "supports_indirect_command_buffers"):
            result["supports_indirect_command_buffers"] = bool(
                getattr(info, "supports_indirect_command_buffers")
            )
        return result
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def _check_pytorch() -> dict[str, Any]:
    try:
        import torch

        mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
        mps_avail = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        return {
            "available": True,
            "version": str(torch.__version__),
            "mps_built": mps_built,
            "mps_available": mps_avail,
        }
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "mps_built": False,
            "mps_available": False,
            "error": str(exc),
        }


def _is_dir_writable(path: Path) -> bool:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return os.access(probe, os.W_OK)


def _check_cache_paths() -> dict[str, Any]:
    env_cache = os.environ.get("M1TRITON_CACHE_DIR", "").strip()
    metal_dir = Path(env_cache).expanduser() if env_cache else Path(".m1triton_cache")
    autotune_dir = get_autotune_cache_dir()
    disabled = os.environ.get("M1TRITON_DISABLE_AUTOTUNE_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "env_M1TRITON_CACHE_DIR": env_cache or None,
        "metal_binary_archive_dir": str(metal_dir),
        "metal_dir_writable": _is_dir_writable(metal_dir),
        "autotune_cache_dir": str(autotune_dir),
        "autotune_dir_writable": _is_dir_writable(autotune_dir),
        "autotune_cache_disabled": disabled,
    }


def _check_opt_binary() -> dict[str, Any]:
    from m1triton.__main__ import find_m1triton_opt

    opt_path = find_m1triton_opt()
    return {
        "available": opt_path is not None,
        "path": str(opt_path) if opt_path is not None else None,
    }


@m1triton.jit
def _doctor_vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@m1triton.jit
def _doctor_matmul_kernel(
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


def _run_gpu_smoke_tests() -> dict[str, Any]:
    import torch
    from m1triton import runtime as rt

    results: dict[str, Any] = {}

    n_elements = 4 * 1024 * 1024
    block_size = 1024
    try:
        torch.manual_seed(0)
        x = torch.randn(n_elements, device="mps", dtype=torch.float32)
        y = torch.randn(n_elements, device="mps", dtype=torch.float32)
        out = torch.empty_like(x)
        torch.mps.synchronize()

        grid = (m1triton.cdiv(n_elements, block_size),)
        for _ in range(2):
            _doctor_vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size, wait=True, sync_mps=False)

        times_ms: list[float] = []
        for _ in range(5):
            compiled = _doctor_vector_add_kernel[grid](
                x, y, out, n_elements, BLOCK_SIZE=block_size, wait=True, sync_mps=False
            )
            if compiled.last_gpu_ms > 0.0:
                times_ms.append(compiled.last_gpu_ms)

        expected = x + y
        torch.mps.synchronize()
        max_abs_err = float((out.cpu() - expected.cpu()).abs().max().item())
        med_ms = float(statistics.median(times_ms)) if times_ms else 0.0
        total_bytes = 3.0 * float(n_elements) * 4.0
        bw_gbs = (total_bytes / (med_ms * 1e-3)) / 1e9 if med_ms > 0.0 else 0.0
        ok = max_abs_err <= 1e-5
        results["vector_add"] = {
            "ok": ok,
            "n_elements": n_elements,
            "max_abs_err": max_abs_err,
            "median_gpu_ms": round(med_ms, 4),
            "bandwidth_gbs": round(bw_gbs, 2),
        }
        del x, y, out, expected
    except Exception as exc:
        results["vector_add"] = {"ok": False, "error": str(exc)}

    m = n = k = 128
    bm = bn = 32
    bk = 16
    try:
        torch.manual_seed(1)
        a = torch.randn((m, k), device="mps", dtype=torch.float32)
        b = torch.randn((k, n), device="mps", dtype=torch.float32)
        c = torch.empty((m, n), device="mps", dtype=torch.float32)
        torch.mps.synchronize()

        grid_mm = (m1triton.cdiv(m, bm), m1triton.cdiv(n, bn))
        for _ in range(2):
            _doctor_matmul_kernel[grid_mm](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                wait=True, sync_mps=False,
            )

        mm_times_ms: list[float] = []
        for _ in range(5):
            compiled = _doctor_matmul_kernel[grid_mm](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                wait=True, sync_mps=False,
            )
            if compiled.last_gpu_ms > 0.0:
                mm_times_ms.append(compiled.last_gpu_ms)

        rt.synchronize()
        ref = torch.matmul(a, b)
        torch.mps.synchronize()
        max_abs_err = float((c.cpu() - ref.cpu()).abs().max().item())
        med_ms = float(statistics.median(mm_times_ms)) if mm_times_ms else 0.0
        gflops = (2.0 * m * n * k / (med_ms * 1e-3)) / 1e9 if med_ms > 0.0 else 0.0
        ok = max_abs_err <= 1e-3
        results["matmul"] = {
            "ok": ok,
            "shape": [m, n, k],
            "max_abs_err": max_abs_err,
            "median_gpu_ms": round(med_ms, 4),
            "gflops": round(gflops, 2),
        }
        del a, b, c, ref
    except Exception as exc:
        results["matmul"] = {"ok": False, "error": str(exc)}

    return results


def run_doctor(*, run_smoke: bool = True) -> dict[str, Any]:
    """Run environment, hardware, toolchain, cache, and live GPU diagnostics."""
    mac_ver = platform.mac_ver()[0] or platform.release()
    python_info = {
        "version": platform.python_version(),
        "executable": sys.executable,
    }
    macos_info = {
        "system": platform.system(),
        "version": mac_ver,
        "machine": platform.machine(),
    }
    gpu_info = _check_gpu()
    clang_info = _check_clang()
    pytorch_info = _check_pytorch()
    cache_info = _check_cache_paths()
    opt_info = _check_opt_binary()

    issues: list[str] = []
    fixes: list[str] = []

    if sys.version_info < (3, 11):
        issues.append(f"Python {python_info['version']} is below minimum supported 3.11")
        fixes.append("Install Python 3.11+ (e.g., via conda or python.org) and recreate your environment.")

    if macos_info["system"] != "Darwin" or macos_info["machine"] != "arm64":
        issues.append(f"Unsupported host platform {macos_info['system']}/{macos_info['machine']} (requires macOS arm64)")
        fixes.append("Run M1Triton on an Apple Silicon Mac (arm64 macOS).")

    if not gpu_info.get("available"):
        issues.append(f"Metal GPU device unavailable: {gpu_info.get('error', 'MTLCreateSystemDefaultDevice failed')}")
        fixes.append("Verify Metal access on macOS and rebuild the native `_C` extension (`pip install -e .`).")
    elif not gpu_info.get("has_unified_memory"):
        issues.append("Metal device does not report Apple Silicon unified memory (UMA)")
        fixes.append("Ensure execution is targeting the integrated Apple Silicon GPU.")

    if not clang_info.get("available"):
        issues.append("Apple Clang (`/usr/bin/clang++`) was not found")
        fixes.append("Install Xcode Command Line Tools: `xcode-select --install`.")

    if not pytorch_info.get("available"):
        issues.append("PyTorch is not installed")
        fixes.append("Install PyTorch with Apple Silicon MPS support: `pip install torch`.")
    elif not pytorch_info.get("mps_available"):
        issues.append("PyTorch MPS backend is not available")
        fixes.append("Upgrade PyTorch on macOS arm64 (`pip install --upgrade torch`) and verify macOS >= 13.")

    if not cache_info.get("autotune_dir_writable"):
        issues.append(f"Autotune cache directory is not writable: {cache_info['autotune_cache_dir']}")
        fixes.append("Set `M1TRITON_CACHE_DIR` to a writable directory or set `M1TRITON_DISABLE_AUTOTUNE_CACHE=1`.")

    smoke_tests: dict[str, Any] = {}
    if run_smoke:
        if gpu_info.get("available") and pytorch_info.get("mps_available"):
            smoke_tests = _run_gpu_smoke_tests()
            for name, res in smoke_tests.items():
                if not res.get("ok"):
                    issues.append(f"GPU smoke test '{name}' failed: {res.get('error', res)}")
                    fixes.append("Rebuild the native extension (`bash scripts/ci_local.sh` or `pip install -e .`) and retry.")
        else:
            smoke_tests = {"skipped": True, "reason": "Metal GPU or PyTorch MPS unavailable"}

    return {
        "healthy": len(issues) == 0,
        "m1triton_version": m1triton.__version__,
        "python": python_info,
        "macos": macos_info,
        "gpu": gpu_info,
        "clang": clang_info,
        "pytorch": pytorch_info,
        "cache": cache_info,
        "m1triton_opt": opt_info,
        "smoke_tests": smoke_tests,
        "issues": issues,
        "fixes": fixes,
    }


def format_report(report: dict[str, Any]) -> str:
    """Format a `run_doctor()` dictionary into a human-readable health summary."""
    status = "HEALTHY" if report["healthy"] else "UNHEALTHY"
    gpu = report["gpu"]
    clang = report["clang"]
    pt = report["pytorch"]
    cache = report["cache"]
    smoke = report["smoke_tests"]
    opt = report.get("m1triton_opt", {})

    lines = [
        f"M1Triton Doctor v{report['m1triton_version']} [{status}]",
        f"  Python : {report['python']['version']} ({report['python']['executable']})",
        f"  macOS  : {report['macos']['system']} {report['macos']['version']} ({report['macos']['machine']})",
    ]
    if gpu.get("available"):
        sram_kb = gpu.get("max_threadgroup_memory_kb", 32.0)
        uma = "UMA" if gpu.get("has_unified_memory") else "non-UMA"
        extra = []
        if "supports_metal3" in gpu:
            extra.append(f"Metal3={gpu['supports_metal3']}")
        if "supports_indirect_command_buffers" in gpu:
            extra.append(f"ICB={gpu['supports_indirect_command_buffers']}")
        extra_str = f", {', '.join(extra)}" if extra else ""
        lines.append(
            f"  GPU    : {gpu['name']} ({uma}, {sram_kb:.0f} KB TG SRAM, "
            f"max {gpu['max_threads_per_threadgroup']} thr/TG, "
            f"{gpu['recommended_max_working_set_gb']:.2f} GB working set{extra_str})"
        )
    else:
        lines.append(f"  GPU    : UNAVAILABLE ({gpu.get('error', 'unknown')})")

    if clang.get("available"):
        lines.append(f"  Clang  : {clang['path']} ({clang['version']})")
    else:
        lines.append("  Clang  : MISSING")

    if pt.get("available"):
        lines.append(
            f"  PyTorch: {pt['version']} (MPS built={pt['mps_built']}, available={pt['mps_available']})"
        )
    else:
        lines.append("  PyTorch: MISSING")

    lines.append(
        f"  Cache  : metal={cache['metal_binary_archive_dir']}, "
        f"autotune={cache['autotune_cache_dir']} "
        f"(env M1TRITON_CACHE_DIR={cache['env_M1TRITON_CACHE_DIR'] or '<default>'})"
    )
    lines.append(f"  CLI Opt: {opt.get('path') or 'not found'}")

    if smoke:
        if smoke.get("skipped"):
            lines.append(f"  Smoke  : SKIPPED ({smoke.get('reason')})")
        else:
            va = smoke.get("vector_add", {})
            mm = smoke.get("matmul", {})
            if va.get("ok"):
                lines.append(
                    f"  Smoke/vector_add : PASS (n={va['n_elements']}, err={va['max_abs_err']:.2e}, "
                    f"{va['median_gpu_ms']:.3f} ms, {va['bandwidth_gbs']:.1f} GB/s)"
                )
            elif va:
                lines.append(f"  Smoke/vector_add : FAIL ({va.get('error', va)})")
            if mm.get("ok"):
                lines.append(
                    f"  Smoke/matmul     : PASS (shape={mm['shape']}, err={mm['max_abs_err']:.2e}, "
                    f"{mm['median_gpu_ms']:.3f} ms, {mm['gflops']:.1f} GFLOPS)"
                )
            elif mm:
                lines.append(f"  Smoke/matmul     : FAIL ({mm.get('error', mm)})")

    if report["issues"]:
        lines.append("  Issues :")
        for issue in report["issues"]:
            lines.append(f"    - {issue}")
        lines.append("  Fixes  :")
        for fix in report["fixes"]:
            lines.append(f"    - {fix}")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="m1triton-doctor",
        description="Inspect Apple Silicon Metal/PyTorch environment and run M1Triton GPU smoke checks.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit structured JSON report.")
    parser.add_argument("--no-smoke", action="store_true", help="Skip live GPU smoke test kernels.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_doctor(run_smoke=not args.no_smoke)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
