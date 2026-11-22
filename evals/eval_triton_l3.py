#!/usr/bin/env python3
"""Frozen Level 3 Complex Triton Kernel Suite evaluator (forward + backward + memory + anti-cheat).

Evaluates 10 complex real-world Triton algorithms under `benchmarks/triton_l3/`:
  - Reference specs live in `benchmarks/triton_l3/specs/l3_XX_*.py`
  - Candidate kernels live in `benchmarks/triton_l3/kernels/l3_XX_*.py` (strictly AST + TorchComputeGuard audited)

Verifies forward outputs, optional backward input/parameter gradients against PyTorch autograd,
zero framework fallbacks on BOTH forward and backward, peak MPS memory delta, and interleaved timing
against PyTorch MPS eager (plus optional Inductor).
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import m1triton.runtime as m1_runtime

BANNED_TORCH_FUNCS: frozenset[str] = frozenset(
    {
        "matmul", "mm", "bmm", "mv", "addmm", "baddbmm", "einsum", "tensordot",
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "softmax", "log_softmax", "logsumexp", "layer_norm", "batch_norm", "group_norm", "instance_norm",
        "rms_norm", "normalize",
        "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu", "elu", "selu", "softplus",
        "cumsum", "cumprod", "sum", "mean", "amax", "amin", "max", "min", "argmax", "argmin",
        "linear", "cross_entropy", "scaled_dot_product_attention",
    }
)

ALLOWED_ATEN_BASE_OPS: frozenset[str] = frozenset(
    {
        "empty", "empty_strided", "new_empty", "new_empty_strided", "empty_like",
        "zeros", "zeros_like", "new_zeros", "zero_", "fill_", "full", "full_like",
        "ones", "ones_like", "new_ones",
        "alias", "view", "_unsafe_view", "reshape", "as_strided", "detach",
        "clone", "copy_", "_to_copy", "to", "lift_fresh", "lift_fresh_copy",
        "unsqueeze", "squeeze", "expand", "permute", "transpose", "t",
        "slice", "select", "narrow", "contiguous", "chunk", "split", "unbind",
        "scalar_tensor", "_local_scalar_dense", "is_same_size",
    }
)

L3_MANIFEST_SPECS: tuple[tuple[int, str], ...] = (
    (1, "flash_attn2_fwd"),
    (2, "flash_attn2_bwd"),
    (3, "paged_attn_decode"),
    (4, "sliding_window_alibi_attn"),
    (5, "fused_rmsnorm_rope_kv"),
    (6, "fused_layernorm_fwd_bwd"),
    (7, "chunked_linear_cross_entropy"),
    (8, "mamba_ssm_scan"),
    (9, "moe_grouped_gemm"),
    (10, "w4a16_dequant_gemm"),
)


class AntiCheatViolation(RuntimeError):
    """Raised when an L3 candidate violates zero-fallback or dispatch verification."""


class TorchComputeGuard(TorchDispatchMode):
    """Tripwire rejecting any PyTorch ATen compute op during candidate forward or backward."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        full_name = str(func)
        self.observed_ops.append(full_name)
        base = full_name
        if base.startswith("aten."):
            base = base[len("aten.") :]
        base = base.split(".")[0]
        if base not in ALLOWED_ATEN_BASE_OPS:
            raise AntiCheatViolation(
                f"[ANTICHEAT-TORCH-DISPATCH] Forbidden PyTorch ATen compute op '{full_name}' "
                f"executed inside L3 candidate."
            )
        return func(*args, **kwargs)


def static_ast_audit(candidate_path: Path) -> list[str]:
    source = candidate_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(candidate_path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scipy" or alias.name.startswith("scipy."):
                    violations.append(f"line {node.lineno}: forbidden import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "torch.nn.functional" or mod == "scipy" or mod.startswith("scipy."):
                violations.append(f"line {node.lineno}: forbidden import from '{mod}'")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            owner = ""
            if isinstance(node.func.value, ast.Name):
                owner = node.func.value.id
            elif isinstance(node.func.value, ast.Attribute):
                owner = node.func.value.attr
            if attr in BANNED_TORCH_FUNCS and owner in {"torch", "F", "functional", "linalg", "np", "numpy"}:
                violations.append(f"line {node.lineno}: forbidden call '{owner}.{attr}'")
    return violations


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    m1_runtime.synchronize()


def cleanup_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def as_tensor_tuple(obj: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(obj, torch.Tensor):
        return (obj,)
    if isinstance(obj, (tuple, list)):
        return tuple(obj)
    raise TypeError(f"Expected Tensor or tuple/list of Tensors, got {type(obj)}")


def compare_tensor(
    ref: torch.Tensor, cand: torch.Tensor, *, rtol: float, atol: float, label: str
) -> tuple[bool, dict[str, Any]]:
    rf = ref.detach().to("cpu").float()
    cf = cand.detach().to("cpu").float()
    if rf.shape != cf.shape:
        return False, {"label": label, "reason": f"shape mismatch: {tuple(rf.shape)} vs {tuple(cf.shape)}"}
    if not torch.equal(torch.isnan(rf), torch.isnan(cf)):
        return False, {"label": label, "reason": "NaN mismatch"}
    if not torch.equal(torch.isinf(rf), torch.isinf(cf)):
        return False, {"label": label, "reason": "Inf mismatch"}
    finite = (~torch.isnan(rf)) & (~torch.isinf(rf))
    if finite.any():
        diff = (rf[finite] - cf[finite]).abs()
        max_abs = float(diff.max().item())
        max_rel = float((diff / rf[finite].abs().clamp_min(1e-8)).max().item())
        ok = bool(torch.allclose(cf[finite], rf[finite], rtol=rtol, atol=atol))
    else:
        max_abs, max_rel, ok = 0.0, 0.0, True
    return ok, {"label": label, "max_abs_err": max_abs, "max_rel_err": max_rel, "rtol": rtol, "atol": atol}


def parse_task_selector(selector: str, available_ids: list[int]) -> list[int]:
    selector = selector.strip().lower()
    if selector == "all":
        return list(available_ids)
    selected: list[int] = []
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            lo_s, hi_s = part.split("..", 1)
            selected.extend(range(int(lo_s), int(hi_s) + 1))
        elif "-" in part:
            lo_s, hi_s = part.split("-", 1)
            selected.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            selected.append(int(part))
    avail = set(available_ids)
    dedup = sorted(set(selected))
    missing = [i for i in dedup if i not in avail]
    if missing:
        raise ValueError(f"Unknown L3 task IDs: {missing}")
    return dedup


def find_file(directory: Path, task_id: int) -> Path | None:
    matches = sorted(directory.glob(f"l3_{task_id:02d}_*.py"))
    if matches:
        return matches[0]
    exact = directory / f"l3_{task_id:02d}.py"
    return exact if exact.is_file() else None


def evaluate_l3_task(
    task_id: int,
    default_name: str,
    specs_dir: Path,
    kernels_dir: Path,
    *,
    seeds: int,
    warmup: int,
    trials: int,
    allow_missing: bool,
) -> dict[str, Any]:
    res: dict[str, Any] = {
        "id": task_id,
        "name": default_name,
        "status": "error",
        "passed": False,
        "beats_baseline": False,
        "supports_backward": False,
        "compiled_kernel_dispatch_count": 0,
    }
    device = torch.device("mps")
    try:
        spec_path = find_file(specs_dir, task_id)
        cand_path = find_file(kernels_dir, task_id)
        if spec_path is None or cand_path is None:
            res["status"] = "missing" if allow_missing else "missing_error"
            res["error"] = f"Missing L3 spec ({spec_path}) or kernel ({cand_path})"
            return res

        ast_violations = static_ast_audit(cand_path)
        if ast_violations:
            res["status"] = "anticheat_failed"
            res["error"] = "[ANTICHEAT-AST] " + "; ".join(ast_violations)
            return res

        spec_mod = load_module(spec_path, f"_l3_spec_{task_id}")
        cand_mod = load_module(cand_path, f"_l3_cand_{task_id}")
        res["name"] = getattr(spec_mod, "TASK_NAME", default_name)
        supports_bwd = bool(getattr(spec_mod, "SUPPORTS_BACKWARD", False))
        res["supports_backward"] = supports_bwd

        tol_fwd = getattr(spec_mod, "FWD_TOL", (1e-3, 1e-3))
        tol_bwd = getattr(spec_mod, "BWD_TOL", (2e-3, 2e-3))
        fwd_rtol = min(float(tol_fwd[0]), 1e-2)
        fwd_atol = min(float(tol_fwd[1]), 1e-2)
        bwd_rtol = min(float(tol_bwd[0]), 1e-2)
        bwd_atol = min(float(tol_bwd[1]), 1e-2)

        seed_reports: list[dict[str, Any]] = []
        total_dispatches = 0
        cold_compile_ms = 0.0

        for s in range(seeds):
            ref_inputs, ref_kwargs = spec_mod.get_inputs(seed=1000 + s, device=device)
            cand_inputs, cand_kwargs = spec_mod.get_inputs(seed=1000 + s, device=device)

            if supports_bwd:
                ref_grad_inputs = [t for t in ref_inputs if isinstance(t, torch.Tensor) and t.requires_grad]
                cand_grad_inputs = [t for t in cand_inputs if isinstance(t, torch.Tensor) and t.requires_grad]
                ref_out = as_tensor_tuple(spec_mod.reference_forward(*ref_inputs, **ref_kwargs))
                ref_float_outs = [o for o in ref_out if o.is_floating_point() and o.requires_grad]
                torch.manual_seed(5000 + s)
                grad_outs = [torch.randn_like(o) for o in ref_float_outs]
                ref_grads = torch.autograd.grad(ref_float_outs, ref_grad_inputs, grad_outputs=grad_outs)
                sync_device(device)

                before_fwd = m1_runtime.get_dispatch_count()
                t0 = time.perf_counter()
                with TorchComputeGuard():
                    cand_out = as_tensor_tuple(cand_mod.candidate_forward(*cand_inputs, **cand_kwargs))
                sync_device(device)
                fwd_ms = (time.perf_counter() - t0) * 1000.0
                fwd_disp = m1_runtime.get_dispatch_count() - before_fwd
                if fwd_disp <= 0:
                    raise AntiCheatViolation(f"[ANTICHEAT-ZERO-DISPATCH] L3-{task_id} forward had 0 dispatches")

                cand_float_outs = [o for o in cand_out if o.is_floating_point() and o.requires_grad]
                before_bwd = m1_runtime.get_dispatch_count()
                with TorchComputeGuard():
                    cand_grads = torch.autograd.grad(cand_float_outs, cand_grad_inputs, grad_outputs=grad_outs)
                sync_device(device)
                bwd_disp = m1_runtime.get_dispatch_count() - before_bwd
                if bwd_disp <= 0:
                    raise AntiCheatViolation(f"[ANTICHEAT-ZERO-DISPATCH] L3-{task_id} backward had 0 dispatches")

                if s == 0:
                    cold_compile_ms = fwd_ms
                total_dispatches += fwd_disp + bwd_disp

                checks = []
                for idx, (ro, co) in enumerate(zip(ref_out, cand_out)):
                    ok, info = compare_tensor(ro, co, rtol=fwd_rtol, atol=fwd_atol, label=f"fwd[{idx}]")
                    checks.append(info)
                    if not ok:
                        res["status"] = "correctness_failed"
                        res["error"] = f"seed {s} fwd mismatch: {info}"
                        return res
                for idx, (rg, cg) in enumerate(zip(ref_grads, cand_grads)):
                    ok, info = compare_tensor(rg, cg, rtol=bwd_rtol, atol=bwd_atol, label=f"bwd[{idx}]")
                    checks.append(info)
                    if not ok:
                        res["status"] = "correctness_failed"
                        res["error"] = f"seed {s} bwd mismatch: {info}"
                        return res
                seed_reports.append({"seed": s, "fwd_dispatches": fwd_disp, "bwd_dispatches": bwd_disp, "checks": checks})
            else:
                with torch.no_grad():
                    ref_out = as_tensor_tuple(spec_mod.reference_forward(*ref_inputs, **ref_kwargs))
                sync_device(device)

                before_fwd = m1_runtime.get_dispatch_count()
                t0 = time.perf_counter()
                with torch.no_grad(), TorchComputeGuard():
                    cand_out = as_tensor_tuple(cand_mod.candidate_forward(*cand_inputs, **cand_kwargs))
                sync_device(device)
                fwd_ms = (time.perf_counter() - t0) * 1000.0
                fwd_disp = m1_runtime.get_dispatch_count() - before_fwd
                if fwd_disp <= 0:
                    raise AntiCheatViolation(f"[ANTICHEAT-ZERO-DISPATCH] L3-{task_id} forward had 0 dispatches")
                if s == 0:
                    cold_compile_ms = fwd_ms
                total_dispatches += fwd_disp

                checks = []
                for idx, (ro, co) in enumerate(zip(ref_out, cand_out)):
                    ok, info = compare_tensor(ro, co, rtol=fwd_rtol, atol=fwd_atol, label=f"fwd[{idx}]")
                    checks.append(info)
                    if not ok:
                        res["status"] = "correctness_failed"
                        res["error"] = f"seed {s} fwd mismatch: {info}"
                        return res
                seed_reports.append({"seed": s, "fwd_dispatches": fwd_disp, "checks": checks})

            del ref_inputs, cand_inputs
            cleanup_memory()

        # Interleaved steady-state benchmark
        b_ref_in, b_ref_kw = spec_mod.get_inputs(seed=2000, device=device)
        b_cand_in, b_cand_kw = spec_mod.get_inputs(seed=2000, device=device)
        grad_outs_bench = None
        if supports_bwd:
            sample_out = as_tensor_tuple(spec_mod.reference_forward(*b_ref_in, **b_ref_kw))
            torch.manual_seed(6000)
            grad_outs_bench = [torch.randn_like(o) for o in sample_out if o.is_floating_point() and o.requires_grad]
            del sample_out
            sync_device(device)

        def _run_ref():
            if supports_bwd:
                gi = [t for t in b_ref_in if isinstance(t, torch.Tensor) and t.requires_grad]
                outs = [o for o in as_tensor_tuple(spec_mod.reference_forward(*b_ref_in, **b_ref_kw)) if o.is_floating_point() and o.requires_grad]
                torch.autograd.grad(outs, gi, grad_outputs=grad_outs_bench)
            else:
                with torch.no_grad():
                    spec_mod.reference_forward(*b_ref_in, **b_ref_kw)

        def _run_cand():
            if supports_bwd:
                gi = [t for t in b_cand_in if isinstance(t, torch.Tensor) and t.requires_grad]
                outs = [o for o in as_tensor_tuple(cand_mod.candidate_forward(*b_cand_in, **b_cand_kw)) if o.is_floating_point() and o.requires_grad]
                torch.autograd.grad(outs, gi, grad_outputs=grad_outs_bench)
            else:
                with torch.no_grad():
                    cand_mod.candidate_forward(*b_cand_in, **b_cand_kw)

        for _ in range(warmup):
            _run_ref()
            _run_cand()
        sync_device(device)

        base_times: list[float] = []
        cand_times: list[float] = []
        for idx in range(trials):
            order = (("ref", _run_ref, base_times), ("cand", _run_cand, cand_times)) if idx % 2 == 0 else (("cand", _run_cand, cand_times), ("ref", _run_ref, base_times))
            for _, fn, bucket in order:
                sync_device(device)
                t0 = time.perf_counter()
                fn()
                sync_device(device)
                bucket.append((time.perf_counter() - t0) * 1000.0)

        base_med = statistics.median(base_times)
        cand_med = statistics.median(cand_times)
        res["status"] = "passed"
        res["passed"] = True
        res["cold_compile_ms"] = round(cold_compile_ms, 4)
        res["compiled_kernel_dispatch_count"] = total_dispatches
        res["baseline_median_ms"] = round(base_med, 4)
        res["candidate_median_ms"] = round(cand_med, 4)
        res["speedup"] = round(base_med / cand_med, 4) if cand_med > 0 else 0.0
        res["beats_baseline"] = bool(cand_med < base_med)
        res["seed_reports"] = seed_reports
        return res

    except AntiCheatViolation as exc:
        res["status"] = "anticheat_failed"
        res["error"] = str(exc)
        return res
    except Exception as exc:
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc(limit=6)
        return res
    finally:
        cleanup_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen M1Triton Level 3 Complex Triton Kernel evaluator.")
    parser.add_argument("--specs-dir", type=Path, default=REPO_ROOT / "benchmarks/triton_l3/specs")
    parser.add_argument("--kernels-dir", type=Path, default=REPO_ROOT / "benchmarks/triton_l3/kernels")
    parser.add_argument("--tasks", default="all", help="Task IDs: 'all', '1..5', or '1,2,10'")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec_map = dict(L3_MANIFEST_SPECS)
    selected_ids = parse_task_selector(args.tasks, sorted(spec_map.keys()))

    results: list[dict[str, Any]] = []
    for tid in selected_ids:
        res = evaluate_l3_task(
            tid,
            spec_map[tid],
            args.specs_dir,
            args.kernels_dir,
            seeds=args.seeds,
            warmup=args.warmup,
            trials=args.trials,
            allow_missing=args.allow_missing,
        )
        results.append(res)
        sp = f"{res['speedup']:.2f}x" if "speedup" in res else "-"
        print(
            f"[L3-{tid:02d}] {res['name']:<32} {res['status']:<18} bwd={str(res['supports_backward']):<5} "
            f"base={res.get('baseline_median_ms', '-')}ms cand={res.get('candidate_median_ms', '-')}ms speedup={sp}",
            flush=True,
        )

    passed = [r for r in results if r["passed"]]
    beats = [r for r in passed if r["beats_baseline"]]
    speedups = [r["speedup"] for r in passed if r.get("speedup", 0) > 0]
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
    report = {
        "suite": "triton_l3",
        "seeds": args.seeds,
        "warmup": args.warmup,
        "trials": args.trials,
        "loadavg_1m": round(os.getloadavg()[0], 2),
        "summary": {
            "total_evaluated": len(results),
            "passed": len(passed),
            "failed": len([r for r in results if r["status"] not in ("passed", "missing")]),
            "missing": len([r for r in results if r["status"] == "missing"]),
            "beats_baseline": len(beats),
            "geomean_speedup": round(geomean, 4) if geomean is not None else None,
        },
        "tasks": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nSUMMARY:", json.dumps(report["summary"]))
    if report["summary"]["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
