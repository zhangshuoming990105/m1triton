#!/usr/bin/env python3
"""Frozen KernelBench Level 2 evaluation harness with anti-cheat, fusion telemetry, and chunked comparison.

Verifies M1Triton candidate models (`ModelNew`) against pinned reference models (`Model`)
from `benchmarks/kernelbench_l2/` across multiple random seeds in deterministic `.eval()` mode,
enforces 3-layer zero-fallback anti-cheat checks (static AST + runtime TorchDispatchMode + native
Metal dispatch count), tracks fusion dispatch telemetry (`dispatches_per_forward`), and benchmarks
warm steady-state execution interleaved with the PyTorch MPS eager baseline (plus optional Inductor).
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
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

COMPARE_CHUNK_ELEMS = 16_777_216

BANNED_TORCH_FUNCS: frozenset[str] = frozenset(
    {
        "matmul", "mm", "bmm", "mv", "addmm", "baddbmm", "einsum", "tensordot", "chain_matmul",
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "softmax", "log_softmax", "logsumexp", "layer_norm", "batch_norm", "group_norm", "instance_norm",
        "rms_norm", "normalize",
        "relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu", "elu", "selu", "softplus",
        "softsign", "hardsigmoid", "hardswish", "hardtanh", "mish", "clamp", "clamp_min", "clamp_max",
        "max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d",
        "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d",
        "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
        "cumsum", "cumprod", "sum", "mean", "amax", "amin", "max", "min", "argmax", "argmin",
        "triu", "tril", "diag", "norm", "cdist", "linear",
        "mse_loss", "cross_entropy", "huber_loss", "smooth_l1_loss", "kl_div",
        "triplet_margin_loss", "hinge_embedding_loss", "scaled_dot_product_attention",
    }
)

BANNED_NN_MODULES: frozenset[str] = frozenset(
    {
        "Conv1d", "Conv2d", "Conv3d",
        "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
        "Linear", "Bilinear",
        "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
        "MaxPool1d", "MaxPool2d", "MaxPool3d",
        "AvgPool1d", "AvgPool2d", "AvgPool3d",
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
        "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "ELU", "SELU",
        "Softplus", "Softsign", "Hardsigmoid", "Hardswish", "Hardtanh", "Mish",
        "Softmax", "LogSoftmax",
        "MSELoss", "CrossEntropyLoss", "HuberLoss", "KLDivLoss",
        "TripletMarginLoss", "HingeEmbeddingLoss",
    }
)

ALLOWED_ATEN_BASE_OPS: frozenset[str] = frozenset(
    {
        "empty", "empty_strided", "new_empty", "new_empty_strided", "empty_like",
        "zeros", "zeros_like", "new_zeros", "zero_", "fill_", "full", "full_like",
        "alias", "view", "_unsafe_view", "reshape", "as_strided", "detach",
        "clone", "copy_", "_to_copy", "to", "lift_fresh", "lift_fresh_copy",
        "unsqueeze", "squeeze", "expand", "permute", "transpose", "t",
        "slice", "select", "narrow", "contiguous", "chunk", "split", "unbind",
        "scalar_tensor", "_local_scalar_dense", "is_same_size",
    }
)


class AntiCheatViolation(RuntimeError):
    """Raised when a candidate violates zero-fallback or dispatch verification."""


class TorchComputeGuard(TorchDispatchMode):
    """Tripwire rejecting any PyTorch ATen compute op executed during candidate forward()."""

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
                f"executed inside candidate forward()."
            )
        return func(*args, **kwargs)


def static_ast_audit(candidate_path: Path) -> list[str]:
    """Inspect candidate Python AST for forbidden PyTorch/NumPy/SciPy compute calls."""
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
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                attr = fn.attr
                owner = ""
                if isinstance(fn.value, ast.Name):
                    owner = fn.value.id
                elif isinstance(fn.value, ast.Attribute):
                    owner = fn.value.attr
                if attr in BANNED_TORCH_FUNCS and owner in {"torch", "F", "functional", "linalg", "np", "numpy"}:
                    violations.append(f"line {node.lineno}: forbidden call '{owner}.{attr}'")
                if attr in BANNED_NN_MODULES and owner in {"nn", "torch"}:
                    violations.append(f"line {node.lineno}: forbidden nn module '{owner}.{attr}'")
            elif isinstance(fn, ast.Name):
                if fn.id in BANNED_NN_MODULES:
                    violations.append(f"line {node.lineno}: forbidden nn module '{fn.id}'")
    return violations


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    avail_set = set(available_ids)
    dedup = sorted(set(selected))
    missing = [i for i in dedup if i not in avail_set]
    if missing:
        raise ValueError(f"Unknown KernelBench L2 task IDs: {missing}")
    return dedup


def move_to_device(items: list[Any], device: torch.device) -> list[Any]:
    return [item.to(device) if isinstance(item, torch.Tensor) else item for item in items]


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    m1_runtime.synchronize()


def cleanup_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def compare_outputs(ref: torch.Tensor, cand: torch.Tensor) -> tuple[bool, dict[str, Any]]:
    """Chunked exact NaN/Inf + uniform tolerance check preventing multi-GB CPU swap spikes."""
    ref_cpu = ref.detach().to("cpu")
    cand_cpu = cand.detach().to("cpu")

    if ref_cpu.shape != cand_cpu.shape:
        return False, {
            "reason": f"shape mismatch: ref={tuple(ref_cpu.shape)} vs cand={tuple(cand_cpu.shape)}"
        }

    if ref_cpu.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        if cand_cpu.dtype != ref_cpu.dtype:
            return False, {"reason": f"dtype mismatch: ref={ref_cpu.dtype} vs cand={cand_cpu.dtype}"}
        exact = bool(torch.equal(ref_cpu, cand_cpu))
        mismatches = int((ref_cpu != cand_cpu).sum().item()) if not exact else 0
        return exact, {"exact": exact, "mismatches": mismatches, "rtol": 0.0, "atol": 0.0}

    if ref_cpu.dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-3, 1e-3

    r_flat = ref_cpu.reshape(-1)
    c_flat = cand_cpu.reshape(-1)
    total = int(r_flat.numel())
    max_abs = 0.0
    max_rel = 0.0
    all_ok = True

    for start in range(0, total, COMPARE_CHUNK_ELEMS):
        end = min(total, start + COMPARE_CHUNK_ELEMS)
        rf = r_flat[start:end].float()
        cf = c_flat[start:end].float()
        r_nan = torch.isnan(rf)
        c_nan = torch.isnan(cf)
        if not torch.equal(r_nan, c_nan):
            return False, {
                "reason": f"NaN mismatch: ref_nans={int(r_nan.sum())}, cand_nans={int(c_nan.sum())}"
            }
        r_inf = torch.isinf(rf)
        c_inf = torch.isinf(cf)
        if not torch.equal(r_inf, c_inf):
            return False, {
                "reason": f"Inf mismatch: ref_infs={int(r_inf.sum())}, cand_infs={int(c_inf.sum())}"
            }
        finite = (~r_nan) & (~r_inf)
        if finite.any():
            r_fin = rf[finite]
            c_fin = cf[finite]
            diff = (r_fin - c_fin).abs()
            c_abs = float(diff.max().item())
            if c_abs > max_abs:
                max_abs = c_abs
            denom = r_fin.abs().clamp_min(1e-8)
            c_rel = float((diff / denom).max().item())
            if c_rel > max_rel:
                max_rel = c_rel
            if not bool(torch.allclose(c_fin, r_fin, rtol=rtol, atol=atol)):
                all_ok = False

    return all_ok, {
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "rtol": rtol,
        "atol": atol,
    }


def find_candidate_file(candidate_dir: Path, task_id: int) -> Path | None:
    exact = candidate_dir / f"task_{task_id:03d}.py"
    if exact.is_file():
        return exact
    matches = sorted(candidate_dir.glob(f"task_{task_id:03d}_*.py"))
    if matches:
        return matches[0]
    alt = candidate_dir / f"task_{task_id}.py"
    if alt.is_file():
        return alt
    return None


def compute_iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[(3 * n) // 4] - s[n // 4]


def instantiate_candidate(cand_mod, ref_model_cpu: torch.nn.Module, init_inputs: list[Any], device: torch.device):
    if hasattr(cand_mod, "ModelNew") and hasattr(cand_mod.ModelNew, "from_reference"):
        model = cand_mod.ModelNew.from_reference(ref_model_cpu)
    else:
        model = cand_mod.ModelNew(*copy.deepcopy(init_inputs))
        ref_sd = ref_model_cpu.state_dict()
        if ref_sd:
            missing, _ = model.load_state_dict(ref_sd, strict=False)
            if missing:
                raise RuntimeError(f"Candidate ModelNew missing state_dict keys from reference: {missing}")
    return model.to(device).eval()


def sync_candidate_state(cand_model: torch.nn.Module, ref_model_cpu: torch.nn.Module, initial_state: dict) -> None:
    if not initial_state:
        return
    ref_model_cpu.load_state_dict(initial_state)
    if hasattr(cand_model, "sync_from_reference"):
        cand_model.sync_from_reference(ref_model_cpu)
    else:
        cand_model.load_state_dict(initial_state, strict=False)


def _measure_inductor_ms(
    ref_model_cpu: torch.nn.Module,
    initial_state: dict,
    bench_inputs: list[Any],
    warmup: int,
    trials: int,
    device: torch.device,
) -> float | None:
    if device.type != "mps":
        return None
    try:
        m_ind = copy.deepcopy(ref_model_cpu).to(device).eval()
        if initial_state:
            m_ind.load_state_dict(initial_state)
        compiled = torch.compile(m_ind, backend="inductor")
        with torch.no_grad():
            for _ in range(warmup):
                out = compiled(*bench_inputs)
                del out
            sync_device(device)
            times: list[float] = []
            for _ in range(trials):
                sync_device(device)
                t0 = time.perf_counter()
                out = compiled(*bench_inputs)
                sync_device(device)
                times.append((time.perf_counter() - t0) * 1000.0)
                del out
        return round(statistics.median(times), 4)
    except Exception:
        return None


def evaluate_task(
    task_meta: dict[str, Any],
    kb_dir: Path,
    candidate_dir: Path,
    *,
    baseline_only: bool,
    compare_inductor: bool,
    seeds: int,
    warmup: int,
    trials: int,
    allow_missing: bool,
) -> dict[str, Any]:
    task_id = int(task_meta["id"])
    task_path = kb_dir / task_meta["filename"]
    result: dict[str, Any] = {
        "id": task_id,
        "name": task_meta["name"],
        "filename": task_meta["filename"],
        "status": "error",
        "baseline_backend": "mps",
        "passed": False,
        "beats_baseline": False,
        "compiled_kernel_dispatch_count": 0,
        "dispatches_per_forward": 0,
    }

    mps_device = torch.device("mps")
    cpu_device = torch.device("cpu")

    try:
        ref_mod = load_module(task_path, f"_kb2_ref_{task_id}")
        torch.manual_seed(42)
        init_inputs = list(ref_mod.get_init_inputs())
        ref_model_cpu = ref_mod.Model(*copy.deepcopy(init_inputs)).eval()
        initial_state = copy.deepcopy(ref_model_cpu.state_dict())

        torch.manual_seed(1000)
        probe_inputs_cpu = list(ref_mod.get_inputs())
        ref_backend = "mps"
        try:
            ref_model = copy.deepcopy(ref_model_cpu).to(mps_device).eval()
            probe_inputs_mps = move_to_device(probe_inputs_cpu, mps_device)
            with torch.no_grad():
                probe_out = ref_model(*probe_inputs_mps)
            sync_device(mps_device)
            del probe_out, probe_inputs_mps
        except (NotImplementedError, RuntimeError) as exc:
            ref_backend = "cpu_fallback_mps_unsupported"
            result["mps_unsupported_reason"] = str(exc).splitlines()[0][:240]
            ref_model = copy.deepcopy(ref_model_cpu).to(cpu_device).eval()
            with torch.no_grad():
                probe_out = ref_model(*probe_inputs_cpu)
            del probe_out
        del probe_inputs_cpu
        cleanup_memory()

        result["baseline_backend"] = ref_backend
        ref_device = mps_device if ref_backend == "mps" else cpu_device

        if baseline_only:
            torch.manual_seed(2000)
            base_inputs = move_to_device(list(ref_mod.get_inputs()), ref_device)
            if initial_state:
                ref_model.load_state_dict(initial_state)
            with torch.no_grad():
                for _ in range(warmup):
                    out = ref_model(*base_inputs)
                    del out
                sync_device(ref_device)
                base_times: list[float] = []
                for _ in range(trials):
                    sync_device(ref_device)
                    t0 = time.perf_counter()
                    out = ref_model(*base_inputs)
                    sync_device(ref_device)
                    base_times.append((time.perf_counter() - t0) * 1000.0)
                    del out
            if compare_inductor and ref_device.type == "mps":
                ind_ms = _measure_inductor_ms(ref_model_cpu, initial_state, base_inputs, warmup, trials, ref_device)
                if ind_ms is not None:
                    result["inductor_median_ms"] = ind_ms
            del base_inputs
            result["status"] = "baseline_measured"
            result["baseline_median_ms"] = round(statistics.median(base_times), 4)
            result["baseline_iqr_ms"] = round(compute_iqr(base_times), 4)
            return result

        cand_path = find_candidate_file(candidate_dir, task_id)
        if cand_path is None:
            result["status"] = "missing" if allow_missing else "missing_error"
            result["error"] = f"Candidate file not found for L2 task {task_id} in {candidate_dir}"
            return result
        result["candidate_file"] = str(cand_path)

        ast_violations = static_ast_audit(cand_path)
        if ast_violations:
            result["status"] = "anticheat_failed"
            result["error"] = "[ANTICHEAT-AST] " + "; ".join(ast_violations)
            return result

        cand_mod = load_module(cand_path, f"_kb2_cand_{task_id}")
        cand_model = instantiate_candidate(cand_mod, ref_model_cpu, init_inputs, mps_device)

        seed_reports: list[dict[str, Any]] = []
        cold_compile_ms = 0.0
        total_dispatches = 0
        per_forward_dispatches: list[int] = []

        for s in range(seeds):
            torch.manual_seed(1000 + s)
            ref_inputs = move_to_device(list(ref_mod.get_inputs()), ref_device)
            if initial_state:
                ref_model.load_state_dict(initial_state)
            with torch.no_grad():
                ref_out = ref_model(*ref_inputs)
            sync_device(ref_device)
            ref_out_cpu = ref_out.detach().to("cpu")
            del ref_inputs, ref_out
            cleanup_memory()

            torch.manual_seed(1000 + s)
            cand_inputs = move_to_device(list(ref_mod.get_inputs()), mps_device)
            sync_candidate_state(cand_model, ref_model_cpu, initial_state)
            sync_device(mps_device)

            before_disp = m1_runtime.get_dispatch_count()
            t0 = time.perf_counter()
            with torch.no_grad(), TorchComputeGuard():
                cand_out = cand_model(*cand_inputs)
            sync_device(mps_device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            after_disp = m1_runtime.get_dispatch_count()

            seed_dispatches = after_disp - before_disp
            if s == 0:
                cold_compile_ms = elapsed_ms
            if seed_dispatches <= 0:
                raise AntiCheatViolation(
                    f"[ANTICHEAT-ZERO-DISPATCH] Candidate forward() executed 0 compiled "
                    f"M1Triton Metal kernel dispatches on seed {s}."
                )
            total_dispatches += seed_dispatches
            per_forward_dispatches.append(seed_dispatches)

            ok, cmp_info = compare_outputs(ref_out_cpu, cand_out)
            cmp_info["seed"] = s
            cmp_info["dispatches"] = seed_dispatches
            seed_reports.append(cmp_info)
            del cand_inputs, cand_out, ref_out_cpu
            cleanup_memory()

            if not ok:
                result["status"] = "correctness_failed"
                result["seed_reports"] = seed_reports
                result["compiled_kernel_dispatch_count"] = total_dispatches
                result["dispatches_per_forward"] = min(per_forward_dispatches)
                result["error"] = f"[CORRECTNESS-MISMATCH] seed {s} failed: {cmp_info}"
                return result

        result["cold_compile_ms"] = round(cold_compile_ms, 4)
        result["compiled_kernel_dispatch_count"] = total_dispatches
        result["dispatches_per_forward"] = int(statistics.median(per_forward_dispatches))
        result["seed_reports"] = seed_reports

        torch.manual_seed(2000)
        bench_raw = list(ref_mod.get_inputs())
        cand_bench_inputs = move_to_device(bench_raw, mps_device)
        ref_bench_inputs = cand_bench_inputs if ref_device.type == "mps" else bench_raw
        del bench_raw

        if initial_state:
            ref_model.load_state_dict(initial_state)
            sync_candidate_state(cand_model, ref_model_cpu, initial_state)

        with torch.no_grad():
            for _ in range(warmup):
                out_r = ref_model(*ref_bench_inputs)
                del out_r
                out_c = cand_model(*cand_bench_inputs)
                del out_c
            sync_device(ref_device)
            sync_device(mps_device)

            base_times = []
            cand_times = []
            for trial_idx in range(trials):
                if trial_idx % 2 == 0:
                    sync_device(ref_device)
                    t0 = time.perf_counter()
                    out_r = ref_model(*ref_bench_inputs)
                    sync_device(ref_device)
                    base_times.append((time.perf_counter() - t0) * 1000.0)
                    del out_r

                    sync_device(mps_device)
                    t0 = time.perf_counter()
                    out_c = cand_model(*cand_bench_inputs)
                    sync_device(mps_device)
                    cand_times.append((time.perf_counter() - t0) * 1000.0)
                    del out_c
                else:
                    sync_device(mps_device)
                    t0 = time.perf_counter()
                    out_c = cand_model(*cand_bench_inputs)
                    sync_device(mps_device)
                    cand_times.append((time.perf_counter() - t0) * 1000.0)
                    del out_c

                    sync_device(ref_device)
                    t0 = time.perf_counter()
                    out_r = ref_model(*ref_bench_inputs)
                    sync_device(ref_device)
                    base_times.append((time.perf_counter() - t0) * 1000.0)
                    del out_r

        if compare_inductor and ref_device.type == "mps":
            ind_ms = _measure_inductor_ms(ref_model_cpu, initial_state, ref_bench_inputs, warmup, trials, ref_device)
            if ind_ms is not None:
                result["inductor_median_ms"] = ind_ms
                cand_med_tmp = statistics.median(cand_times)
                result["speedup_vs_inductor"] = round(ind_ms / cand_med_tmp, 4) if cand_med_tmp > 0 else 0.0

        del cand_bench_inputs, ref_bench_inputs

        base_med = statistics.median(base_times)
        cand_med = statistics.median(cand_times)
        speedup = base_med / cand_med if cand_med > 0 else 0.0

        result["status"] = "passed"
        result["passed"] = True
        result["baseline_median_ms"] = round(base_med, 4)
        result["baseline_iqr_ms"] = round(compute_iqr(base_times), 4)
        result["candidate_median_ms"] = round(cand_med, 4)
        result["candidate_iqr_ms"] = round(compute_iqr(cand_times), 4)
        result["speedup"] = round(speedup, 4)
        result["beats_baseline"] = bool(cand_med < base_med)
        return result

    except AntiCheatViolation as exc:
        result["status"] = "anticheat_failed"
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=6)
        return result
    finally:
        cleanup_memory()


def _run_process_isolated(args: argparse.Namespace, selected_ids: list[int], manifest: dict[str, Any]) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    chunk = max(1, int(args.chunk_size))
    with tempfile.TemporaryDirectory(prefix="kb2_iso_") as tmpdir:
        for i in range(0, len(selected_ids), chunk):
            sub_ids = selected_ids[i : i + chunk]
            sub_sel = ",".join(str(x) for x in sub_ids)
            sub_out = Path(tmpdir) / f"chunk_{i:03d}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--kb-dir",
                str(args.kb_dir),
                "--candidate-dir",
                str(args.candidate_dir),
                "--tasks",
                sub_sel,
                "--seeds",
                str(args.seeds),
                "--warmup",
                str(args.warmup),
                "--trials",
                str(args.trials),
                "--out",
                str(sub_out),
            ]
            if args.baseline_only:
                cmd.append("--baseline-only")
            if args.compare_inductor:
                cmd.append("--compare-inductor")
            if args.allow_missing:
                cmd.append("--allow-missing")
            subprocess.run(cmd, check=False)
            if sub_out.is_file():
                sub_data = json.loads(sub_out.read_text(encoding="utf-8"))
                all_results.extend(sub_data.get("tasks", []))
            else:
                for tid in sub_ids:
                    all_results.append(
                        {
                            "id": tid,
                            "name": f"task_{tid}",
                            "status": "error",
                            "baseline_backend": "mps",
                            "passed": False,
                            "beats_baseline": False,
                            "compiled_kernel_dispatch_count": 0,
                            "dispatches_per_forward": 0,
                            "error": "Subprocess exited without writing report",
                        }
                    )
    return _build_report(all_results, manifest, args)


def _build_report(results: list[dict[str, Any]], manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    passed = [r for r in results if r["passed"]]
    beats = [r for r in passed if r["beats_baseline"]]
    speedups = [r["speedup"] for r in passed if r.get("speedup", 0) > 0]
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
    mps_unsupported = [r["id"] for r in results if r.get("baseline_backend") != "mps"]
    disp_per_fwd = [r["dispatches_per_forward"] for r in passed if r.get("dispatches_per_forward", 0) > 0]

    report: dict[str, Any] = {
        "kernelbench_level": 2,
        "kernelbench_commit_sha": manifest["commit_sha"],
        "mode": "baseline_only" if args.baseline_only else "full_eval",
        "seeds": args.seeds,
        "warmup": args.warmup,
        "trials": args.trials,
        "loadavg_1m": round(os.getloadavg()[0], 2),
        "summary": {
            "total_evaluated": len(results),
            "passed": len(passed),
            "failed": len([r for r in results if r["status"] not in ("passed", "baseline_measured", "missing")]),
            "missing": len([r for r in results if r["status"] == "missing"]),
            "beats_baseline": len(beats),
            "geomean_speedup": round(geomean, 4) if geomean is not None else None,
            "median_dispatches_per_forward": int(statistics.median(disp_per_fwd)) if disp_per_fwd else 0,
            "mps_unsupported_task_ids": mps_unsupported,
        },
        "tasks": results,
    }
    if args.roofline is not None and args.roofline.is_file():
        report["roofline"] = json.loads(args.roofline.read_text(encoding="utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen M1Triton KernelBench Level 2 evaluator.")
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "benchmarks/kernelbench_l2")
    parser.add_argument("--candidate-dir", type=Path, default=REPO_ROOT / "benchmarks/kernels_l2")
    parser.add_argument("--tasks", default="all", help="Task IDs: 'all', '1..20', or '1,9,12'")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--compare-inductor", action="store_true")
    parser.add_argument("--process-isolated", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--roofline", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.kb_dir / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = args.kb_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_by_id = {int(t["id"]): t for t in manifest["tasks"]}
    selected_ids = parse_task_selector(args.tasks, sorted(tasks_by_id.keys()))

    if args.process_isolated:
        report = _run_process_isolated(args, selected_ids, manifest)
    else:
        results: list[dict[str, Any]] = []
        for tid in selected_ids:
            res = evaluate_task(
                tasks_by_id[tid],
                args.kb_dir,
                args.candidate_dir,
                baseline_only=args.baseline_only,
                compare_inductor=args.compare_inductor,
                seeds=args.seeds,
                warmup=args.warmup,
                trials=args.trials,
                allow_missing=args.allow_missing,
            )
            results.append(res)
            if args.baseline_only:
                ind_s = f" ind={res['inductor_median_ms']}ms" if "inductor_median_ms" in res else ""
                print(
                    f"[L2-{tid:3d}] {res['name']:<52} backend={res['baseline_backend']:<26} "
                    f"base={res.get('baseline_median_ms', 'ERR')} ms{ind_s}",
                    flush=True,
                )
            else:
                sp = f"{res['speedup']:.2f}x" if "speedup" in res else "-"
                print(
                    f"[L2-{tid:3d}] {res['name']:<52} {res['status']:<18} "
                    f"base={res.get('baseline_median_ms', '-')}ms cand={res.get('candidate_median_ms', '-')}ms "
                    f"speedup={sp} dpf={res['dispatches_per_forward']}",
                    flush=True,
                )
        report = _build_report(results, manifest, args)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nSUMMARY:", json.dumps(report["summary"]))
    if not args.baseline_only and report["summary"]["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
