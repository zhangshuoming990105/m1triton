"""Positive and negative anti-cheat vetting tests for frozen L2 and L3 evaluators."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_l2_eval_rejects_static_and_dynamic_cheat(tmp_path: Path) -> None:
    ev = _load(REPO_ROOT / "evals/eval_kernelbench_l2.py", "_test_eval_l2")

    cheat_file = tmp_path / "task_001.py"
    cheat_file.write_text(
        "import torch, torch.nn as nn\n"
        "class ModelNew(nn.Module):\n"
        "    def forward(self, x): return torch.relu(x)\n",
        encoding="utf-8",
    )
    violations = ev.static_ast_audit(cheat_file)
    assert any("torch.relu" in v for v in violations)

    x = torch.randn(64, device="cpu")
    with pytest.raises(ev.AntiCheatViolation):
        with ev.TorchComputeGuard():
            _ = x + 1.0


def test_l2_compare_outputs_chunked_and_nan_strict() -> None:
    ev = _load(REPO_ROOT / "evals/eval_kernelbench_l2.py", "_test_eval_l2_cmp")
    a = torch.randn(1024, dtype=torch.float32)
    b = a + 5e-4
    ok, info = ev.compare_outputs(a, b)
    assert ok and info["rtol"] == 1e-3

    c = a.clone()
    c[0] = float("nan")
    ok_nan, info_nan = ev.compare_outputs(c, a)
    assert not ok_nan and "NaN" in info_nan["reason"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_l3_eval_positive_and_negative_controls(tmp_path: Path) -> None:
    ev = _load(REPO_ROOT / "evals/eval_triton_l3.py", "_test_eval_l3")
    specs = tmp_path / "specs"
    kernels = tmp_path / "kernels"
    specs.mkdir()
    kernels.mkdir()

    (specs / "l3_01_scale_add.py").write_text(
        "import torch\n"
        "TASK_NAME = 'scale_add_bwd'\n"
        "SUPPORTS_BACKWARD = True\n"
        "def get_inputs(seed, device):\n"
        "    torch.manual_seed(seed)\n"
        "    x = torch.randn(1024, device=device, dtype=torch.float32, requires_grad=True)\n"
        "    return [x], {}\n"
        "def reference_forward(x):\n"
        "    return x * 2.0 + 1.0\n",
        encoding="utf-8",
    )

    # Negative control 1: static AST cheat
    (kernels / "l3_01_scale_add.py").write_text(
        "import torch\n"
        "def candidate_forward(x): return torch.relu(x)\n",
        encoding="utf-8",
    )
    bad_ast = ev.evaluate_l3_task(1, "scale_add", specs, kernels, seeds=1, warmup=1, trials=2, allow_missing=False)
    assert bad_ast["status"] == "anticheat_failed"

    # Negative control 2: dynamic ATen cheat bypassing static AST
    (kernels / "l3_01_scale_add.py").write_text(
        "def candidate_forward(x): return x * 2.0 + 1.0\n",
        encoding="utf-8",
    )
    bad_dyn = ev.evaluate_l3_task(1, "scale_add", specs, kernels, seeds=1, warmup=1, trials=2, allow_missing=False)
    assert bad_dyn["status"] == "anticheat_failed"

    # Positive control: real M1Triton kernel for BOTH forward and backward
    (kernels / "l3_01_scale_add.py").write_text(
        "import torch\n"
        "import m1triton\n"
        "import m1triton.language as tl\n"
        "@m1triton.jit\n"
        "def _fwd(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
        "    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    m = offs < n\n"
        "    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0 + 1.0, mask=m)\n"
        "@m1triton.jit\n"
        "def _bwd(g_ptr, dx_ptr, n, BLOCK: tl.constexpr):\n"
        "    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    m = offs < n\n"
        "    tl.store(dx_ptr + offs, tl.load(g_ptr + offs, mask=m) * 2.0, mask=m)\n"
        "class _Fn(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, x):\n"
        "        o = torch.empty_like(x)\n"
        "        _fwd[(m1triton.cdiv(x.numel(), 256),)](x, o, x.numel(), BLOCK=256)\n"
        "        return o\n"
        "    @staticmethod\n"
        "    def backward(ctx, g):\n"
        "        gc = g.contiguous()\n"
        "        dx = torch.empty_like(gc)\n"
        "        _bwd[(m1triton.cdiv(gc.numel(), 256),)](gc, dx, gc.numel(), BLOCK=256)\n"
        "        return dx\n"
        "def candidate_forward(x): return _Fn.apply(x)\n",
        encoding="utf-8",
    )
    good = ev.evaluate_l3_task(1, "scale_add", specs, kernels, seeds=2, warmup=1, trials=3, allow_missing=False)
    assert good["passed"] is True, good
    assert good["compiled_kernel_dispatch_count"] >= 4
