"""Vetting tests for evals/eval_kernelbench.py positive and negative controls."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "evals" / "eval_kernelbench.py"


def _run_eval(kb_dir: Path, cand_dir: Path, out_path: Path) -> tuple[int, dict]:
    proc = subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--kb-dir",
            str(kb_dir),
            "--candidate-dir",
            str(cand_dir),
            "--tasks",
            "1",
            "--seeds",
            "3",
            "--warmup",
            "2",
            "--trials",
            "5",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    return proc.returncode, report


def _make_mini_kb(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "mini_kb"
    kb_dir.mkdir()
    task_file = kb_dir / "1_Mini_ReLU.py"
    task_file.write_text(
        """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024)]

def get_init_inputs():
    return []
""",
        encoding="utf-8",
    )
    manifest = {
        "commit_sha": "test_mini_kb",
        "task_count": 1,
        "tasks": [{"id": 1, "filename": "1_Mini_ReLU.py", "name": "Mini_ReLU"}],
    }
    (kb_dir / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return kb_dir


def test_eval_rejects_static_torch_cheat(tmp_path: Path):
    kb_dir = _make_mini_kb(tmp_path)
    cand_dir = tmp_path / "cand_ast_cheat"
    cand_dir.mkdir()
    (cand_dir / "task_001.py").write_text(
        """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
""",
        encoding="utf-8",
    )
    rc, report = _run_eval(kb_dir, cand_dir, tmp_path / "ast.json")
    assert rc != 0
    assert report["tasks"][0]["status"] == "anticheat_failed"
    assert "ANTICHEAT-AST" in report["tasks"][0]["error"]


def test_eval_rejects_dynamic_torch_dispatch_cheat(tmp_path: Path):
    kb_dir = _make_mini_kb(tmp_path)
    cand_dir = tmp_path / "cand_dyn_cheat"
    cand_dir.mkdir()
    # Deliberately obfuscate torch call via getattr to verify TorchComputeGuard catches it.
    (cand_dir / "task_001.py").write_text(
        """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        fn = getattr(torch, "re" + "lu")
        return fn(x)
""",
        encoding="utf-8",
    )
    rc, report = _run_eval(kb_dir, cand_dir, tmp_path / "dyn.json")
    assert rc != 0
    assert report["tasks"][0]["status"] == "anticheat_failed"
    assert "ANTICHEAT-TORCH-DISPATCH" in report["tasks"][0]["error"]


def test_eval_rejects_zero_dispatch_noop(tmp_path: Path):
    kb_dir = _make_mini_kb(tmp_path)
    cand_dir = tmp_path / "cand_zero_disp"
    cand_dir.mkdir()
    (cand_dir / "task_001.py").write_text(
        """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)
""",
        encoding="utf-8",
    )
    rc, report = _run_eval(kb_dir, cand_dir, tmp_path / "zero.json")
    assert rc != 0
    assert report["tasks"][0]["status"] == "anticheat_failed"
    assert "ANTICHEAT-ZERO-DISPATCH" in report["tasks"][0]["error"]


def test_eval_rejects_wrong_metal_result_and_accepts_valid_metal_kernel(tmp_path: Path):
    kb_dir = _make_mini_kb(tmp_path)

    # Degraded control: dispatches a real Metal kernel that writes wrong values.
    bad_dir = tmp_path / "cand_bad_math"
    bad_dir.mkdir()
    (bad_dir / "task_001.py").write_text(
        """
import torch
import torch.nn as nn
import m1triton.runtime as rt

_MSL = '''
#include <metal_stdlib>
using namespace metal;
kernel void bad_relu(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = x[gid] + 1.0f;
}
'''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kid = rt.compile_kernel(_MSL, "bad_relu")

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        tg = 256
        grid = (n + tg - 1) // tg
        rt.launch_kernel(
            self._kid,
            [rt.make_buffer_arg(x, 0), rt.make_buffer_arg(out, 1, writable=True)],
            [rt.make_scalar_arg(n, 2, "u32")],
            grid=(grid, 1, 1),
            threads_per_threadgroup=(tg, 1, 1),
        )
        return out
""",
        encoding="utf-8",
    )
    rc_bad, rep_bad = _run_eval(kb_dir, bad_dir, tmp_path / "bad.json")
    assert rc_bad != 0
    assert rep_bad["tasks"][0]["status"] == "correctness_failed"
    assert rep_bad["tasks"][0]["compiled_kernel_dispatch_count"] >= 1

    # Positive control: dispatches a real Metal ReLU kernel over zero-copy MPS buffers.
    good_dir = tmp_path / "cand_good"
    good_dir.mkdir()
    (good_dir / "task_001.py").write_text(
        """
import torch
import torch.nn as nn
import m1triton.runtime as rt

_MSL = '''
#include <metal_stdlib>
using namespace metal;
kernel void good_relu(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] = max(x[gid], 0.0f);
}
'''

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kid = rt.compile_kernel(_MSL, "good_relu")

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        tg = 256
        grid = (n + tg - 1) // tg
        rt.launch_kernel(
            self._kid,
            [rt.make_buffer_arg(x, 0), rt.make_buffer_arg(out, 1, writable=True)],
            [rt.make_scalar_arg(n, 2, "u32")],
            grid=(grid, 1, 1),
            threads_per_threadgroup=(tg, 1, 1),
        )
        return out
""",
        encoding="utf-8",
    )
    rc_good, rep_good = _run_eval(kb_dir, good_dir, tmp_path / "good.json")
    assert rc_good == 0
    assert rep_good["tasks"][0]["status"] == "passed"
    assert rep_good["tasks"][0]["passed"] is True
    assert rep_good["tasks"][0]["compiled_kernel_dispatch_count"] == 3
    assert rep_good["tasks"][0]["seed_reports"][0]["max_abs_err"] == 0.0
