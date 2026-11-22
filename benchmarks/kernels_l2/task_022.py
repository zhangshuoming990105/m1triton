"""KernelBench L2 #022: Matmul + Scale + SelfAdd + Clamp + LogSumExp + x * Mish(x)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
import _l2_gemm_common as C


class ModelNew(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_factor: float,
        clamp_min: float,
        clamp_max: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((hidden_size, input_size), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        hidden_s, in_s = ref.matmul.weight.shape
        m = cls(in_s, hidden_s, float(ref.scale_factor), float(ref.clamp_min), float(ref.clamp_max))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.matmul.weight)
        C.copy_tensor_(self.bias, ref.matmul.bias)
        self.scale_factor = float(ref.scale_factor)
        self.clamp_min = float(ref.clamp_min)
        self.clamp_max = float(ref.clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        _, bn, _, _, _, _ = C._select_fused_sgmm_schedule(
            m, n, int(x_c.shape[1]), trans_b=True, tail_kind="row_logsumexp"
        )
        grid_n = (n + bn - 1) // bn
        partials = torch.empty((m, grid_n, 2), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, 1), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.bias,
                epilogue_steps=[
                    ("mul_const", self.scale_factor),
                    ("mul_const", 2.0),
                    ("clamp", self.clamp_min, self.clamp_max),
                ],
                tail={"kind": "row_logsumexp"},
                out=partials,
                sync_mps=False,
            )
            C.finalize_row_lse(partials, keepdim=True, post_op=1, out=out, sync_mps=False)
        return out
