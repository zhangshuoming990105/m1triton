"""KernelBench L2 #009: Matmul + Subtract + Multiply + ReLU (1 fused SGMM dispatch)."""

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
    def __init__(self, in_features: int, out_features: int, subtract_value: float, multiply_value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.linear.weight.shape
        m = cls(in_f, out_f, float(ref.subtract_value), float(ref.multiply_value))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.linear.weight)
        C.copy_tensor_(self.bias, ref.linear.bias)
        self.subtract_value = float(ref.subtract_value)
        self.multiply_value = float(ref.multiply_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[
                ("sub_const", self.subtract_value),
                ("mul_const", self.multiply_value),
                ("relu",),
            ],
        )
        return out
