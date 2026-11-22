"""KernelBench L2 #012: Gemm + Multiply + LeakyReLU (1 fused SGMM dispatch)."""

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
    def __init__(self, in_features: int, out_features: int, multiplier: float, negative_slope: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(in_f, out_f, float(ref.multiplier), float(ref.leaky_relu.negative_slope))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.bias, ref.gemm.bias)
        self.multiplier = float(ref.multiplier)
        self.negative_slope = float(ref.leaky_relu.negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[
                ("mul_const", self.multiplier),
                ("leaky_relu", self.negative_slope),
            ],
        )
        return out
