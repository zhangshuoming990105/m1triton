"""KernelBench L2 #086: Matmul + Divide + GELU (1 fused SGMM dispatch)."""

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
    def __init__(self, input_size: int, output_size: int, divisor: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((output_size, input_size), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((output_size,), dtype=torch.float32))
        self.divisor = float(divisor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_s, in_s = ref.linear.weight.shape
        m = cls(in_s, out_s, float(ref.divisor))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.linear.weight)
        C.copy_tensor_(self.bias, ref.linear.bias)
        self.divisor = float(ref.divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[("div_const", self.divisor), ("gelu",)],
        )
        return out
