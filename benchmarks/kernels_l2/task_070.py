"""KernelBench L2 #070: Gemm + Sigmoid + Scaling + ResidualAdd (1 fused SGMM dispatch)."""

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
    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((hidden_size, input_size), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        h_s, in_s = ref.gemm.weight.shape
        m = cls(in_s, h_s, float(ref.scaling_factor))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.bias, ref.gemm.bias)
        self.scaling_factor = float(ref.scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[
                ("save_reg", 0),
                ("sigmoid",),
                ("mul_const", self.scaling_factor),
                ("add_saved", 0),
            ],
        )
        return out
