"""KernelBench L2 #068: Matmul + Min(x, c) - c == min(x - c, 0) (1 fused SGMM dispatch)."""

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
    def __init__(self, in_features: int, out_features: int, constant: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.constant = nn.Parameter(torch.tensor(float(constant), dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.linear.weight.shape
        m = cls(in_f, out_f, float(ref.constant.detach().item()))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.linear.weight)
        C.copy_tensor_(self.bias, ref.linear.bias)
        C.copy_tensor_(self.constant, ref.constant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_val = float(self.constant.item())
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[("sub_scalar", "s0"), ("min_const", 0.0)],
            extra_scalars=[c_val],
        )
        return out
