"""KernelBench L2 #076: Gemm(no bias) + BiasAdd + ReLU (1 fused SGMM dispatch)."""

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
    def __init__(self, in_features: int, out_features: int, bias_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(bias_shape, dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(in_f, out_f, tuple(ref.bias.shape))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.bias, ref.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[("relu",)],
        )
        return out
