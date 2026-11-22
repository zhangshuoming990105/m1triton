"""KernelBench L2 #081: Gemm + Swish + Divide(2) + Clamp(-1,1) + Tanh + Clamp(-1,1) (1 dispatch)."""

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
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(in_f, out_f, bias=ref.gemm.bias is not None)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        if self.bias is not None and ref.gemm.bias is not None:
            C.copy_tensor_(self.bias, ref.gemm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = C.fused_linear_epilogue(
            x,
            self.weight,
            self.bias,
            epilogue_steps=[
                ("swish",),
                ("div_const", 2.0),
                ("clamp", -1.0, 1.0),
                ("tanh",),
                ("clamp", -1.0, 1.0),
            ],
        )
        return out
