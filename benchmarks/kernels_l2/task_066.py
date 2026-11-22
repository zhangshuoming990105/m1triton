"""KernelBench L2 #066: Matmul + Dropout(eval identity) + Softmax (2 fused dispatches)."""

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
    def __init__(self, in_features: int, out_features: int, dropout_p: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.dropout_p = float(dropout_p)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.matmul.weight.shape
        m = cls(in_f, out_f, float(ref.dropout.p))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.matmul.weight)
        C.copy_tensor_(self.bias, ref.matmul.bias)
        self.dropout_p = float(ref.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        logits = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(x_c, self.weight, self.bias, out=logits, sync_mps=False)
            C.softmax_rows(logits, out=out, sync_mps=False)
        return out
