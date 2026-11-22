"""KernelBench L2 #028: Linear + InstanceNorm(row) + ResidualAdd + Multiply."""

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
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.eps = float(eps)
        self.momentum = float(momentum)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.bmm.weight.shape
        m = cls(in_f, out_f, eps=float(ref.instance_norm.eps), momentum=float(ref.instance_norm.momentum))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.bmm.weight)
        C.copy_tensor_(self.bias, ref.bmm.bias)
        self.eps = float(ref.instance_norm.eps)
        self.momentum = float(ref.instance_norm.momentum)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        y_c = y if y.is_contiguous() else y.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        lin_out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        norm_buf = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=False, sync_mps=True):
            C.fused_linear_epilogue(x_c, self.weight, self.bias, out=lin_out, sync_mps=False)
            C.row_instancenorm_add_mul(
                lin_out, y_c, eps=self.eps, norm_buf=norm_buf, out=out, sync_mps=False
            )
        return out
