"""KernelBench L2 #018: Matmul + Sum + (Max/Mean/LogSumExp/LogSumExp on dim=1 size-1)."""

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
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.linear.weight.shape
        m = cls(in_f, out_f)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.linear.weight)
        C.copy_tensor_(self.bias, ref.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        _, bn, _, _, _, _ = C._select_fused_sgmm_schedule(m, n, int(x_c.shape[1]), trans_b=True, tail_kind="row_sum")
        grid_n = (n + bn - 1) // bn
        partials = torch.empty((m, grid_n), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, 1), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=False, sync_mps=True):
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.bias,
                tail={"kind": "row_sum"},
                out=partials,
                sync_mps=False,
            )
            C.finalize_row_sum(partials, keepdim=True, out=out, sync_mps=False)
        return out
