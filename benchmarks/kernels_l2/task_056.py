"""KernelBench L2 #056: Matmul + Sigmoid + Sum (FusedTail RowSum + Stage 2)."""

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
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((hidden_size, input_size), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        h_s, in_s = ref.linear.weight.shape
        m = cls(in_s, h_s)
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
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.bias,
                epilogue_steps=[("sigmoid",)],
                tail={"kind": "row_sum"},
                out=partials,
                sync_mps=False,
            )
            C.finalize_row_sum(partials, keepdim=True, out=out, sync_mps=False)
        return out
