"""KernelBench L2 #051: Gemm + Subtract + Mean + LogSumExp(1) + GELU + ResidualAdd."""

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
        self.subtract = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        has_bias = ref.gemm.bias is not None
        m = cls(in_f, out_f, bias=has_bias)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        if self.bias is not None and ref.gemm.bias is not None:
            C.copy_tensor_(self.bias, ref.gemm.bias)
        C.copy_tensor_(self.subtract, ref.subtract)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m, in_f = int(x_c.shape[0]), int(x_c.shape[1])
        n = int(self.weight.shape[0])
        _, bn, _, _, _, _ = C._select_fused_sgmm_schedule(m, n, in_f, trans_b=True, tail_kind="row_mean")
        grid_n = (n + bn - 1) // bn
        partials = torch.empty((m, grid_n), dtype=x_c.dtype, device=x_c.device)
        row_gelu = torch.empty((m, 1), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, in_f), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.bias,
                epilogue_steps=[("sub_bias", "buf0")],
                extra_bufs=[self.subtract],
                tail={"kind": "row_mean"},
                out=partials,
                sync_mps=False,
            )
            C.finalize_row_sum(partials, keepdim=True, post_op=1, out=row_gelu, sync_mps=False)
            C.broadcast_row_add_2d(row_gelu, x_c, out=out, sync_mps=False)
        return out
