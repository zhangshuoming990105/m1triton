"""KernelBench L2 #080: Gemm + Max(dim=max_dim) + SubtractMean(dim=1) + GELU."""

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
    def __init__(self, in_features: int, out_features: int, max_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.max_dim = int(max_dim)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(in_f, out_f, int(ref.max_dim))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.bias, ref.gemm.bias)
        self.max_dim = int(ref.max_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        if self.max_dim == 1 or self.max_dim == -1:
            _, bn, _, _, _, _ = C._select_fused_sgmm_schedule(
                m, n, int(x_c.shape[1]), trans_b=True, tail_kind="row_max"
            )
            grid_n = (n + bn - 1) // bn
            partials = torch.empty((m, grid_n), dtype=x_c.dtype, device=x_c.device)
            out = torch.empty((m, 1), dtype=x_c.dtype, device=x_c.device)
            with C.m1_runtime.command_batch(wait=True, sync_mps=True):
                C.fused_linear_epilogue(
                    x_c,
                    self.weight,
                    self.bias,
                    tail={"kind": "row_max"},
                    out=partials,
                    sync_mps=False,
                    wait=False,
                )
                C.finalize_row_max(partials, keepdim=True, post_op=1, out=out, sync_mps=False)
            return out
        col_max = torch.empty((1, n), dtype=x_c.dtype, device=x_c.device)
        block_n = 256
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            lin, _ = C.fused_linear_epilogue(x_c, self.weight, self.bias, sync_mps=False, wait=False)
            C.axis_max_dim0_kernel[((n + block_n - 1) // block_n,)](
                lin, col_max, m, n, BLOCK_N=block_n, sync_mps=False, wait=False
            )
            out = C.row_mean_sub_gelu(col_max, sync_mps=False, wait=False)
        return out
