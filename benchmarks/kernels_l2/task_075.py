"""KernelBench L2 #075: Gemm + GroupNorm + Min(dim=1) + Broadcast BiasAdd (3 dispatches)."""

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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        bias_shape: tuple[int, ...],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.linear_bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.gn_weight = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(bias_shape, dtype=torch.float32))
        self.num_groups = int(num_groups)
        self.gn_eps = float(eps)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(
            in_f,
            out_f,
            int(ref.group_norm.num_groups),
            tuple(ref.bias.shape),
            eps=float(ref.group_norm.eps),
        )
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.linear_bias, ref.gemm.bias)
        C.copy_tensor_(self.gn_weight, ref.group_norm.weight)
        C.copy_tensor_(self.gn_bias, ref.group_norm.bias)
        C.copy_tensor_(self.bias, ref.bias)
        self.num_groups = int(ref.group_norm.num_groups)
        self.gn_eps = float(ref.group_norm.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        c_bias = int(self.bias.shape[1])
        hidden = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        row_min = torch.empty((m, 1), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((1, c_bias, m, 1), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=False, sync_mps=True):
            C.fused_linear_epilogue(x_c, self.weight, self.linear_bias, out=hidden, sync_mps=False)
            C.groupnorm_2d_row_min(
                hidden,
                self.gn_weight,
                self.gn_bias,
                num_groups=self.num_groups,
                eps=self.gn_eps,
                out=row_min,
                sync_mps=False,
            )
            C.broadcast_outer_add_4d(row_min, self.bias, out=out, sync_mps=False)
        return out
