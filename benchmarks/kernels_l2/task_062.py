"""KernelBench L2 #062: Matmul + GroupNorm + LeakyReLU + SelfSum (2 fused dispatches)."""

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
        input_size: int,
        hidden_size: int,
        num_groups: int,
        eps: float = 1e-5,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((hidden_size, input_size), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.gn_weight = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.gn_bias = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.num_groups = int(num_groups)
        self.gn_eps = float(eps)
        self.negative_slope = float(negative_slope)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        h_s, in_s = ref.fc.weight.shape
        m = cls(
            in_s,
            h_s,
            int(ref.gn.num_groups),
            eps=float(ref.gn.eps),
            negative_slope=float(ref.leaky_relu.negative_slope),
        )
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.fc.weight)
        C.copy_tensor_(self.bias, ref.fc.bias)
        C.copy_tensor_(self.gn_weight, ref.gn.weight)
        C.copy_tensor_(self.gn_bias, ref.gn.bias)
        self.num_groups = int(ref.gn.num_groups)
        self.gn_eps = float(ref.gn.eps)
        self.negative_slope = float(ref.leaky_relu.negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        hidden = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(x_c, self.weight, self.bias, out=hidden, sync_mps=False)
            C.groupnorm_2d_fused(
                hidden,
                self.gn_weight,
                self.gn_bias,
                num_groups=self.num_groups,
                eps=self.gn_eps,
                post_op=2,
                p0=self.negative_slope,
                out=out,
                sync_mps=False,
            )
        return out
