"""KernelBench L2 Task 003: ConvTranspose3d + Sum + LayerNorm + AvgPool3d + GELU (2-dispatch fused)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import m1triton
from _l2_conv_transpose_common import (
    ConvTransposeL2Base,
    as_tuple,
    launch_conv_transpose_fused,
    layernorm_avgpool3d_gelu,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        padding: int | Sequence[int],
        output_padding: int | Sequence[int],
        sum_weight: float,
        norm_shape: Sequence[int],
        pool_kernel_size: int | Sequence[int],
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.sum_weight = nn.Parameter(torch.tensor(float(sum_weight)))
        self._init_ln_params(norm_shape, eps=1e-5)
        self.pool_k = as_tuple(pool_kernel_size, 3)
        self.pool_s = self.pool_k
        self.pool_p = (0, 0, 0)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.sum_weight = nn.Parameter(torch.empty_like(ref.sum_weight.detach()))
        m._init_ln_params(tuple(int(v) for v in ref.norm.normalized_shape), eps=float(ref.norm.eps))
        m.pool_k = as_tuple(ref.avg_pool.kernel_size, 3)
        m.pool_s = as_tuple(ref.avg_pool.stride if ref.avg_pool.stride is not None else ref.avg_pool.kernel_size, 3)
        m.pool_p = as_tuple(ref.avg_pool.padding, 3)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self._sync_ln_from_ref(ref.norm)
        with torch.no_grad():
            self.sum_weight.copy_(ref.sum_weight.detach().to(self.sum_weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        sw = float(self.sum_weight.detach().item())
        ln_w = self.ln_weight.detach().contiguous().view(-1)
        ln_b = self.ln_bias.detach().contiguous().view(-1)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=(("add_scalar", "es0"),),
                es0=sw,
            )
            out = layernorm_avgpool3d_gelu(
                y,
                ln_w,
                ln_b,
                eps=self.ln_eps,
                pool_k=self.pool_k,
                pool_s=self.pool_s,
                pool_p=self.pool_p,
            )
        return out
