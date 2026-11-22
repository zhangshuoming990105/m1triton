"""KernelBench L2 Task 050: ConvTranspose3d + Scale1 + AvgPool3d + BiasAdd + Scale2 (2-dispatch fused)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import m1triton
from _l2_conv_transpose_common import (
    ConvTransposeL2Base,
    as_tuple,
    avgpool3d_bias_scale,
    launch_conv_transpose_fused,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale1: float,
        scale2: float,
        bias_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.scale1 = nn.Parameter(torch.tensor(float(scale1)))
        self.extra_bias = nn.Parameter(torch.empty(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(float(scale2)))
        self.pool_k = (2, 2, 2)
        self.pool_s = (2, 2, 2)
        self.pool_p = (0, 0, 0)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.scale1 = nn.Parameter(torch.empty_like(ref.scale1.detach()))
        m.extra_bias = nn.Parameter(torch.empty_like(ref.bias.detach()))
        m.scale2 = nn.Parameter(torch.empty_like(ref.scale2.detach()))
        ap = ref.avg_pool
        m.pool_k = as_tuple(ap.kernel_size, 3)
        m.pool_s = as_tuple(ap.stride if ap.stride is not None else ap.kernel_size, 3)
        m.pool_p = as_tuple(ap.padding, 3)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        with torch.no_grad():
            self.scale1.copy_(ref.scale1.detach().to(self.scale1.device))
            self.extra_bias.copy_(ref.bias.detach().to(self.extra_bias.device))
            self.scale2.copy_(ref.scale2.detach().to(self.scale2.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        s1 = float(self.scale1.detach().item())
        s2 = float(self.scale2.detach().item())
        eb = self.extra_bias.detach().contiguous().view(-1)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(
                x_c, prep, epilogue=(("mul_scalar", "es0"),), es0=s1
            )
            out = avgpool3d_bias_scale(
                y, eb, s2, k=self.pool_k, s=self.pool_s, p=self.pool_p
            )
        return out
