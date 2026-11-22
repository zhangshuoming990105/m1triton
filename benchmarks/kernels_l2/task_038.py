"""KernelBench L2 Task 038: AvgPool3d + ConvTranspose3d + Clamp + SpatialSoftmax + Scale."""

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
    launch_conv_transpose_fused,
    pool_nd,
    spatial_softmax_mul_scale,
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
        output_padding: int,
        pool_kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ) -> None:
        super().__init__()
        self.pool_k = as_tuple(pool_kernel_size, 3)
        self.pool_s = self.pool_k
        self.pool_p = (0, 0, 0)
        self._init_conv_transpose(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        ap = ref.avg_pool
        m.pool_k = as_tuple(ap.kernel_size, 3)
        m.pool_s = as_tuple(ap.stride if ap.stride is not None else ap.kernel_size, 3)
        m.pool_p = as_tuple(ap.padding, 3)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.clamp_min = float(ref.clamp_min)
        m.clamp_max = float(ref.clamp_max)
        m.scale = nn.Parameter(torch.empty_like(ref.scale.detach()))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.clamp_min = float(ref.clamp_min)
        self.clamp_max = float(ref.clamp_max)
        with torch.no_grad():
            self.scale.copy_(ref.scale.detach().to(self.scale.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        sc = self.scale.detach().contiguous().view(-1)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = pool_nd(
                x_c, kind="avg", ndim=3, kernel_size=self.pool_k, stride=self.pool_s, padding=self.pool_p
            )
            y = launch_conv_transpose_fused(
                y, prep, epilogue=(("clamp", self.clamp_min, self.clamp_max),)
            )
            out = spatial_softmax_mul_scale(y, sc)
        return out
