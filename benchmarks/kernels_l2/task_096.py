"""KernelBench L2 Task 096: ConvTranspose3d + Multiply + MaxPool3d + GlobalAvgPool + Clamp (2-dispatch fused)."""

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
    maxpool3d_gap_clamp,
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
        scale: float,
        maxpool_kernel_size: int,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.scale = float(scale)
        self.pool_k = as_tuple(maxpool_kernel_size, 3)
        self.pool_s = self.pool_k
        self.pool_p = (0, 0, 0)
        self.pool_d = (1, 1, 1)
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.scale = float(ref.scale)
        mp = ref.maxpool
        m.pool_k = as_tuple(mp.kernel_size, 3)
        m.pool_s = as_tuple(mp.stride if mp.stride is not None else mp.kernel_size, 3)
        m.pool_p = as_tuple(mp.padding, 3)
        m.pool_d = as_tuple(mp.dilation, 3)
        m.clamp_min = float(ref.clamp_min)
        m.clamp_max = float(ref.clamp_max)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.scale = float(ref.scale)
        self.clamp_min = float(ref.clamp_min)
        self.clamp_max = float(ref.clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(
                x_c, prep, epilogue=(("mul_const", float(self.scale)),)
            )
            out = maxpool3d_gap_clamp(
                y,
                k=self.pool_k,
                s=self.pool_s,
                p=self.pool_p,
                d=self.pool_d,
                cmin=self.clamp_min,
                cmax=self.clamp_max,
            )
        return out
