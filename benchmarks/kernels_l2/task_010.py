"""KernelBench L2 Task 010: ConvTranspose2d + MaxPool2d + Hardtanh + Mean + Tanh (2-dispatch fused)."""

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
    conv_transpose_nd_epilogue,
    maxpool2d_hardtanh_mean_tanh,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        maxpool_kernel_size: int,
        maxpool_stride: int,
        hardtanh_min: float,
        hardtanh_max: float,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.pool_k = as_tuple(maxpool_kernel_size, 2)
        self.pool_s = as_tuple(maxpool_stride, 2)
        self.pool_p = (0, 0)
        self.pool_d = (1, 1)
        self.hmin = float(hardtanh_min)
        self.hmax = float(hardtanh_max)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        mp = ref.maxpool
        m.pool_k = as_tuple(mp.kernel_size, 2)
        m.pool_s = as_tuple(mp.stride if mp.stride is not None else mp.kernel_size, 2)
        m.pool_p = as_tuple(mp.padding, 2)
        m.pool_d = as_tuple(mp.dilation, 2)
        m.hmin = float(ref.hardtanh.min_val)
        m.hmax = float(ref.hardtanh.max_val)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.hmin = float(ref.hardtanh.min_val)
        self.hmax = float(ref.hardtanh.max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = conv_transpose_nd_epilogue(
            x,
            self.conv_weight.detach(),
            None if self.conv_bias is None else self.conv_bias.detach(),
            ndim=2,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            allow_winograd=True,
        )
        return maxpool2d_hardtanh_mean_tanh(
            y,
            k=self.pool_k,
            s=self.pool_s,
            p=self.pool_p,
            d=self.pool_d,
            hmin=self.hmin,
            hmax=self.hmax,
        )
