"""KernelBench L2 Task #32: Conv2d + Scaling + Min(dim=1, keepdim=True) via fused M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, axis_min, conv_nd_epilogue, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.scale_factor = float(scale_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(ref.conv.in_channels, ref.conv.out_channels, ref.conv.kernel_size, ref.scale_factor)
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with m1_rt.command_batch(wait=True, sync_mps=True):
            y = conv_nd_epilogue(
                x,
                self.conv_weight,
                self.conv_bias,
                ndim=2,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                epilogue_steps=(("mul_const", self.scale_factor),),
                sync_mps=False,
                wait=False,
            )
            y = axis_min(y, dim=1, keepdim=True)
        return y
