"""KernelBench L2 Task #24: Conv3d + Min(dim=dim) + Softmax(dim=1) via M1Triton Metal kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, axis_min, channel_softmax, conv_nd_epilogue, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.dim = int(dim)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(ref.conv.in_channels, ref.conv.out_channels, ref.conv.kernel_size, ref.dim)
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
                ndim=3,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                sync_mps=False,
                wait=False,
            )
            y = axis_min(y, dim=self.dim, keepdim=False)
            y = channel_softmax(y, sync_mps=False, wait=False)
        return y
