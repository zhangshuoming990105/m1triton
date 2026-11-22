"""KernelBench L2 Task #6: Conv3d + Softmax(dim=1) + MaxPool3d + MaxPool3d via M1Triton Metal kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import (
    ConvStdL2Base,
    as_tuple,
    channel_softmax,
    conv_nd_epilogue,
    m1_rt,
    pool_nd,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        pk = as_tuple(pool_kernel_size, 3)
        self.pool1_kernel_size = pk
        self.pool2_kernel_size = pk

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        p1 = as_tuple(ref.pool1.kernel_size, 3)
        p2 = as_tuple(ref.pool2.kernel_size, 3)
        model = cls(ref.conv.in_channels, ref.conv.out_channels, ref.conv.kernel_size, p1)
        model.pool1_kernel_size = p1
        model.pool2_kernel_size = p2
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        comb_k = tuple(self.pool1_kernel_size[i] * self.pool2_kernel_size[i] for i in range(3))
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
            y = channel_softmax(y, sync_mps=False, wait=False)
            y = pool_nd(y, kind="max", ndim=3, kernel_size=comb_k, stride=comb_k)
        return y
