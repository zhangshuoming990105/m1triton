"""KernelBench L2 Task #43: Conv3d + MaxPool3d + LogSumExp(dim=1) + ReLU via M1Triton Metal kernels."""

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
    channel_logsumexp_relu,
    conv_nd_epilogue,
    m1_rt,
    pool_nd,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.pool_kernel_size = (2, 2, 2)
        self.pool_stride = (2, 2, 2)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.conv.stride,
            ref.conv.padding,
        )
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.pool_kernel_size = as_tuple(ref.max_pool.kernel_size, 3)
        model.pool_stride = as_tuple(ref.max_pool.stride, 3)
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
            y = pool_nd(y, kind="max", ndim=3, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
            y = channel_logsumexp_relu(y, sync_mps=False, wait=False)
        return y
