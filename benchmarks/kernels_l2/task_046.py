"""KernelBench L2 Task #46: Conv2d + Sub + Tanh + Sub + AvgPool2d via fused M1Triton Metal kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, as_tuple, conv_nd_epilogue, m1_rt, pool_nd


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.pool_kernel_size = as_tuple(kernel_size_pool, 2)
        self.pool_stride = as_tuple(kernel_size_pool, 2)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.subtract1_value,
            ref.subtract2_value,
            ref.avgpool.kernel_size,
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.pool_stride = as_tuple(ref.avgpool.stride, 2)
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
                epilogue_steps=(
                    ("sub_const", self.subtract1_value),
                    ("tanh",),
                    ("sub_const", self.subtract2_value),
                ),
                sync_mps=False,
                wait=False,
            )
            y = pool_nd(y, kind="avg", ndim=2, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        return y
