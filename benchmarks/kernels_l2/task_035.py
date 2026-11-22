"""KernelBench L2 Task #35: Conv2d + Subtract + HardSwish + MaxPool2d + Mish via fused M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, as_tuple, conv_nd_epilogue, m1_rt, maxpool2d_mish


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = as_tuple(pool_kernel_size, 2)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.subtract_value,
            ref.pool.kernel_size,
        )
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
                epilogue_steps=(
                    ("sub_const", self.subtract_value),
                    ("hardswish",),
                ),
                sync_mps=False,
                wait=False,
            )
            y = maxpool2d_mish(y, self.pool_kernel_size, sync_mps=False, wait=False)
        return y
