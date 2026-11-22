"""KernelBench L2 Task #27: Conv3d + HardSwish + GroupNorm + SpatialMean via fused M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, conv_nd_epilogue, group_norm_spatial_mean, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=bias)
        self._init_group_norm(num_groups, out_channels)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            num_groups=ref.group_norm.num_groups,
            bias=(ref.conv.bias is not None),
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        self._sync_group_norm_from(ref.group_norm)

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
                epilogue_steps=(("hardswish",),),
                sync_mps=False,
                wait=False,
            )
            y = group_norm_spatial_mean(
                y,
                self.gn_weight,
                self.gn_bias,
                num_groups=self.gn_num_groups,
                eps=self.gn_eps,
                reduce_channels=False,
                sync_mps=False,
                wait=False,
            )
        return y
