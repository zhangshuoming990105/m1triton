"""KernelBench L2 Task #85: Conv2d + GroupNorm + Scale + MaxPool2d + Clamp via fused M1Triton kernels."""

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
    conv_nd_epilogue,
    group_norm_scale_maxpool2d_clamp,
    m1_rt,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self._init_group_norm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = as_tuple(maxpool_kernel_size, 2)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.group_norm.num_groups,
            tuple(ref.scale.shape),
            ref.maxpool.kernel_size,
            ref.clamp_min,
            ref.clamp_max,
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
        with torch.no_grad():
            self.scale.copy_(ref.scale.detach().to(self.scale.device))

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
                sync_mps=False,
                wait=False,
            )
            y = group_norm_scale_maxpool2d_clamp(
                y,
                self.gn_weight,
                self.gn_bias,
                self.scale,
                num_groups=self.gn_num_groups,
                eps=self.gn_eps,
                pool_kernel_size=self.maxpool_kernel_size,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
                sync_mps=False,
                wait=False,
            )
        return y
