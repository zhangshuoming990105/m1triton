"""KernelBench L2 Task #8: Conv3d + Divide + MaxPool3d + GlobalAvgPool3d + BiasAdd + Sum via M1Triton kernels."""

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
    axis_sum,
    conv_nd_epilogue,
    m1_rt,
    maxpool3d_gap_bias,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.divisor = float(divisor)
        self.pool_size = as_tuple(pool_size, 3)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = int(sum_dim)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.divisor,
            ref.max_pool.kernel_size,
            tuple(ref.bias.shape),
            ref.sum_dim,
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        with torch.no_grad():
            self.bias.copy_(ref.bias.detach().to(self.bias.device))

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
                epilogue_steps=(("div_const", self.divisor),),
                sync_mps=False,
                wait=False,
            )
            y = maxpool3d_gap_bias(y, self.pool_size, self.bias, sync_mps=False, wait=False)
            y = axis_sum(y, dim=self.sum_dim, keepdim=False)
        return y
