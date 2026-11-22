"""KernelBench L2 Task #54: Conv2d + Multiply + LeakyReLU + GELU via fused M1Triton EpilogueDAG."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, conv_nd_epilogue


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.01

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            tuple(ref.multiplier.shape),
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.negative_slope = float(getattr(ref.leaky_relu, "negative_slope", 0.01))
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        with torch.no_grad():
            self.multiplier.copy_(ref.multiplier.detach().to(self.multiplier.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_nd_epilogue(
            x,
            self.conv_weight,
            self.conv_bias,
            ndim=2,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            epilogue_steps=(
                ("mul_bias", "ep0_ptr"),
                ("leaky_relu", self.negative_slope),
                ("gelu",),
            ),
            extra_bufs=(self.multiplier.view(-1),),
        )
