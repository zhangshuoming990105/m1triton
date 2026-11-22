"""KernelBench L2 Task #31: Conv2d + MinConst + BiasAdd + Multiply via fused M1Triton EpilogueDAG."""

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

    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.constant_value,
            tuple(ref.bias.shape),
            ref.scaling_factor,
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
                ("min_const", self.constant_value),
                ("add_bias", "ep0_ptr"),
                ("mul_const", self.scaling_factor),
            ),
            extra_bufs=(self.bias.view(-1),),
        )
