"""KernelBench L2 Task #48: Conv3d + Scaling + Tanh + Multiply + Sigmoid via 1 fused M1Triton kernel."""

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
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            1.0,
            tuple(ref.bias.shape),
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
            self.scaling_factor.copy_(ref.scaling_factor.detach().to(self.scaling_factor.device))
            self.bias.copy_(ref.bias.detach().to(self.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_nd_epilogue(
            x,
            self.conv_weight,
            self.conv_bias,
            ndim=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            epilogue_steps=(
                ("mul_bias", "ep0_ptr"),
                ("tanh",),
                ("mul_bias", "ep1_ptr"),
                ("sigmoid",),
            ),
            extra_bufs=(self.scaling_factor.view(-1), self.bias.view(-1)),
        )
