"""KernelBench L2 Task #90: Conv3d + LeakyReLU + BiasAdd + Clamp + GELU via 1 fused M1Triton kernel."""

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

    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            tuple(ref.sum_tensor.shape),
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
            self.sum_tensor.copy_(ref.sum_tensor.detach().to(self.sum_tensor.device))

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
                ("leaky_relu", 0.2),
                ("add_bias", "ep0_ptr"),
                ("clamp", -1.0, 1.0),
                ("gelu",),
            ),
            extra_bufs=(self.sum_tensor.view(-1),),
        )
