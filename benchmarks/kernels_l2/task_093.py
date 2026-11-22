"""KernelBench L2 Task 093: ConvTranspose2d + Add + Min(0) + GELU + Multiply (1-dispatch fused)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _l2_conv_transpose_common import ConvTransposeL2Base, conv_transpose_nd_epilogue


class ModelNew(ConvTransposeL2Base):
    ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        add_value: float,
        multiply_value: float,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, bias=True
        )
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.add_value = float(ref.add_value)
        m.multiply_value = float(ref.multiply_value)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.add_value = float(ref.add_value)
        self.multiply_value = float(ref.multiply_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ep = (
            ("add_const", float(self.add_value)),
            ("min_const", 0.0),
            ("gelu",),
            ("mul_const", float(self.multiply_value)),
        )
        return conv_transpose_nd_epilogue(
            x,
            self.conv_weight.detach(),
            None if self.conv_bias is None else self.conv_bias.detach(),
            ndim=2,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            epilogue=ep,
        )
