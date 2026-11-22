"""KernelBench L2 Task 026: ConvTranspose3d + ResidualAdd + x * HardSwish(x) (1-dispatch fused)."""

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
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.extra_bias = nn.Parameter(torch.empty(bias_shape))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.extra_bias = nn.Parameter(torch.empty_like(ref.bias.detach()))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        with torch.no_grad():
            self.extra_bias.copy_(ref.bias.detach().to(self.extra_bias.device))

    def forward(self, x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
        ep = (
            ("add_residual", "eb0"),
            ("save_reg", 0),
            ("hardswish",),
            ("mul_saved", 0),
        )
        return conv_transpose_nd_epilogue(
            x,
            self.conv_weight.detach(),
            None if self.conv_bias is None else self.conv_bias.detach(),
            ndim=3,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            epilogue=ep,
            eb0=add_input.contiguous(),
        )
