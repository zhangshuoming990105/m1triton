"""KernelBench L2 Task 013: ConvTranspose3d + DepthMean + AddBias + ChannelSoftmax + Tanh + Scale."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import m1triton
from _l2_conv_transpose_common import (
    ConvTransposeL2Base,
    channel_softmax_post,
    depth_mean_add_bias,
    launch_conv_transpose_fused,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scaling_factor: float,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.extra_bias = nn.Parameter(torch.empty(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.extra_bias = nn.Parameter(torch.empty_like(ref.bias.detach()))
        m.scaling_factor = float(ref.scaling_factor)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.scaling_factor = float(ref.scaling_factor)
        with torch.no_grad():
            self.extra_bias.copy_(ref.bias.detach().to(self.extra_bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        eb = self.extra_bias.detach().contiguous().view(-1)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep)
            y = depth_mean_add_bias(y, eb)
            out = channel_softmax_post(y, mode=1, scale=self.scaling_factor)
        return out
