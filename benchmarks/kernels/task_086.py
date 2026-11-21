"""KernelBench L1 Task #86: Depthwise-separable Conv2D via an M1Triton depthwise kernel plus a 1x1 SGMM kernel."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_dw_common import depthwise_conv2d, pointwise_conv2d


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        dw_bound = 1.0 / math.sqrt(kernel_size * kernel_size)
        pw_bound = 1.0 / math.sqrt(self.in_channels)
        self.dw_weight = nn.Parameter(torch.empty(self.in_channels, 1, kernel_size, kernel_size).uniform_(-dw_bound, dw_bound))
        self.pw_weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels, 1, 1).uniform_(-pw_bound, pw_bound))
        if bias:
            self.dw_bias = nn.Parameter(torch.empty(self.in_channels).uniform_(-dw_bound, dw_bound))
            self.pw_bias = nn.Parameter(torch.empty(self.out_channels).uniform_(-pw_bound, pw_bound))
        else:
            self.register_parameter("dw_bias", None)
            self.register_parameter("pw_bias", None)

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ModelNew":
        dw = ref_model.depthwise
        pw = ref_model.pointwise
        model = cls(
            int(dw.in_channels),
            int(pw.out_channels),
            int(dw.kernel_size[0]),
            int(dw.stride[0]),
            int(dw.padding[0]),
            int(dw.dilation[0]),
            dw.bias is not None,
        )
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        dw = ref_model.depthwise
        pw = ref_model.pointwise
        with torch.no_grad():
            self.dw_weight.copy_(dw.weight.detach().to(self.dw_weight.device))
            self.pw_weight.copy_(pw.weight.detach().to(self.pw_weight.device))
            if self.dw_bias is not None:
                self.dw_bias.copy_(dw.bias.detach().to(self.dw_bias.device))
                self.pw_bias.copy_(pw.bias.detach().to(self.pw_bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw_bias = self.dw_bias.detach() if self.dw_bias is not None else None
        pw_bias = self.pw_bias.detach() if self.pw_bias is not None else None
        mid = depthwise_conv2d(
            x,
            self.dw_weight.detach(),
            dw_bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            wait=False,
        )
        return pointwise_conv2d(mid, self.pw_weight.detach(), pw_bias)
