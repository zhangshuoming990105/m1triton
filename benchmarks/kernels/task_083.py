"""KernelBench L1 Task #83: Depthwise Conv2D (square input, asymmetric (k, 1) kernel) via a single SRAM-halo M1Triton Metal depthwise kernel."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_dw_common import depthwise_conv2d


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        bound = 1.0 / math.sqrt(kernel_size)
        self.weight = nn.Parameter(torch.empty(self.out_channels, 1, kernel_size, 1).uniform_(-bound, bound))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels).uniform_(-bound, bound))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ModelNew":
        conv = ref_model.conv2d
        model = cls(*(int(conv.in_channels), int(conv.kernel_size[0]), int(conv.stride[0]), int(conv.padding[0]), int(conv.dilation[0]), conv.bias is not None))
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        conv = ref_model.conv2d
        with torch.no_grad():
            self.weight.copy_(conv.weight.detach().to(self.weight.device))
            if self.bias is not None:
                self.bias.copy_(conv.bias.detach().to(self.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.detach() if self.bias is not None else None
        return depthwise_conv2d(
            x,
            self.weight.detach(),
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
