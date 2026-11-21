"""KernelBench L1 Task #63: Standard 2D convolution, square input and square kernel via a single M1Triton implicit-GEMM Metal kernel."""

from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_std_common import ConvStdModule


class ModelNew(ConvStdModule):
    _NDIM = 2
    _REF_ATTR = "conv2d"

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False) -> None:
        nn.Module.__init__(self)
        self._setup(in_channels, out_channels, (kernel_size, kernel_size), stride, padding, dilation, groups, bias)
