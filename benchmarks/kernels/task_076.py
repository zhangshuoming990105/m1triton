"""KernelBench L1 Task #76: Standard 1D convolution, dilated and strided via a single M1Triton implicit-GEMM Metal kernel."""

from __future__ import annotations

from pathlib import Path
import sys

import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_std_common import ConvStdModule


class ModelNew(ConvStdModule):
    _NDIM = 1
    _REF_ATTR = "conv1d"

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False) -> None:
        nn.Module.__init__(self)
        self._setup(in_channels, out_channels, kernel_size, stride, 0, dilation, 1, bias)
