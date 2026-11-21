"""KernelBench L1 Task #87: Pointwise (1x1) 2D convolution via a single M1Triton implicit-GEMM Metal kernel."""

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
    _REF_ATTR = "conv1d"

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        nn.Module.__init__(self)
        self._setup(in_channels, out_channels, 1, 1, 0, 1, 1, bias)
