"""KernelBench L1 Task #50: Standard 2D convolution (AlexNet conv1, 3->96, 11x11, stride 4, padding 2, bias) via a single M1Triton implicit-GEMM Metal kernel."""

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
    _REF_ATTR = "conv1"

    def __init__(self, num_classes=1000) -> None:
        nn.Module.__init__(self)
        self._setup(3, 96, 11, 4, 2, 1, 1, True)
