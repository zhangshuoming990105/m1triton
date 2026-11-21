"""KernelBench L1 Task #69: ConvTranspose2d (asymmetric input, asymmetric kernel) via one M1Triton implicit-GEMM Metal kernel."""

from __future__ import annotations

from pathlib import Path
import sys

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_transpose_common import ConvTransposeNdBase


class ModelNew(ConvTransposeNdBase):
    ndim = 2
    ref_attr = "conv_transpose2d"

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
