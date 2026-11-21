"""KernelBench L1 Task #64: ConvTranspose1d (1D) via one M1Triton implicit-GEMM Metal kernel."""

from __future__ import annotations

from pathlib import Path
import sys

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _conv_transpose_common import ConvTransposeNdBase


class ModelNew(ConvTransposeNdBase):
    ndim = 1
    ref_attr = "conv1d_transpose"

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
