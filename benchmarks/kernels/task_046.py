"""KernelBench L1 Task #46: Average Pooling 3D via a single M1Triton Metal pooling kernel."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _pool_common import pool_nd


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pool_nd(
            x,
            kind="avg",
            ndim=3,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
