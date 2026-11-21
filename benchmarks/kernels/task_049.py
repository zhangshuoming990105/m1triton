"""KernelBench L1 Task #49: Max reduction over a dimension."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _axis_reduce_common import axis_max


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return axis_max(x, self.dim, keepdim=False)
