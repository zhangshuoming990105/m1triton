"""KernelBench L1 Task #5: Matrix-scalar multiplication (65536x16384 * s)."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _gemm_common import mat_scalar_mul


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return mat_scalar_mul(A, s)
