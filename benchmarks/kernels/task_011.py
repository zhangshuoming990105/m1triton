"""KernelBench L1 Task #11: 4D tensor-matrix multiplication (8x256x512x256 @ 256x768)."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _gemm_common import gemm_2d


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, i, j, l = A.shape
        k = B.shape[1]
        a_2d = A.contiguous().reshape(b * i * j, l)
        out_2d = gemm_2d(a_2d, B)
        return out_2d.reshape(b, i, j, k)
