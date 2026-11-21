"""KernelBench L1 Task #6: Large-K matrix multiplication (256x524288 @ 524288x256)."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _gemm_common import gemm_splitk


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._workspace: torch.Tensor | None = None

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        out, self._workspace = gemm_splitk(A, B, split_k=32, workspace=self._workspace)
        return out
