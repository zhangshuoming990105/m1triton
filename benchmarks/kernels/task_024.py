"""KernelBench L1 Task #24: LogSoftmax over dim=1 via fused online softmax."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import softmax_rows

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if dim == x.ndim - 1:
            return softmax_rows(x, log_softmax=True)
        perm = [i for i in range(x.ndim) if i != dim] + [dim]
        inv_perm = [0] * x.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return softmax_rows(x.permute(*perm).contiguous(), log_softmax=True).permute(*inv_perm).contiguous()
