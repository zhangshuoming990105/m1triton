"""KernelBench L1 Task #34: InstanceNorm2d (affine=False) via per-row Welford."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import row_normalize

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return row_normalize(x, "instance_norm", int(x.shape[2]) * int(x.shape[3]), eps=self.eps)
