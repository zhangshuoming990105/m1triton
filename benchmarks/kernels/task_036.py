"""KernelBench L1 Task #36: RMSNorm over channels via 1-read SRAM [64, 64] microkernel."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import rms_norm_channels

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_channels(x, eps=self.eps)
