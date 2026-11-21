"""KernelBench L1 Task #100: Hinge loss via fused broadcast reduction."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _loss_common import hinge_loss


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return hinge_loss(predictions, targets)
