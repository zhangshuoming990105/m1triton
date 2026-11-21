"""KernelBench L1 Task #99: Triplet Margin loss via fused L2 distance reduction."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _loss_common import triplet_margin_loss


class ModelNew(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return triplet_margin_loss(anchor, positive, negative, margin=self.margin, eps=1e-6)
