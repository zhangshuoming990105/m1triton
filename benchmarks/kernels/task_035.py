"""KernelBench L1 Task #35: GroupNorm via two-stage parallel Welford."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import group_norm

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_groups = int(num_groups)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ModelNew":
        gn = ref_model.gn
        model = cls(int(gn.num_channels), int(gn.num_groups), float(gn.eps))
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        gn = ref_model.gn
        with torch.no_grad():
            self.weight.copy_(gn.weight.detach().to(self.weight.device))
            self.bias.copy_(gn.bias.detach().to(self.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return group_norm(x, self.weight.detach(), self.bias.detach(), num_groups=self.num_groups, eps=self.eps)
