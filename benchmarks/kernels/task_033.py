"""KernelBench L1 Task #33: Training-mode BatchNorm2d via two-stage parallel Welford."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import batch_norm_train

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ModelNew":
        model = cls(int(ref_model.bn.num_features), float(ref_model.bn.eps))
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        bn = ref_model.bn
        with torch.no_grad():
            self.weight.copy_(bn.weight.detach().to(self.weight.device))
            self.bias.copy_(bn.bias.detach().to(self.bias.device))
            self.running_mean.copy_(bn.running_mean.detach().to(self.running_mean.device))
            self.running_var.copy_(bn.running_var.detach().to(self.running_var.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return batch_norm_train(x, self.weight.detach(), self.bias.detach(), eps=self.eps)
