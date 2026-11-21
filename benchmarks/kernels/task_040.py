"""KernelBench L1 Task #40: LayerNorm via two-stage parallel Welford."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import layer_norm

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = tuple(int(s) for s in normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    @classmethod
    def from_reference(cls, ref_model: nn.Module) -> "ModelNew":
        ln = ref_model.ln
        model = cls(tuple(ln.normalized_shape), float(ln.eps))
        model.sync_from_reference(ref_model)
        return model

    def sync_from_reference(self, ref_model: nn.Module) -> None:
        ln = ref_model.ln
        with torch.no_grad():
            self.weight.copy_(ln.weight.detach().to(self.weight.device))
            self.bias.copy_(ln.bias.detach().to(self.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight.detach(), self.bias.detach(), eps=self.eps)
