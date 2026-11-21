"""KernelBench L1 Task #97: ScaledDotProductAttention on (32, 32, 512, 1024) float32."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _flash_attn_common import m1_scaled_dot_product_attention


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return m1_scaled_dot_product_attention(Q, K, V)
