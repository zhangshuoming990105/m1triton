"""KernelBench L1 Task #37: Frobenius norm normalization via two-stage grid reduction."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _norm_softmax_common import frobenius_stage1_kernel, frobenius_stage2_kernel

_STAGE1_MAX_TGS = 1024
_STAGE2_MAX_TGS = 2048


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x if x.is_contiguous() else x.contiguous()
        total = int(xc.numel())
        if total == 0:
            return torch.empty_like(xc)
        g1 = max(1, min(_STAGE1_MAX_TGS, _cdiv(total, 256 * 16)))
        g2 = max(1, min(_STAGE2_MAX_TGS, _cdiv(total, 256 * 8)))
        partial = torch.empty((g1,), device=xc.device, dtype=torch.float32)
        out = torch.empty_like(xc)
        frobenius_stage1_kernel[(g1, 1, 1)](xc, partial, total)
        frobenius_stage2_kernel[(g2, 1, 1)](xc, out, partial, total, g1, sync_mps=False)
        return out
