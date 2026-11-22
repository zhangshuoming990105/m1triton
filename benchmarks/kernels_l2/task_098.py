"""KernelBench L2 #098: Matmul + AvgPool1d + GELU + Scale + Max (FusedTail Pool1DAvg + Stage 2)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
import _l2_gemm_common as C


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, pool_kernel_size: int, scale_factor: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.matmul.weight.shape
        ks = ref.avg_pool.kernel_size
        k_val = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
        m = cls(in_f, out_f, k_val, float(ref.scale_factor))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.matmul.weight)
        C.copy_tensor_(self.bias, ref.matmul.bias)
        ks = ref.avg_pool.kernel_size
        self.pool_kernel_size = int(ks[0] if isinstance(ks, (tuple, list)) else ks)
        self.scale_factor = float(ref.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        kp = self.pool_kernel_size
        if 64 % kp == 0 or 128 % kp == 0:
            pooled = torch.empty((m, n // kp), dtype=x_c.dtype, device=x_c.device)
            out = torch.empty((m,), dtype=x_c.dtype, device=x_c.device)
            with C.m1_runtime.command_batch(wait=False, sync_mps=True):
                C.fused_linear_epilogue(
                    x_c,
                    self.weight,
                    self.bias,
                    tail={"kind": "pool1d_avg", "kernel_size": kp, "stride": kp},
                    out=pooled,
                    sync_mps=False,
                )
                C.finalize_row_max(
                    pooled, scale=self.scale_factor, keepdim=False, pre_op=1, out=out, sync_mps=False
                )
            return out
        lin, _ = C.fused_linear_epilogue(x_c, self.weight, self.bias)
        pooled = C.pool1d_2d_fallback(lin, kind="avg", kernel_size=kp, sync_mps=False)
        return C.finalize_row_max(pooled, scale=self.scale_factor, keepdim=False, pre_op=1, sync_mps=False)
