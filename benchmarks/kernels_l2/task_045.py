"""KernelBench L2 #045: Linear1 + Sigmoid + Linear2 + LogSumExp (3 fused dispatches)."""

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
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((hidden_size, input_size), dtype=torch.float32))
        self.b1 = nn.Parameter(torch.empty((hidden_size,), dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty((output_size, hidden_size), dtype=torch.float32))
        self.b2 = nn.Parameter(torch.empty((output_size,), dtype=torch.float32))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        h_s, in_s = ref.linear1.weight.shape
        out_s = ref.linear2.weight.shape[0]
        m = cls(in_s, h_s, out_s)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.w1, ref.linear1.weight)
        C.copy_tensor_(self.b1, ref.linear1.bias)
        C.copy_tensor_(self.w2, ref.linear2.weight)
        C.copy_tensor_(self.b2, ref.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        h_dim = int(self.w1.shape[0])
        o_dim = int(self.w2.shape[0])
        _, bn2, _, _, _, _ = C._select_fused_sgmm_schedule(m, o_dim, h_dim, trans_b=True, tail_kind="row_logsumexp")
        grid_n2 = (o_dim + bn2 - 1) // bn2
        h = torch.empty((m, h_dim), dtype=x_c.dtype, device=x_c.device)
        partials = torch.empty((m, grid_n2, 2), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m,), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.fused_linear_epilogue(
                x_c,
                self.w1,
                self.b1,
                epilogue_steps=[("sigmoid",)],
                out=h,
                sync_mps=False,
            )
            C.fused_linear_epilogue(
                h,
                self.w2,
                self.b2,
                tail={"kind": "row_logsumexp"},
                out=partials,
                sync_mps=False,
            )
            C.finalize_row_lse(partials, keepdim=False, out=out, sync_mps=False)
        return out
