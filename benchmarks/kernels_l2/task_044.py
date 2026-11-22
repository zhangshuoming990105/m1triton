"""KernelBench L2 Task 044: ConvTranspose2d + Multiply + GlobalAvgPool + GlobalAvgPool (2-dispatch fused)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import m1triton
from _l2_conv_transpose_common import (
    ConvTransposeL2Base,
    finalize_spatial_partials_sum,
    launch_conv_transpose_fused,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        multiplier: float,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.multiplier = float(multiplier)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.multiplier = float(ref.multiplier)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.multiplier = float(ref.multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        n = int(x_c.shape[0])
        prep = self._prepare_conv(has_tail=True)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            partials = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=(("mul_const", float(self.multiplier)),),
                tail_kind="spatial_global_avg_pool",
            )
            out = finalize_spatial_partials_sum(partials, (n, self.out_channels, 1, 1))
        return out
