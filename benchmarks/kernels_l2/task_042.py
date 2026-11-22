"""KernelBench L2 Task 042: ConvTranspose2d + SpatialGlobalAvgPool + BiasAdd + LogSumExp + Sum + Multiply."""

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
    row_logsumexp_scale,
    zero_pad_conv_transpose_gap,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._init_conv_transpose(in_channels, out_channels, kernel_size, bias=True)
        self.extra_bias = nn.Parameter(torch.empty(bias_shape))

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.extra_bias = nn.Parameter(torch.empty_like(ref.bias.detach()))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        with torch.no_grad():
            self.extra_bias.copy_(ref.bias.detach().to(self.extra_bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        n = int(x_c.shape[0])
        eb = self.extra_bias.detach().contiguous().view(-1)
        if self.padding == (0, 0) and self.output_padding == (0, 0) and self.groups == 1:
            w_c = self.conv_weight.detach().contiguous()
            cb_c = None if self.conv_bias is None else self.conv_bias.detach().contiguous()
            with m1triton.runtime.command_batch(wait=True, sync_mps=True):
                gap = zero_pad_conv_transpose_gap(
                    x_c,
                    w_c,
                    cb_c,
                    ndim=2,
                    stride=self.stride,
                    dilation=self.dilation,
                    out_shape=(n, self.out_channels),
                    extra_bias=eb,
                )
                out = row_logsumexp_scale(gap, 10.0)
            return out
        prep = self._prepare_conv(has_tail=True)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            partials = launch_conv_transpose_fused(x_c, prep, tail_kind="spatial_global_avg_pool")
            gap = finalize_spatial_partials_sum(partials, (n, self.out_channels), bias=eb)
            out = row_logsumexp_scale(gap, 10.0)
        return out
