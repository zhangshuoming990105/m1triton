"""KernelBench L2 Task 074: ConvTranspose3d + LeakyReLU + Multiply + LeakyReLU + MaxPool3d (2-dispatch fused)."""

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
    as_tuple,
    launch_conv_transpose_fused,
    pool_nd,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        multiplier_shape: tuple[int, ...],
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
        self.multiplier = nn.Parameter(torch.empty(multiplier_shape))
        self.negative_slope = 0.2
        self.pool_k = (2, 2, 2)
        self.pool_s = (2, 2, 2)
        self.pool_p = (0, 0, 0)
        self.pool_d = (1, 1, 1)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.multiplier = nn.Parameter(torch.empty_like(ref.multiplier.detach()))
        m.negative_slope = float(ref.leaky_relu.negative_slope)
        mp = ref.max_pool
        m.pool_k = as_tuple(mp.kernel_size, 3)
        m.pool_s = as_tuple(mp.stride if mp.stride is not None else mp.kernel_size, 3)
        m.pool_p = as_tuple(mp.padding, 3)
        m.pool_d = as_tuple(mp.dilation, 3)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.negative_slope = float(ref.leaky_relu.negative_slope)
        with torch.no_grad():
            self.multiplier.copy_(ref.multiplier.detach().to(self.multiplier.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        mul = self.multiplier.detach().contiguous().view(-1)
        ns = float(self.negative_slope)
        ep = (
            ("leaky_relu", ns),
            ("mul_bias", "eb0"),
            ("leaky_relu", ns),
        )
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep, epilogue=ep, eb0=mul)
            out = pool_nd(
                y,
                kind="max",
                ndim=3,
                kernel_size=self.pool_k,
                stride=self.pool_s,
                padding=self.pool_p,
                dilation=self.pool_d,
            )
        return out
