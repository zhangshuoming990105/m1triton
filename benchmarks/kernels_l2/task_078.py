"""KernelBench L2 Task 078: ConvTranspose3d + MaxPool3d + MaxPool3d + Sum(dim=1)."""

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
    axis_sum,
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
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self.p1_k = (2, 2, 2)
        self.p1_s = (2, 2, 2)
        self.p1_p = (0, 0, 0)
        self.p1_d = (1, 1, 1)
        self.p2_k = (3, 3, 3)
        self.p2_s = (3, 3, 3)
        self.p2_p = (0, 0, 0)
        self.p2_d = (1, 1, 1)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        mp1, mp2 = ref.max_pool1, ref.max_pool2
        m.p1_k = as_tuple(mp1.kernel_size, 3)
        m.p1_s = as_tuple(mp1.stride if mp1.stride is not None else mp1.kernel_size, 3)
        m.p1_p = as_tuple(mp1.padding, 3)
        m.p1_d = as_tuple(mp1.dilation, 3)
        m.p2_k = as_tuple(mp2.kernel_size, 3)
        m.p2_s = as_tuple(mp2.stride if mp2.stride is not None else mp2.kernel_size, 3)
        m.p2_p = as_tuple(mp2.padding, 3)
        m.p2_d = as_tuple(mp2.dilation, 3)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        can_merge = (
            self.p1_s == self.p1_k
            and self.p2_s == self.p2_k
            and self.p1_p == (0, 0, 0)
            and self.p2_p == (0, 0, 0)
            and self.p1_d == (1, 1, 1)
            and self.p2_d == (1, 1, 1)
        )
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep)
            if can_merge:
                merged_k = tuple(a * b for a, b in zip(self.p1_k, self.p2_k))
                y = pool_nd(y, kind="max", ndim=3, kernel_size=merged_k, stride=merged_k, padding=0, dilation=1)
            else:
                y = pool_nd(
                    y, kind="max", ndim=3, kernel_size=self.p1_k, stride=self.p1_s, padding=self.p1_p, dilation=self.p1_d
                )
                y = pool_nd(
                    y, kind="max", ndim=3, kernel_size=self.p2_k, stride=self.p2_s, padding=self.p2_p, dilation=self.p2_d
                )
            out = axis_sum(y, dim=1, keepdim=True)
        return out
