"""KernelBench L2 Task 061: ConvTranspose3d + ReLU + GroupNorm."""

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
    group_norm,
    launch_conv_transpose_fused,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(in_channels, out_channels, kernel_size, bias=bias)
        self._init_gn_params(groups, out_channels, eps=1e-5)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m._init_gn_params(
            int(ref.group_norm.num_groups), int(ref.group_norm.num_channels), eps=float(ref.group_norm.eps)
        )
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self._sync_gn_from_ref(ref.group_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        gn_w = self.gn_weight.detach().contiguous()
        gn_b = self.gn_bias.detach().contiguous()
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep, epilogue=(("relu",),))
            out = group_norm(y, gn_w, gn_b, num_groups=self.gn_num_groups, eps=self.gn_eps)
        return out
