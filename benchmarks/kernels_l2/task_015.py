"""KernelBench L2 Task 015: ConvTranspose3d + BatchNorm + SubtractSpatialMean."""

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
    bn_fold_affine,
    launch_conv_transpose_fused,
    subtract_spatial_mean,
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
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self._init_bn_buffers(out_channels, eps=1e-5)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m._init_bn_buffers(int(ref.batch_norm.num_features), eps=float(ref.batch_norm.eps))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self._sync_bn_from_ref(ref.batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        bn_w = self.bn_weight.detach().contiguous()
        bn_b = self.bn_bias.detach().contiguous()
        bn_rm = self.bn_running_mean.detach().contiguous()
        bn_rv = self.bn_running_var.detach().contiguous()
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            bn_sc, bn_sh = bn_fold_affine(bn_w, bn_b, bn_rm, bn_rv, self.bn_eps)
            y = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=(("mul_bias", "eb0"), ("add_bias", "eb1")),
                eb0=bn_sc,
                eb1=bn_sh,
            )
            out = subtract_spatial_mean(y)
        return out
