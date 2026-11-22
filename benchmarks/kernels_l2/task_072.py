"""KernelBench L2 Task 072: ConvTranspose3d + BatchNorm + AvgPool3d + AvgPool3d (3-dispatch fused)."""

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
    bn_fold_affine,
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
        bias_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True
        )
        self._init_bn_buffers(out_channels, eps=1e-5)
        self.p1_k = (2, 2, 2)
        self.p1_s = (2, 2, 2)
        self.p1_p = (0, 0, 0)
        self.p2_k = (2, 2, 2)
        self.p2_s = (2, 2, 2)
        self.p2_p = (0, 0, 0)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m._init_bn_buffers(int(ref.batch_norm.num_features), eps=float(ref.batch_norm.eps))
        ap1, ap2 = ref.avg_pool1, ref.avg_pool2
        m.p1_k = as_tuple(ap1.kernel_size, 3)
        m.p1_s = as_tuple(ap1.stride if ap1.stride is not None else ap1.kernel_size, 3)
        m.p1_p = as_tuple(ap1.padding, 3)
        m.p2_k = as_tuple(ap2.kernel_size, 3)
        m.p2_s = as_tuple(ap2.stride if ap2.stride is not None else ap2.kernel_size, 3)
        m.p2_p = as_tuple(ap2.padding, 3)
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
        can_merge_pools = (
            self.p1_s == self.p1_k
            and self.p2_s == self.p2_k
            and self.p1_p == (0, 0, 0)
            and self.p2_p == (0, 0, 0)
        )
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            bn_sc, bn_sh = bn_fold_affine(bn_w, bn_b, bn_rm, bn_rv, self.bn_eps)
            y = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=(("mul_bias", "eb0"), ("add_bias", "eb1")),
                eb0=bn_sc,
                eb1=bn_sh,
            )
            if can_merge_pools:
                merged_k = tuple(a * b for a, b in zip(self.p1_k, self.p2_k))
                out = pool_nd(y, kind="avg", ndim=3, kernel_size=merged_k, stride=merged_k, padding=0)
            else:
                y = pool_nd(y, kind="avg", ndim=3, kernel_size=self.p1_k, stride=self.p1_s, padding=self.p1_p)
                out = pool_nd(y, kind="avg", ndim=3, kernel_size=self.p2_k, stride=self.p2_s, padding=self.p2_p)
        return out
