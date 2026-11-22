"""KernelBench L2 Task 011: ConvTranspose2d + BatchNorm + Tanh + MaxPool2d + GroupNorm."""

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
    group_norm,
    launch_conv_transpose_fused,
    pool_nd,
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
        groups: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=1, bias=True
        )
        self._init_bn_buffers(out_channels, eps=1e-5)
        self.pool_k = (2, 2)
        self.pool_s = (2, 2)
        self.pool_p = (0, 0)
        self.pool_d = (1, 1)
        self._init_gn_params(num_groups, out_channels, eps=1e-5)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m._init_bn_buffers(int(ref.batch_norm.num_features), eps=float(ref.batch_norm.eps))
        mp = ref.max_pool
        m.pool_k = as_tuple(mp.kernel_size, 2)
        m.pool_s = as_tuple(mp.stride if mp.stride is not None else mp.kernel_size, 2)
        m.pool_p = as_tuple(mp.padding, 2)
        m.pool_d = as_tuple(mp.dilation, 2)
        m._init_gn_params(int(ref.group_norm.num_groups), int(ref.group_norm.num_channels), eps=float(ref.group_norm.eps))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self._sync_bn_from_ref(ref.batch_norm)
        self._sync_gn_from_ref(ref.group_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        bn_w = self.bn_weight.detach().contiguous()
        bn_b = self.bn_bias.detach().contiguous()
        bn_rm = self.bn_running_mean.detach().contiguous()
        bn_rv = self.bn_running_var.detach().contiguous()
        gn_w = self.gn_weight.detach().contiguous()
        gn_b = self.gn_bias.detach().contiguous()
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            bn_sc, bn_sh = bn_fold_affine(bn_w, bn_b, bn_rm, bn_rv, self.bn_eps)
            y = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=(("mul_bias", "eb0"), ("add_bias", "eb1"), ("tanh",)),
                eb0=bn_sc,
                eb1=bn_sh,
            )
            y = pool_nd(
                y,
                kind="max",
                ndim=2,
                kernel_size=self.pool_k,
                stride=self.pool_s,
                padding=self.pool_p,
                dilation=self.pool_d,
            )
            out = group_norm(y, gn_w, gn_b, num_groups=self.gn_num_groups, eps=self.gn_eps)
        return out
