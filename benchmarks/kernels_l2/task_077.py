"""KernelBench L2 Task 077: ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool (3-dispatch fused)."""

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
    finalize_spatial_partials_sum,
    launch_conv_transpose_fused,
    zero_pad_conv_transpose_gap,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(in_channels, out_channels, kernel_size, bias=True)
        self.scale_factor = float(scale_factor)
        self._init_bn_buffers(out_channels, eps=eps)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m.scale_factor = float(ref.scale_factor)
        m._init_bn_buffers(int(ref.batch_norm.num_features), eps=float(ref.batch_norm.eps))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self.scale_factor = float(ref.scale_factor)
        self._sync_bn_from_ref(ref.batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        n = int(x_c.shape[0])
        bn_w = self.bn_weight.detach().contiguous()
        bn_b = self.bn_bias.detach().contiguous()
        bn_rm = self.bn_running_mean.detach().contiguous()
        bn_rv = self.bn_running_var.detach().contiguous()
        if self.padding == (0, 0, 0) and self.output_padding == (0, 0, 0) and self.groups == 1:
            w_c = self.conv_weight.detach().contiguous()
            cb_c = None if self.conv_bias is None else self.conv_bias.detach().contiguous()
            with m1triton.runtime.command_batch(wait=True, sync_mps=True):
                bn_sc, bn_sh = bn_fold_affine(bn_w, bn_b, bn_rm, bn_rv, self.bn_eps)
                out = zero_pad_conv_transpose_gap(
                    x_c,
                    w_c,
                    cb_c,
                    ndim=3,
                    stride=self.stride,
                    dilation=self.dilation,
                    out_shape=(n, self.out_channels, 1, 1, 1),
                    pre_scale=float(self.scale_factor),
                    bn_scale=bn_sc,
                    bn_shift=bn_sh,
                )
            return out
        prep = self._prepare_conv(has_tail=True)
        ep = (
            ("mul_const", float(self.scale_factor)),
            ("mul_bias", "eb0"),
            ("add_bias", "eb1"),
        )
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            bn_sc, bn_sh = bn_fold_affine(bn_w, bn_b, bn_rm, bn_rv, self.bn_eps)
            partials = launch_conv_transpose_fused(
                x_c,
                prep,
                epilogue=ep,
                eb0=bn_sc,
                eb1=bn_sh,
                tail_kind="spatial_global_avg_pool",
            )
            out = finalize_spatial_partials_sum(partials, (n, self.out_channels, 1, 1, 1))
        return out
