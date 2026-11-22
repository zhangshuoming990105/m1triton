"""KernelBench L2 Task #52: Conv2d + Mish + BatchNorm via live GPU BN-fold + fused M1Triton EpilogueDAG."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, compute_bn_affine, conv_nd_epilogue, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self._init_batch_norm(out_channels, eps=eps, affine=True)
        self.bn_momentum = float(momentum)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            eps=float(ref.bn.eps),
            momentum=float(ref.bn.momentum or 0.1),
        )
        model._init_batch_norm(ref.conv.out_channels, eps=float(ref.bn.eps), affine=bool(ref.bn.affine))
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        self._sync_batch_norm_from(ref.bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with m1_rt.command_batch(wait=True, sync_mps=True):
            bn_scale, bn_shift = compute_bn_affine(
                self.bn_weight,
                self.bn_bias,
                self.bn_running_mean,
                self.bn_running_var,
                eps=self.bn_eps,
                post_scale=1.0,
                sync_mps=False,
                wait=False,
            )
            y = conv_nd_epilogue(
                x,
                self.conv_weight,
                self.conv_bias,
                ndim=2,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                epilogue_steps=(
                    ("mish",),
                    ("mul_bias", "ep0_ptr"),
                    ("add_bias", "ep1_ptr"),
                ),
                extra_bufs=(bn_scale, bn_shift),
                sync_mps=False,
                wait=False,
            )
        return y
