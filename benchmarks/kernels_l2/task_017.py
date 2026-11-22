"""KernelBench L2 Task #17: Conv2d + InstanceNorm2d + Divide via fused M1Triton Metal kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, conv_nd_epilogue, instance_norm_divide, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self._init_instance_norm(out_channels, eps=1e-5, affine=False)
        self.divide_by = float(divide_by)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(ref.conv.in_channels, ref.conv.out_channels, ref.conv.kernel_size, ref.divide_by)
        model._init_instance_norm(
            ref.conv.out_channels,
            eps=float(ref.instance_norm.eps),
            affine=bool(ref.instance_norm.affine),
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        self._sync_instance_norm_from(ref.instance_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with m1_rt.command_batch(wait=True, sync_mps=True):
            y = conv_nd_epilogue(
                x,
                self.conv_weight,
                self.conv_bias,
                ndim=2,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                use_staged_4x3=True,
                sync_mps=False,
                wait=False,
            )
            y = instance_norm_divide(
                y,
                self.in_weight,
                self.in_bias,
                eps=self.in_eps,
                divide_by=self.divide_by,
                sync_mps=False,
                wait=False,
            )
        return y
