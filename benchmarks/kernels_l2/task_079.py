"""KernelBench L2 Task #79: Conv3d + Mul + InstanceNorm3d + Clamp + Mul + Max(dim=1) via M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import (
    ConvStdL2Base,
    conv_nd_epilogue,
    instance_norm_clamp_mul_max_dim1,
    m1_rt,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self._init_instance_norm(out_channels, eps=1e-5, affine=False)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            tuple(ref.multiplier.shape),
            ref.clamp_min,
            ref.clamp_max,
        )
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
        with torch.no_grad():
            self.multiplier.copy_(ref.multiplier.detach().to(self.multiplier.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with m1_rt.command_batch(wait=True, sync_mps=True):
            y = conv_nd_epilogue(
                x,
                self.conv_weight,
                self.conv_bias,
                ndim=3,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                epilogue_steps=(("mul_bias", "ep0_ptr"),),
                extra_bufs=(self.multiplier.view(-1),),
                sync_mps=False,
                wait=False,
            )
            y = instance_norm_clamp_mul_max_dim1(
                y,
                self.multiplier,
                self.in_weight,
                self.in_bias,
                eps=self.in_eps,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
                sync_mps=False,
                wait=False,
            )
        return y
