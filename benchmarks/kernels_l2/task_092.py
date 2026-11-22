"""KernelBench L2 Task #92: Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp via M1Triton."""

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
    group_norm_tanh_hardswish_res_logsumexp,
    m1_rt,
)


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self._init_group_norm(groups, out_channels, eps=eps)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.group_norm.num_groups,
            eps=float(ref.group_norm.eps),
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        self._sync_group_norm_from(ref.group_norm)

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
                sync_mps=False,
                wait=False,
            )
            y = group_norm_tanh_hardswish_res_logsumexp(
                y,
                self.gn_weight,
                self.gn_bias,
                num_groups=self.gn_num_groups,
                eps=self.gn_eps,
                sync_mps=False,
                wait=False,
            )
        return y
