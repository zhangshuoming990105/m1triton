"""KernelBench L2 Task #83: Conv3d + GroupNorm + Min + Clamp + Dropout(eval identity) via M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, conv_nd_epilogue, group_norm_min_clamp, m1_rt


class ModelNew(ConvStdL2Base):
    _NDIM = 3

    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self._init_group_norm(groups, out_channels)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.dropout_p = float(dropout_p)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        ref_mod = sys.modules.get(ref.__class__.__module__)
        min_v = float(getattr(ref, "min_value", getattr(ref_mod, "min_value", 0.0)))
        max_v = float(getattr(ref, "max_value", getattr(ref_mod, "max_value", 1.0)))
        drop_p = float(getattr(ref.dropout, "p", 0.0)) if hasattr(ref, "dropout") else 0.0
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.norm.num_groups,
            min_v,
            max_v,
            drop_p,
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        self._sync_group_norm_from(ref.norm)

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
                sync_mps=False,
                wait=False,
            )
            y = group_norm_min_clamp(
                y,
                self.gn_weight,
                self.gn_bias,
                num_groups=self.gn_num_groups,
                eps=self.gn_eps,
                min_value=self.min_value,
                max_value=self.max_value,
                sync_mps=False,
                wait=False,
            )
        return y
