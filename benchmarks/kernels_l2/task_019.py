"""KernelBench L2 Task 019: ConvTranspose2d + GELU + GroupNorm."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import m1triton
from _l2_conv_std_common import conv_nd_epilogue
from _l2_conv_transpose_common import (
    ConvTransposeL2Base,
    group_norm,
    launch_conv_transpose_fused,
)


class ModelNew(ConvTransposeL2Base):
    ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        num_groups: int,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=True
        )
        self._init_gn_params(num_groups, out_channels, eps=1e-5)

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
        gn_w = self.gn_weight.detach().contiguous()
        gn_b = self.gn_bias.detach().contiguous()
        h_out = (int(x_c.shape[2]) - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        w_out = (int(x_c.shape[3]) - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        if (
            self.kernel_size == (3, 3)
            and self.stride == (1, 1)
            and self.dilation == (1, 1)
            and self.output_padding == (0, 0)
            and self.groups == 1
            and self.padding[0] <= 2
            and self.padding[1] <= 2
            and x_c.dtype == torch.float32
            and int(x_c.shape[1]) >= 8
            and self.out_channels >= 16
            and h_out * w_out >= 256
        ):
            with m1triton.runtime.command_batch(wait=True, sync_mps=True):
                y = conv_nd_epilogue(
                    x_c,
                    self.conv_weight.detach(),
                    None if self.conv_bias is None else self.conv_bias.detach(),
                    ndim=2,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    epilogue_steps=(("gelu",),),
                    transposed=True,
                    use_staged_4x3=False,
                    sync_mps=False,
                    wait=False,
                )
                out = group_norm(y, gn_w, gn_b, num_groups=self.gn_num_groups, eps=self.gn_eps)
            return out
        prep = self._prepare_conv()
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep, epilogue=(("gelu",),))
            out = group_norm(y, gn_w, gn_b, num_groups=self.gn_num_groups, eps=self.gn_eps)
        return out
