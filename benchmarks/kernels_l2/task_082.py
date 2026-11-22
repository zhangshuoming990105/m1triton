"""KernelBench L2 Task #82: Conv2d + Tanh + Scaling + BiasAdd + MaxPool2d via fused M1Triton kernels."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from _l2_conv_std_common import ConvStdL2Base, as_tuple, conv_nd_epilogue, m1_rt, pool_nd


class ModelNew(ConvStdL2Base):
    _NDIM = 2

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        nn.Module.__init__(self)
        self._init_conv(in_channels, out_channels, kernel_size, bias=True)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = as_tuple(pool_kernel_size, 2)
        self.pool_stride = as_tuple(pool_kernel_size, 2)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        model = cls(
            ref.conv.in_channels,
            ref.conv.out_channels,
            ref.conv.kernel_size,
            ref.scaling_factor,
            tuple(ref.bias.shape),
            ref.max_pool.kernel_size,
        )
        model.stride = ref.conv.stride
        model.padding = ref.conv.padding
        model.dilation = ref.conv.dilation
        model.groups = ref.conv.groups
        model.pool_stride = as_tuple(ref.max_pool.stride, 2)
        model.sync_from_reference(ref)
        return model

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_from(ref.conv)
        with torch.no_grad():
            self.bias.copy_(ref.bias.detach().to(self.bias.device))

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
                epilogue_steps=(
                    ("tanh",),
                    ("mul_const", self.scaling_factor),
                    ("add_bias", "ep0_ptr"),
                ),
                extra_bufs=(self.bias.view(-1),),
                sync_mps=False,
                wait=False,
            )
            y = pool_nd(y, kind="max", ndim=2, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        return y
