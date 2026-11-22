"""KernelBench L2 Task 034: ConvTranspose3d + LayerNorm + GELU + Scaling (2-dispatch fused)."""

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
    launch_conv_transpose_fused,
    layernorm_gelu_scale,
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
        bias: bool = True,
        eps: float = 1e-5,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self._init_conv_transpose(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self._init_ln_params((out_channels,), eps=eps)
        self.scaling_factor = float(scaling_factor)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> "ModelNew":
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m._init_conv_transpose_from_ref(ref.conv_transpose)
        m._init_ln_params(tuple(int(v) for v in ref.layer_norm.normalized_shape), eps=float(ref.layer_norm.eps))
        m.scaling_factor = float(ref.scaling_factor)
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        self._sync_conv_transpose_from_ref(ref.conv_transpose)
        self._sync_ln_from_ref(ref.layer_norm)
        self.scaling_factor = float(ref.scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        prep = self._prepare_conv()
        ln_w = self.ln_weight.detach().contiguous().view(-1)
        ln_b = self.ln_bias.detach().contiguous().view(-1)
        with m1triton.runtime.command_batch(wait=True, sync_mps=True):
            y = launch_conv_transpose_fused(x_c, prep)
            out = layernorm_gelu_scale(y, ln_w, ln_b, eps=self.ln_eps, scale=self.scaling_factor)
        return out
