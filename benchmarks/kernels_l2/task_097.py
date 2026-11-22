"""KernelBench L2 #097: Matmul + BatchNorm1d + BiasAdd + Divide + Swish (2 fused dispatches)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_DIR = str(Path(__file__).resolve().parent)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
import _l2_gemm_common as C


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bias_shape: tuple[int, ...] = (1,),
        divide_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.linear_bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.bn_weight = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.register_buffer("bn_running_mean", torch.zeros((out_features,), dtype=torch.float32))
        self.register_buffer("bn_running_var", torch.ones((out_features,), dtype=torch.float32))
        self.extra_bias = nn.Parameter(torch.empty(bias_shape, dtype=torch.float32))
        self.bn_eps = float(bn_eps)
        self.bn_momentum = float(bn_momentum)
        self.divide_value = float(divide_value)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.matmul.weight.shape
        m = cls(
            in_f,
            out_f,
            bn_eps=float(ref.bn.eps),
            bn_momentum=float(ref.bn.momentum),
            bias_shape=tuple(ref.bias.shape),
            divide_value=float(ref.divide_value),
        )
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.matmul.weight)
        C.copy_tensor_(self.linear_bias, ref.matmul.bias)
        C.copy_tensor_(self.bn_weight, ref.bn.weight)
        C.copy_tensor_(self.bn_bias, ref.bn.bias)
        C.copy_tensor_(self.bn_running_mean, ref.bn.running_mean)
        C.copy_tensor_(self.bn_running_var, ref.bn.running_var)
        C.copy_tensor_(self.extra_bias, ref.bias)
        self.bn_eps = float(ref.bn.eps)
        self.divide_value = float(ref.divide_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        eff_scale = torch.empty((n,), dtype=x_c.dtype, device=x_c.device)
        eff_shift = torch.empty((n,), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=True, sync_mps=True):
            C.bn_fold_affine_gpu(
                self.bn_running_mean,
                self.bn_running_var,
                self.bn_weight,
                self.bn_bias,
                eps=self.bn_eps,
                post_bias=self.extra_bias,
                divide_value=self.divide_value,
                out_scale=eff_scale,
                out_shift=eff_shift,
                sync_mps=False,
            )
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.linear_bias,
                epilogue_steps=[("mul_bias", "buf0"), ("add_bias", "buf1"), ("swish",)],
                extra_bufs=[eff_scale, eff_shift],
                out=out,
                sync_mps=False,
            )
        return out
