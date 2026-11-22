"""KernelBench L2 #033: Gemm + Scale + BatchNorm1d (live GPU BN fold + 1 fused SGMM)."""

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
        scale_shape: tuple[int, ...],
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(scale_shape, dtype=torch.float32))
        self.bn_weight = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.bn_bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self.register_buffer("bn_running_mean", torch.zeros((out_features,), dtype=torch.float32))
        self.register_buffer("bn_running_var", torch.ones((out_features,), dtype=torch.float32))
        self.bn_eps = float(eps)
        self.bn_momentum = float(momentum)

    @classmethod
    def from_reference(cls, ref: nn.Module) -> ModelNew:
        out_f, in_f = ref.gemm.weight.shape
        m = cls(in_f, out_f, tuple(ref.scale.shape), eps=float(ref.bn.eps), momentum=float(ref.bn.momentum))
        m.sync_from_reference(ref)
        return m

    def sync_from_reference(self, ref: nn.Module) -> None:
        C.copy_tensor_(self.weight, ref.gemm.weight)
        C.copy_tensor_(self.bias, ref.gemm.bias)
        C.copy_tensor_(self.scale, ref.scale)
        C.copy_tensor_(self.bn_weight, ref.bn.weight)
        C.copy_tensor_(self.bn_bias, ref.bn.bias)
        C.copy_tensor_(self.bn_running_mean, ref.bn.running_mean)
        C.copy_tensor_(self.bn_running_var, ref.bn.running_var)
        self.bn_eps = float(ref.bn.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x if x.is_contiguous() else x.contiguous()
        m = int(x_c.shape[0])
        n = int(self.weight.shape[0])
        eff_scale = torch.empty((n,), dtype=x_c.dtype, device=x_c.device)
        eff_shift = torch.empty((n,), dtype=x_c.dtype, device=x_c.device)
        out = torch.empty((m, n), dtype=x_c.dtype, device=x_c.device)
        with C.m1_runtime.command_batch(wait=False, sync_mps=True):
            C.bn_fold_affine_gpu(
                self.bn_running_mean,
                self.bn_running_var,
                self.bn_weight,
                self.bn_bias,
                eps=self.bn_eps,
                pre_scale=self.scale,
                out_scale=eff_scale,
                out_shift=eff_shift,
                sync_mps=False,
            )
            C.fused_linear_epilogue(
                x_c,
                self.weight,
                self.bias,
                epilogue_steps=[("mul_bias", "buf0"), ("add_bias", "buf1")],
                extra_bufs=[eff_scale, eff_shift],
                out=out,
                sync_mps=False,
            )
        return out
