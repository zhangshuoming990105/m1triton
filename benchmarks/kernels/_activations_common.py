"""Vectorized elementwise activation kernels for KernelBench L1 (relu .. hardtanh, MinGPT GELU)."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

BLOCK_SIZE = 1024

SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946
SQRT_HALF = 0.7071067811865476
GELU_TANH_COEF = 0.7978845608028654
GELU_TANH_CUBIC = 0.044715


@m1triton.jit
def relu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@m1triton.jit
def leaky_relu_kernel(x_ptr, o_ptr, n, SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 0.0, x, x * SLOPE), mask=mask)


@m1triton.jit
def sigmoid_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


@m1triton.jit
def tanh_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.tanh(x), mask=mask)


@m1triton.jit
def swish_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x * (1.0 / (1.0 + tl.exp(-x))), mask=mask)


@m1triton.jit
def gelu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476)), mask=mask)


@m1triton.jit
def selu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    neg = 1.6732632423543772848170429916717 * (tl.exp(x) - 1.0)
    tl.store(o_ptr + offs, 1.0507009873554804934193349852946 * tl.where(x > 0.0, x, neg), mask=mask)


@m1triton.jit
def hardsigmoid_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(o_ptr + offs, y, mask=mask)


@m1triton.jit
def softplus_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x))), mask=mask)


@m1triton.jit
def softsign_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, x / (1.0 + tl.abs(x)), mask=mask)


@m1triton.jit
def elu_kernel(x_ptr, o_ptr, n, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 0.0, x, ALPHA * (tl.exp(x) - 1.0)), mask=mask)


@m1triton.jit
def hardtanh_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.minimum(tl.maximum(x, -1.0), 1.0), mask=mask)


@m1triton.jit
def mingpt_gelu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (x + 0.044715 * x ** 3.0)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + tl.tanh(inner)), mask=mask)


def _launch(kernel, x: torch.Tensor, **consts) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (m1triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](x, out, n, BLOCK=BLOCK_SIZE, **consts)
    return out


def relu(x: torch.Tensor) -> torch.Tensor:
    return _launch(relu_kernel, x)


def leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return _launch(leaky_relu_kernel, x, SLOPE=float(negative_slope))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return _launch(sigmoid_kernel, x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return _launch(tanh_kernel, x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return _launch(swish_kernel, x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return _launch(gelu_kernel, x)


def selu(x: torch.Tensor) -> torch.Tensor:
    return _launch(selu_kernel, x)


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return _launch(hardsigmoid_kernel, x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return _launch(softplus_kernel, x)


def softsign(x: torch.Tensor) -> torch.Tensor:
    return _launch(softsign_kernel, x)


def elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _launch(elu_kernel, x, ALPHA=float(alpha))


def hardtanh(x: torch.Tensor) -> torch.Tensor:
    return _launch(hardtanh_kernel, x)


def mingpt_gelu(x: torch.Tensor) -> torch.Tensor:
    return _launch(mingpt_gelu_kernel, x)
