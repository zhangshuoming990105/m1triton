"""Vectorized elementwise activation kernels for KernelBench L1 (relu .. hardtanh, MinGPT GELU)."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

BLOCK_SIZE = 1024
_FAST_UNROLL = 4
_FAST_TILE = BLOCK_SIZE * _FAST_UNROLL
_INT32_FAST_LIMIT = 2_000_000_000


@m1triton.jit
def relu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x < 0.0, 0.0, x), mask=mask)


@m1triton.jit
def relu_fast4_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    base = tl.program_id(0) * (BLOCK * 4) + tl.arange(0, BLOCK)
    x0 = tl.load(x_ptr + base)
    x1 = tl.load(x_ptr + (base + BLOCK))
    x2 = tl.load(x_ptr + (base + BLOCK * 2))
    x3 = tl.load(x_ptr + (base + BLOCK * 3))
    tl.store(o_ptr + base, tl.where(x0 < 0.0, 0.0, x0))
    tl.store(o_ptr + (base + BLOCK), tl.where(x1 < 0.0, 0.0, x1))
    tl.store(o_ptr + (base + BLOCK * 2), tl.where(x2 < 0.0, 0.0, x2))
    tl.store(o_ptr + (base + BLOCK * 3), tl.where(x3 < 0.0, 0.0, x3))


@m1triton.jit
def leaky_relu_kernel(x_ptr, o_ptr, n, SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 0.0, x, x * SLOPE), mask=mask)


@m1triton.jit
def leaky_relu_fast4_kernel(x_ptr, o_ptr, SLOPE: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    base = tl.program_id(0) * (BLOCK * 4) + tl.arange(0, BLOCK)
    x0 = tl.load(x_ptr + base)
    x1 = tl.load(x_ptr + (base + BLOCK))
    x2 = tl.load(x_ptr + (base + BLOCK * 2))
    x3 = tl.load(x_ptr + (base + BLOCK * 3))
    tl.store(o_ptr + base, tl.where(x0 > 0.0, x0, x0 * SLOPE))
    tl.store(o_ptr + (base + BLOCK), tl.where(x1 > 0.0, x1, x1 * SLOPE))
    tl.store(o_ptr + (base + BLOCK * 2), tl.where(x2 > 0.0, x2, x2 * SLOPE))
    tl.store(o_ptr + (base + BLOCK * 3), tl.where(x3 > 0.0, x3, x3 * SLOPE))


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
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    pos = 1.0507009873554804934193349852946 * x
    neg = 1.7580993408473768599402175208123 * (tl.exp(tl.where(x > 0.0, 0.0, x)) - 1.0)
    tl.store(o_ptr + offs, tl.where(x > 0.0, pos, neg), mask=mask)


@m1triton.jit
def selu_fast4_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    base = tl.program_id(0) * (BLOCK * 4) + tl.arange(0, BLOCK)
    x0 = tl.load(x_ptr + base)
    x1 = tl.load(x_ptr + (base + BLOCK))
    x2 = tl.load(x_ptr + (base + BLOCK * 2))
    x3 = tl.load(x_ptr + (base + BLOCK * 3))
    y0 = tl.where(
        x0 > 0.0,
        1.0507009873554804934193349852946 * x0,
        1.7580993408473768599402175208123 * (tl.exp(tl.where(x0 > 0.0, 0.0, x0)) - 1.0),
    )
    y1 = tl.where(
        x1 > 0.0,
        1.0507009873554804934193349852946 * x1,
        1.7580993408473768599402175208123 * (tl.exp(tl.where(x1 > 0.0, 0.0, x1)) - 1.0),
    )
    y2 = tl.where(
        x2 > 0.0,
        1.0507009873554804934193349852946 * x2,
        1.7580993408473768599402175208123 * (tl.exp(tl.where(x2 > 0.0, 0.0, x2)) - 1.0),
    )
    y3 = tl.where(
        x3 > 0.0,
        1.0507009873554804934193349852946 * x3,
        1.7580993408473768599402175208123 * (tl.exp(tl.where(x3 > 0.0, 0.0, x3)) - 1.0),
    )
    tl.store(o_ptr + base, y0)
    tl.store(o_ptr + (base + BLOCK), y1)
    tl.store(o_ptr + (base + BLOCK * 2), y2)
    tl.store(o_ptr + (base + BLOCK * 3), y3)


@m1triton.jit
def hardsigmoid_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.where(x <= -3.0, 0.0, tl.where(x >= 3.0, 1.0, (x + 3.0) * (1.0 / 6.0)))
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
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x > 0.0, x, ALPHA * (tl.exp(tl.where(x > 0.0, 0.0, x)) - 1.0)), mask=mask)


@m1triton.jit
def elu_fast4_kernel(x_ptr, o_ptr, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    base = tl.program_id(0) * (BLOCK * 4) + tl.arange(0, BLOCK)
    x0 = tl.load(x_ptr + base)
    x1 = tl.load(x_ptr + (base + BLOCK))
    x2 = tl.load(x_ptr + (base + BLOCK * 2))
    x3 = tl.load(x_ptr + (base + BLOCK * 3))
    tl.store(o_ptr + base, tl.where(x0 > 0.0, x0, ALPHA * (tl.exp(tl.where(x0 > 0.0, 0.0, x0)) - 1.0)))
    tl.store(o_ptr + (base + BLOCK), tl.where(x1 > 0.0, x1, ALPHA * (tl.exp(tl.where(x1 > 0.0, 0.0, x1)) - 1.0)))
    tl.store(o_ptr + (base + BLOCK * 2), tl.where(x2 > 0.0, x2, ALPHA * (tl.exp(tl.where(x2 > 0.0, 0.0, x2)) - 1.0)))
    tl.store(o_ptr + (base + BLOCK * 3), tl.where(x3 > 0.0, x3, ALPHA * (tl.exp(tl.where(x3 > 0.0, 0.0, x3)) - 1.0)))


@m1triton.jit
def hardtanh_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, tl.where(x < -1.0, -1.0, tl.where(x > 1.0, 1.0, x)), mask=mask)


@m1triton.jit
def hardtanh_fast4_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    base = tl.program_id(0) * (BLOCK * 4) + tl.arange(0, BLOCK)
    x0 = tl.load(x_ptr + base)
    x1 = tl.load(x_ptr + (base + BLOCK))
    x2 = tl.load(x_ptr + (base + BLOCK * 2))
    x3 = tl.load(x_ptr + (base + BLOCK * 3))
    tl.store(o_ptr + base, tl.where(x0 < -1.0, -1.0, tl.where(x0 > 1.0, 1.0, x0)))
    tl.store(o_ptr + (base + BLOCK), tl.where(x1 < -1.0, -1.0, tl.where(x1 > 1.0, 1.0, x1)))
    tl.store(o_ptr + (base + BLOCK * 2), tl.where(x2 < -1.0, -1.0, tl.where(x2 > 1.0, 1.0, x2)))
    tl.store(o_ptr + (base + BLOCK * 3), tl.where(x3 < -1.0, -1.0, tl.where(x3 > 1.0, 1.0, x3)))


@m1triton.jit
def mingpt_gelu_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    tlx.vec_width(4)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    inner = 0.7978845608028654 * (x + 0.044715 * x ** 3.0)
    tl.store(o_ptr + offs, 0.5 * x * (1.0 + tl.tanh(inner)), mask=mask)


def _launch(kernel, x: torch.Tensor, **consts) -> torch.Tensor:
    x = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (m1triton.cdiv(n, BLOCK_SIZE),)
    kernel[grid](x, out, n, BLOCK=BLOCK_SIZE, **consts)
    return out


def _launch_fast4(fast_kernel, masked_kernel, x: torch.Tensor, **consts) -> torch.Tensor:
    x = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0 and (n % _FAST_TILE) == 0 and n <= _INT32_FAST_LIMIT:
        fast_kernel[(n // _FAST_TILE,)](x, out, BLOCK=BLOCK_SIZE, **consts)
    else:
        masked_kernel[(m1triton.cdiv(n, BLOCK_SIZE),)](x, out, n, BLOCK=BLOCK_SIZE, **consts)
    return out


def relu(x: torch.Tensor) -> torch.Tensor:
    return _launch_fast4(relu_fast4_kernel, relu_kernel, x)


def leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return _launch_fast4(leaky_relu_fast4_kernel, leaky_relu_kernel, x, SLOPE=float(negative_slope))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return _launch(sigmoid_kernel, x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return _launch(tanh_kernel, x)


def swish(x: torch.Tensor) -> torch.Tensor:
    return _launch(swish_kernel, x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return _launch(gelu_kernel, x)


def selu(x: torch.Tensor) -> torch.Tensor:
    return _launch_fast4(selu_fast4_kernel, selu_kernel, x)


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return _launch(hardsigmoid_kernel, x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return _launch(softplus_kernel, x)


def softsign(x: torch.Tensor) -> torch.Tensor:
    return _launch(softsign_kernel, x)


def elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _launch_fast4(elu_fast4_kernel, elu_kernel, x, ALPHA=float(alpha))


def hardtanh(x: torch.Tensor) -> torch.Tensor:
    return _launch_fast4(hardtanh_fast4_kernel, hardtanh_kernel, x)


def mingpt_gelu(x: torch.Tensor) -> torch.Tensor:
    return _launch(mingpt_gelu_kernel, x)
