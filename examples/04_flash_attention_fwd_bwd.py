"""Example 04: End-to-end FlashAttention-2 Forward + Backward (`torch.autograd.grad`) on Apple Silicon."""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as rt
import m1triton.tlx as tlx


@m1triton.jit
def _fa2_fwd_lse_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 4,
):
    tlx.flash_attention_tile(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
        lse_ptr=lse_ptr,
    )


@m1triton.jit
def _fa2_delta_kernel(
    do_ptr,
    o_ptr,
    delta_ptr,
    D,
    BLOCK_D: tl.constexpr = 64,
):
    row = tl.program_id(0).to(tl.int64)
    d_i64 = D.to(tl.int64)
    row_base = row * d_i64
    ar = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(0, D, BLOCK_D):
        k_offs = k + ar
        mask = k_offs < D
        do_v = tl.load(do_ptr + row_base + k_offs.to(tl.int64), mask=mask, other=0.0)
        o_v = tl.load(o_ptr + row_base + k_offs.to(tl.int64), mask=mask, other=0.0)
        acc = acc + do_v * o_v
    tl.store(delta_ptr + row, tl.sum(acc, axis=0))


@m1triton.jit
def _fa2_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 4,
):
    tlx.flash_attention_bwd_dkdv(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
    )


@m1triton.jit
def _fa2_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    BH,
    L_q,
    L_kv,
    D,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 4,
):
    tlx.flash_attention_bwd_dq(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        delta_ptr,
        dq_ptr,
        BH,
        L_q,
        L_kv,
        D,
        scale=SCALE,
        causal=CAUSAL,
    )


class M1FlashAttention2(torch.autograd.Function):
    """Differentiable O(N)-memory FlashAttention-2 using M1Triton forward & backward Metal kernels."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
        q_c, k_c, v_c = q.contiguous(), k.contiguous(), v.contiguous()
        b, h, l_q, d = q_c.shape
        l_kv = k_c.shape[2]
        bh = b * h
        scale = 1.0 / math.sqrt(float(d))

        out = torch.empty_like(q_c)
        lse = torch.empty((bh, l_q), device=q_c.device, dtype=torch.float32)
        need_bwd = any(ctx.needs_input_grad[:3])
        _fa2_fwd_lse_kernel[(m1triton.cdiv(l_q, 32), bh, 1)](
            q_c,
            k_c,
            v_c,
            out,
            lse,
            bh,
            l_q,
            l_kv,
            d,
            SCALE=scale,
            CAUSAL=bool(causal),
            BLOCK_M=32,
            BLOCK_N=32,
            NUM_SG=2,
            wait=not need_bwd,
            sync_mps=True,
        )
        ctx.save_for_backward(q_c, k_c, v_c, out, lse)
        ctx.scale = scale
        ctx.causal = bool(causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_c, k_c, v_c, out, lse = ctx.saved_tensors
        do_c = grad_out.contiguous()
        b, h, l_q, d = q_c.shape
        l_kv = k_c.shape[2]
        bh = b * h
        scale = ctx.scale
        causal = ctx.causal

        delta = torch.empty((bh, l_q), device=q_c.device, dtype=torch.float32)
        dq = torch.empty_like(q_c)
        dk = torch.empty_like(k_c)
        dv = torch.empty_like(v_c)

        with rt.command_batch(wait=True, sync_mps=True):
            _fa2_delta_kernel[(bh * l_q, 1, 1)](do_c, out, delta, d, BLOCK_D=64, wait=False, sync_mps=False)
            _fa2_bwd_dkdv_kernel[(m1triton.cdiv(l_kv, 32), bh, 1)](
                q_c,
                k_c,
                v_c,
                do_c,
                lse,
                delta,
                dk,
                dv,
                bh,
                l_q,
                l_kv,
                d,
                SCALE=scale,
                CAUSAL=causal,
                BLOCK_M=32,
                BLOCK_N=32,
                NUM_SG=4,
                wait=False,
                sync_mps=False,
            )
            _fa2_bwd_dq_kernel[(m1triton.cdiv(l_q, 64), bh, 1)](
                q_c,
                k_c,
                v_c,
                do_c,
                lse,
                delta,
                dq,
                bh,
                l_q,
                l_kv,
                d,
                SCALE=scale,
                CAUSAL=causal,
                BLOCK_M=64,
                BLOCK_N=32,
                NUM_SG=8,
                wait=False,
                sync_mps=False,
            )
        return dq, dk, dv, None


def flash_attention2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    return M1FlashAttention2.apply(q, k, v, bool(causal))


def eager_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        l = q.shape[-2]
        mask = torch.triu(torch.ones((l, l), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple Silicon MPS backend is required to run this example.")

    b, h, seq_len, d = 2, 8, 512, 64
    device = torch.device("mps")
    gen = torch.Generator(device="cpu").manual_seed(42)

    q_ref = torch.randn((b, h, seq_len, d), generator=gen).to(device).requires_grad_(True)
    k_ref = torch.randn((b, h, seq_len, d), generator=gen).to(device).requires_grad_(True)
    v_ref = torch.randn((b, h, seq_len, d), generator=gen).to(device).requires_grad_(True)
    grad_out = torch.randn((b, h, seq_len, d), generator=gen).to(device)

    q_m1 = q_ref.detach().clone().requires_grad_(True)
    k_m1 = k_ref.detach().clone().requires_grad_(True)
    v_m1 = v_ref.detach().clone().requires_grad_(True)

    out_ref = eager_reference(q_ref, k_ref, v_ref, causal=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_outputs=grad_out)
    torch.mps.synchronize()

    out_m1 = flash_attention2(q_m1, k_m1, v_m1, causal=True)
    dq_m1, dk_m1, dv_m1 = torch.autograd.grad(out_m1, (q_m1, k_m1, v_m1), grad_outputs=grad_out)
    torch.mps.synchronize()
    rt.synchronize()

    torch.testing.assert_close(out_m1.cpu(), out_ref.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dq_m1.cpu(), dq_ref.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(dk_m1.cpu(), dk_ref.cpu(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(dv_m1.cpu(), dv_ref.cpu(), rtol=2e-3, atol=2e-3)

    fwd_err = float((out_m1.cpu() - out_ref.cpu()).abs().max().item())
    dq_err = float((dq_m1.cpu() - dq_ref.cpu()).abs().max().item())
    dk_err = float((dk_m1.cpu() - dk_ref.cpu()).abs().max().item())
    dv_err = float((dv_m1.cpu() - dv_ref.cpu()).abs().max().item())

    for _ in range(3):
        o = eager_reference(q_ref, k_ref, v_ref, causal=True)
        torch.autograd.grad(o, (q_ref, k_ref, v_ref), grad_outputs=grad_out)
        o2 = flash_attention2(q_m1, k_m1, v_m1, causal=True)
        torch.autograd.grad(o2, (q_m1, k_m1, v_m1), grad_outputs=grad_out)
    torch.mps.synchronize()
    rt.synchronize()

    ref_times: list[float] = []
    m1_times: list[float] = []
    for _ in range(10):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        o = eager_reference(q_ref, k_ref, v_ref, causal=True)
        torch.autograd.grad(o, (q_ref, k_ref, v_ref), grad_outputs=grad_out)
        torch.mps.synchronize()
        ref_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        o2 = flash_attention2(q_m1, k_m1, v_m1, causal=True)
        torch.autograd.grad(o2, (q_m1, k_m1, v_m1), grad_outputs=grad_out)
        torch.mps.synchronize()
        rt.synchronize()
        m1_times.append((time.perf_counter() - t0) * 1000.0)

    ref_med = statistics.median(ref_times)
    m1_med = statistics.median(m1_times)
    dev = rt.device_info()
    print(f"Device       : {dev.name}")
    print(f"Shape        : B={b}, H={h}, L={seq_len}, D={d} float32 causal (zero N*N materialization)")
    print(f"Correctness  : PASS (out={fwd_err:.2e}, dQ={dq_err:.2e}, dK={dk_err:.2e}, dV={dv_err:.2e})")
    print(f"PyTorch MPS  : {ref_med:.3f} ms (fwd + bwd)")
    print(f"M1Triton FA2 : {m1_med:.3f} ms (fwd + bwd, {ref_med / m1_med:.2f}x speedup)")


if __name__ == "__main__":
    main()
