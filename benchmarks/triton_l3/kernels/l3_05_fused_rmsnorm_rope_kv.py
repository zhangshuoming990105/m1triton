"""Level 3 Kernel L3-05: Fused QK-RMSNorm + Half-Split RoPE + In-Place KV-Cache Scatter."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.runtime as m1_rt


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_H": 1}, num_warps=1),
        m1triton.Config({"BLOCK_H": 2}, num_warps=2),
        m1triton.Config({"BLOCK_H": 4}, num_warps=4),
        m1triton.Config({"BLOCK_H": 8}, num_warps=8),
        m1triton.Config({"BLOCK_H": 16}, num_warps=8),
    ],
    key=["N_tok", "H_q", "D"],
)
@m1triton.jit(fast_math=True)
def _q_rmsnorm_rope_kernel(
    q_ptr,
    qw_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    N_tok,
    H_q,
    D,
    half_d,
    inv_d,
    eps,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    hg = tl.program_id(1)
    h_offs = hg * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H_q
    d_offs = tl.arange(0, BLOCK_HALF_D)
    d_mask = d_offs < half_d
    d_i64 = D.to(tl.int64)
    half_i64 = half_d.to(tl.int64)
    tok_i64 = tok.to(tl.int64)

    row_base = tok_i64 * (H_q.to(tl.int64) * d_i64) + h_offs[:, None].to(tl.int64) * d_i64
    d_offs_i64 = d_offs[None, :].to(tl.int64)
    ptrs0 = q_ptr + row_base + d_offs_i64
    ptrs1 = q_ptr + row_base + (d_offs_i64 + half_i64)
    mask_2d = h_mask[:, None] & d_mask[None, :]

    x0 = tl.load(ptrs0, mask=mask_2d, other=0.0)
    x1 = tl.load(ptrs1, mask=mask_2d, other=0.0)
    sq_sum = tl.sum(x0 * x0 + x1 * x1, axis=1)
    inv_rms = tl.rsqrt(sq_sum * inv_d + eps)[:, None]

    w0 = tl.load(qw_ptr + d_offs, mask=d_mask, other=0.0)[None, :]
    w1 = tl.load(qw_ptr + (d_offs + half_d), mask=d_mask, other=0.0)[None, :]
    cs_base = tok_i64 * half_i64 + d_offs.to(tl.int64)
    c = tl.load(cos_ptr + cs_base, mask=d_mask, other=0.0)[None, :]
    s = tl.load(sin_ptr + cs_base, mask=d_mask, other=0.0)[None, :]

    y0 = x0 * inv_rms * w0
    y1 = x1 * inv_rms * w1
    o0 = y0 * c - y1 * s
    o1 = y0 * s + y1 * c

    out_ptrs0 = q_out_ptr + row_base + d_offs_i64
    out_ptrs1 = q_out_ptr + row_base + (d_offs_i64 + half_i64)
    tl.store(out_ptrs0, o0, mask=mask_2d)
    tl.store(out_ptrs1, o1, mask=mask_2d)


@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_H": 1}, num_warps=1),
        m1triton.Config({"BLOCK_H": 2}, num_warps=2),
        m1triton.Config({"BLOCK_H": 4}, num_warps=4),
        m1triton.Config({"BLOCK_H": 8}, num_warps=8),
    ],
    key=["N_tok", "H_kv", "D"],
    restore_value=("k_cache_ptr", "v_cache_ptr"),
)
@m1triton.jit(fast_math=True)
def _kv_rmsnorm_rope_scatter_kernel(
    k_ptr,
    v_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    slot_mapping_ptr,
    k_cache_ptr,
    v_cache_ptr,
    N_tok,
    H_kv,
    D,
    half_d,
    inv_d,
    eps,
    BLOCK_HALF_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    hg = tl.program_id(1)
    h_offs = hg * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H_kv
    d_offs = tl.arange(0, BLOCK_HALF_D)
    d_mask = d_offs < half_d
    d_i64 = D.to(tl.int64)
    half_i64 = half_d.to(tl.int64)
    tok_i64 = tok.to(tl.int64)
    stride_tok = H_kv.to(tl.int64) * d_i64

    slot = tl.load(slot_mapping_ptr + tok_i64).to(tl.int64)
    in_base = tok_i64 * stride_tok + h_offs[:, None].to(tl.int64) * d_i64
    out_base = slot * stride_tok + h_offs[:, None].to(tl.int64) * d_i64
    d_offs_i64 = d_offs[None, :].to(tl.int64)
    mask_2d = h_mask[:, None] & d_mask[None, :]

    k0 = tl.load(k_ptr + in_base + d_offs_i64, mask=mask_2d, other=0.0)
    k1 = tl.load(k_ptr + in_base + (d_offs_i64 + half_i64), mask=mask_2d, other=0.0)
    sq_sum = tl.sum(k0 * k0 + k1 * k1, axis=1)
    inv_rms = tl.rsqrt(sq_sum * inv_d + eps)[:, None]

    w0 = tl.load(kw_ptr + d_offs, mask=d_mask, other=0.0)[None, :]
    w1 = tl.load(kw_ptr + (d_offs + half_d), mask=d_mask, other=0.0)[None, :]
    cs_base = tok_i64 * half_i64 + d_offs.to(tl.int64)
    c = tl.load(cos_ptr + cs_base, mask=d_mask, other=0.0)[None, :]
    s = tl.load(sin_ptr + cs_base, mask=d_mask, other=0.0)[None, :]

    kn0 = k0 * inv_rms * w0
    kn1 = k1 * inv_rms * w1
    kr0 = kn0 * c - kn1 * s
    kr1 = kn0 * s + kn1 * c

    tl.store(k_cache_ptr + out_base + d_offs_i64, kr0, mask=mask_2d)
    tl.store(k_cache_ptr + out_base + (d_offs_i64 + half_i64), kr1, mask=mask_2d)

    v0 = tl.load(v_ptr + in_base + d_offs_i64, mask=mask_2d, other=0.0)
    v1 = tl.load(v_ptr + in_base + (d_offs_i64 + half_i64), mask=mask_2d, other=0.0)
    tl.store(v_cache_ptr + out_base + d_offs_i64, v0, mask=mask_2d)
    tl.store(v_cache_ptr + out_base + (d_offs_i64 + half_i64), v1, mask=mask_2d)


_TUNED_Q_KEYS: set[tuple[int, int, int, torch.dtype]] = set()
_TUNED_KV_KEYS: set[tuple[int, int, int, torch.dtype]] = set()


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def candidate_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK-RMSNorm + RoPE + indirect KV-cache scatter in one batched Metal submission."""
    q_c = _contig(q)
    k_c = _contig(k)
    v_c = _contig(v)
    qw_c = _contig(q_weight)
    kw_c = _contig(k_weight)
    cos_c = _contig(cos)
    sin_c = _contig(sin)
    slots_c = _contig(slot_mapping)
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise ValueError("In-place k_cache and v_cache must be contiguous tensors")

    n_tok, h_q, d = int(q_c.shape[0]), int(q_c.shape[1]), int(q_c.shape[2])
    h_kv = int(k_c.shape[1])
    half_d = d // 2
    block_half_d = m1triton.next_power_of_2(half_d)
    inv_d = 1.0 / float(d)
    eps_f = float(eps)

    q_out = torch.empty_like(q_c)

    q_grid = lambda meta: (n_tok, tl.cdiv(h_q, meta["BLOCK_H"]))
    kv_grid = lambda meta: (n_tok, tl.cdiv(h_kv, meta["BLOCK_H"]))

    q_key = (n_tok, h_q, d, q_c.dtype)
    if q_key not in _TUNED_Q_KEYS:
        _q_rmsnorm_rope_kernel._select_config(
            q_grid,
            (q_c, qw_c, cos_c, sin_c, q_out, n_tok, h_q, d, half_d, inv_d, eps_f),
            {"BLOCK_HALF_D": block_half_d},
        )
        _TUNED_Q_KEYS.add(q_key)

    kv_key = (n_tok, h_kv, d, k_c.dtype)
    if kv_key not in _TUNED_KV_KEYS:
        _kv_rmsnorm_rope_scatter_kernel._select_config(
            kv_grid,
            (k_c, v_c, kw_c, cos_c, sin_c, slots_c, k_cache, v_cache, n_tok, h_kv, d, half_d, inv_d, eps_f),
            {"BLOCK_HALF_D": block_half_d},
        )
        _TUNED_KV_KEYS.add(kv_key)

    with m1_rt.command_batch(wait=True, sync_mps=True):
        _q_rmsnorm_rope_kernel[q_grid](
            q_c,
            qw_c,
            cos_c,
            sin_c,
            q_out,
            n_tok,
            h_q,
            d,
            half_d,
            inv_d,
            eps_f,
            BLOCK_HALF_D=block_half_d,
            sync_mps=False,
            wait=False,
        )
        _kv_rmsnorm_rope_scatter_kernel[kv_grid](
            k_c,
            v_c,
            kw_c,
            cos_c,
            sin_c,
            slots_c,
            k_cache,
            v_cache,
            n_tok,
            h_kv,
            d,
            half_d,
            inv_d,
            eps_f,
            BLOCK_HALF_D=block_half_d,
            sync_mps=False,
            wait=False,
        )

    return q_out, k_cache, v_cache
