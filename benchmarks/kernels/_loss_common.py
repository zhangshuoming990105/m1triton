"""Reusable M1Triton two-stage fused reduction loss kernels for Apple M1 Max."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

_TG_THREADS = 256
_FLAT_STAGE1_MAX_TGS = 2048
_ROW_STAGE1_MAX_TGS = 2048
_HINGE_STAGE1_MAX_TGS = 1024


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _scratch(n: int, like: torch.Tensor) -> torch.Tensor:
    return torch.empty((max(int(n), 1),), device=like.device, dtype=torch.float32)


@m1triton.jit
def loss_stage1_flat_kernel(
    pred_ptr,
    tgt_ptr,
    partial_ptr,
    total,
    KIND: tl.constexpr,
    BETA: tl.constexpr,
):
    tlx.fused_loss(KIND, pred_ptr, tgt_ptr, partial_ptr, total, beta=BETA, stage=1)


@m1triton.jit
def loss_stage1_hinge_kernel(
    pred_ptr,
    tgt_ptr,
    partial_ptr,
    rows,
    cols,
):
    tlx.fused_loss("hinge", pred_ptr, tgt_ptr, partial_ptr, rows, cols, stage=1)


@m1triton.jit
def loss_stage1_ce_kernel(
    pred_ptr,
    tgt_ptr,
    partial_ptr,
    batch_size,
    num_classes,
):
    tlx.fused_loss("cross_entropy", pred_ptr, tgt_ptr, partial_ptr, batch_size, num_classes, stage=1)


@m1triton.jit
def loss_stage1_triplet_kernel(
    anchor_ptr,
    pos_ptr,
    neg_ptr,
    partial_ptr,
    batch_size,
    dim_size,
    MARGIN: tl.constexpr,
    EPS: tl.constexpr,
):
    tlx.fused_loss(
        "triplet_margin",
        anchor_ptr,
        pos_ptr,
        neg_ptr,
        partial_ptr,
        batch_size,
        dim_size,
        margin=MARGIN,
        eps=EPS,
        stage=1,
    )


@m1triton.jit
def loss_stage2_kernel(
    partial_ptr,
    out_ptr,
    num_partials,
    denom,
    KIND: tl.constexpr,
):
    tlx.fused_loss(KIND, partial_ptr, out_ptr, num_partials, denom, stage=2)


def _finalize_scalar(partial: torch.Tensor, out: torch.Tensor, num_partials: int, denom: float, kind: str) -> torch.Tensor:
    loss_stage2_kernel[(1, 1, 1)](partial, out, int(num_partials), float(denom), KIND=kind, sync_mps=False)
    return out.reshape(())


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error over arbitrary contiguous shapes via two-stage fused reduction."""
    pred = _contig(predictions)
    tgt = _contig(targets)
    total = int(pred.numel())
    g1 = max(1, min(_FLAT_STAGE1_MAX_TGS, _cdiv(total, _TG_THREADS * 16)))
    partial = _scratch(g1, pred)
    out = _scratch(1, pred)
    loss_stage1_flat_kernel[(g1, 1, 1)](pred, tgt, partial, total, KIND="mse", BETA=1.0)
    return _finalize_scalar(partial, out, g1, float(total), "mse")


def huber_loss(predictions: torch.Tensor, targets: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    """Smooth L1 (Huber) loss with mean reduction via two-stage fused reduction."""
    pred = _contig(predictions)
    tgt = _contig(targets)
    total = int(pred.numel())
    g1 = max(1, min(_FLAT_STAGE1_MAX_TGS, _cdiv(total, _TG_THREADS * 16)))
    partial = _scratch(g1, pred)
    out = _scratch(1, pred)
    loss_stage1_flat_kernel[(g1, 1, 1)](pred, tgt, partial, total, KIND="huber", BETA=float(beta))
    return _finalize_scalar(partial, out, g1, float(total), "huber")


def kl_div_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused KL divergence on probability inputs with batchmean reduction."""
    pred = _contig(predictions)
    tgt = _contig(targets)
    total = int(pred.numel())
    batch_denom = float(pred.shape[0]) if pred.ndim > 0 else 1.0
    g1 = max(1, min(_FLAT_STAGE1_MAX_TGS, _cdiv(total, _TG_THREADS * 16)))
    partial = _scratch(g1, pred)
    out = _scratch(1, pred)
    loss_stage1_flat_kernel[(g1, 1, 1)](pred, tgt, partial, total, KIND="kl_div", BETA=1.0)
    return _finalize_scalar(partial, out, g1, batch_denom, "kl_div")


def hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean hinge loss supporting trailing-dim target broadcast and elementwise shapes."""
    pred = _contig(predictions)
    tgt = targets
    if tgt.shape == pred.shape:
        tgt = _contig(tgt)
        rows = 1
        cols = int(pred.numel())
    elif (
        pred.ndim >= 2
        and 1 <= tgt.ndim <= pred.ndim
        and int(tgt.shape[-1]) == int(pred.shape[-1])
        and all(int(d) == 1 for d in tgt.shape[:-1])
    ):
        tgt = _contig(tgt)
        cols = int(pred.shape[-1])
        rows = int(pred.numel()) // max(cols, 1)
    else:
        bshape = torch.broadcast_shapes(pred.shape, tgt.shape)
        pred = pred.expand(bshape).contiguous()
        tgt = tgt.expand(bshape).contiguous()
        rows = 1
        cols = int(pred.numel())

    total = rows * cols
    if rows >= _TG_THREADS:
        g1 = max(1, min(_HINGE_STAGE1_MAX_TGS, rows))
    else:
        g1 = max(1, min(_HINGE_STAGE1_MAX_TGS, _cdiv(cols, _TG_THREADS * 16)))
    partial = _scratch(g1, pred)
    out = _scratch(1, pred)
    loss_stage1_hinge_kernel[(g1, 1, 1)](pred, tgt, partial, rows, cols)
    return _finalize_scalar(partial, out, g1, float(total), "hinge")


def cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused online-softmax cross entropy loss over (B, C) logits and (B,) int64 targets."""
    pred = _contig(predictions)
    tgt = _contig(targets)
    if tgt.dtype not in (torch.int64, torch.int32):
        tgt = tgt.to(dtype=torch.int64)
    if pred.ndim == 1:
        batch_size = 1
        num_classes = int(pred.shape[0])
    else:
        batch_size = int(pred.shape[0])
        num_classes = int(pred.shape[1])
    g1 = max(1, min(_ROW_STAGE1_MAX_TGS, batch_size))
    partial = _scratch(g1, pred)
    out = _scratch(1, pred)
    loss_stage1_ce_kernel[(g1, 1, 1)](pred, tgt, partial, batch_size, num_classes)
    return _finalize_scalar(partial, out, g1, float(batch_size), "cross_entropy")


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused L2 triplet margin loss matching pairwise distance eps semantics."""
    anc = _contig(anchor)
    pos = _contig(positive)
    neg = _contig(negative)
    if anc.ndim == 1:
        batch_size = 1
        dim_size = int(anc.shape[0])
    else:
        dim_size = int(anc.shape[-1])
        batch_size = int(anc.numel()) // max(dim_size, 1)
    g1 = max(1, min(_ROW_STAGE1_MAX_TGS, batch_size))
    partial = _scratch(g1, anc)
    out = _scratch(1, anc)
    loss_stage1_triplet_kernel[(g1, 1, 1)](
        anc, pos, neg, partial, batch_size, dim_size, MARGIN=float(margin), EPS=float(eps)
    )
    return _finalize_scalar(partial, out, g1, float(batch_size), "triplet_margin")


__all__ = [
    "cross_entropy_loss",
    "hinge_loss",
    "huber_loss",
    "kl_div_loss",
    "loss_stage1_ce_kernel",
    "loss_stage1_flat_kernel",
    "loss_stage1_hinge_kernel",
    "loss_stage1_triplet_kernel",
    "loss_stage2_kernel",
    "mse_loss",
    "triplet_margin_loss",
]
