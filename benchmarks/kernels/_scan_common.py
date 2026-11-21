"""Reusable M1Triton hardware row-scan kernels (cumsum / cumprod / reverse / exclusive / masked)."""

from __future__ import annotations

import torch

import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

ROWS_PER_THREADGROUP = 8


@m1triton.jit
def row_scan_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    row_len,
    OP: tl.constexpr = "add",
    EXCLUSIVE: tl.constexpr = False,
    REVERSE: tl.constexpr = False,
):
    tlx.row_scan(x_ptr, out_ptr, num_rows, row_len, op=OP, exclusive=EXCLUSIVE, reverse=REVERSE)


@m1triton.jit
def masked_row_scan_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    num_rows,
    row_len,
    OP: tl.constexpr = "add",
    EXCLUSIVE: tl.constexpr = False,
    REVERSE: tl.constexpr = False,
):
    tlx.row_scan(
        x_ptr,
        out_ptr,
        num_rows,
        row_len,
        op=OP,
        exclusive=EXCLUSIVE,
        reverse=REVERSE,
        mask_ptr=mask_ptr,
    )


def _device_aligned(t: torch.Tensor, nbytes: int) -> bool:
    if t.device.type == "mps":
        return (int(t.storage_offset()) * int(t.element_size())) % nbytes == 0
    return (int(t.data_ptr()) % nbytes) == 0


def _rows_last(t: torch.Tensor, dim: int, nbytes: int) -> torch.Tensor:
    moved = t if dim == t.dim() - 1 else t.movedim(dim, -1)
    moved = moved.contiguous()
    if not _device_aligned(moved, nbytes):
        moved = moved.clone()
    return moved


def row_scan(
    x: torch.Tensor,
    *,
    dim: int = -1,
    op: str = "add",
    exclusive: bool = False,
    reverse: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefix scan along `dim`; the last dim of a contiguous tensor is dispatched without any copy."""
    if x.dim() == 0:
        raise ValueError("row_scan expects a tensor with at least one dimension")
    d = dim % x.dim()
    compute_dtype = x.dtype
    xs = x
    if x.dtype != torch.float32:
        xs = x.to(torch.float32)
    xc = _rows_last(xs, d, 16)
    out = torch.empty_like(xc)

    n = int(xc.shape[-1])
    needs_post_mps = d != x.dim() - 1 or compute_dtype != torch.float32
    if xc.numel() > 0:
        rows = xc.numel() // n
        flags = {"OP": op, "EXCLUSIVE": bool(exclusive), "REVERSE": bool(reverse)}
        if mask is None:
            row_scan_kernel[(m1triton.cdiv(rows, ROWS_PER_THREADGROUP), 1, 1)](
                xc, out, rows, n, wait=needs_post_mps, **flags
            )
        else:
            mc = mask if mask.dtype == torch.bool else mask.to(torch.bool)
            mc = _rows_last(mc.expand_as(x), d, 4)
            masked_row_scan_kernel[(m1triton.cdiv(rows, ROWS_PER_THREADGROUP), 1, 1)](
                xc, out, mc, rows, n, wait=needs_post_mps, **flags
            )

    if d != x.dim() - 1:
        out = out.movedim(-1, d).contiguous()
    if compute_dtype != torch.float32:
        out = out.to(compute_dtype)
    return out


__all__ = ["masked_row_scan_kernel", "row_scan", "row_scan_kernel"]
