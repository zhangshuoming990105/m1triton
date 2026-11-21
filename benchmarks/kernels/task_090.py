"""KernelBench L1 Task #90: Cumulative product along a dimension."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

import m1triton
import m1triton.tlx as tlx

_KERNELS_DIR = str(Path(__file__).resolve().parent)
if _KERNELS_DIR not in sys.path:
    sys.path.insert(0, _KERNELS_DIR)

from _scan_common import row_scan

_COOP_MIN_ROW_LEN = 4096


@m1triton.jit(hints=[tlx.tile_spec(bm=1, bn=4096, num_simdgroups=8)])
def _cumprod_coop_row_kernel(x_ptr, out_ptr, num_rows, row_len):
    tlx.row_scan(x_ptr, out_ptr, num_rows, row_len, op="mul", exclusive=False, reverse=False)


def _device_aligned(t: torch.Tensor, nbytes: int) -> bool:
    if t.device.type == "mps":
        return (int(t.storage_offset()) * int(t.element_size())) % nbytes == 0
    return (int(t.data_ptr()) % nbytes) == 0


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0 or x.numel() == 0:
            return row_scan(x, dim=self.dim, op="mul")
        d = self.dim % x.dim()
        if int(x.shape[d]) < _COOP_MIN_ROW_LEN:
            return row_scan(x, dim=self.dim, op="mul")

        orig_dtype = x.dtype
        xs = x if orig_dtype == torch.float32 else x.to(torch.float32)
        xc = xs if d == xs.dim() - 1 else xs.movedim(d, -1)
        xc = xc if xc.is_contiguous() else xc.contiguous()
        if not _device_aligned(xc, 16):
            xc = xc.clone()

        n = int(xc.shape[-1])
        rows = xc.numel() // n
        out = torch.empty_like(xc)
        _cumprod_coop_row_kernel[(rows, 1, 1)](xc, out, rows, n)

        if d != x.dim() - 1:
            out = out.movedim(-1, d).contiguous()
        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out
