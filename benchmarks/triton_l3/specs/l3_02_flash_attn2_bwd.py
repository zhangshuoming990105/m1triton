"""L3-02 Reference Spec: Causal FlashAttention-2 Forward + Backward (B=2, H=8, L=512, D=64, FP32)."""

from __future__ import annotations

import math
from typing import Any

import torch

TASK_NAME = "flash_attn2_fwd_bwd_causal"
SUPPORTS_BACKWARD = True
FWD_TOL = (1e-3, 1e-3)
BWD_TOL = (2e-3, 2e-3)


def get_inputs(seed: int, device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    b, h, seq_len, d = 2, 8, 512, 64
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device).requires_grad_(True)
    k = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device).requires_grad_(True)
    v = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device).requires_grad_(True)
    return (q, k, v), {"is_causal": True}


def reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
