"""L3-01 Reference Spec: Causal GQA FlashAttention-2 Forward (B=4, H_q=16, H_kv=4, L=1024, D=64, FP32)."""

from __future__ import annotations

import math
from typing import Any

import torch

TASK_NAME = "flash_attn2_fwd_gqa_causal"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)


def get_inputs(seed: int, device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    b, h_q, h_kv, seq_len, d = 4, 16, 4, 1024, 64
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((b, h_q, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    k = torch.randn((b, h_kv, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    v = torch.randn((b, h_kv, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    return (q, k, v), {"is_causal": True}


def reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    kv_group_size = q.shape[1] // k.shape[1]
    k_exp = k.repeat_interleave(kv_group_size, dim=1) if kv_group_size > 1 else k
    v_exp = v.repeat_interleave(kv_group_size, dim=1) if kv_group_size > 1 else v
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_exp)
