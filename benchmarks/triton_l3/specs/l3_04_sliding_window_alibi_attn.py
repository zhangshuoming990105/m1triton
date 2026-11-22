"""L3-04 Reference Spec: Sliding-Window (W=128) + ALiBi + Logit-Softcap (50.0) Causal Attention."""

from __future__ import annotations

import math
from typing import Any

import torch

TASK_NAME = "sliding_window_alibi_softcap_attn"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    ratio = 2.0 ** (-8.0 / float(num_heads))
    return torch.tensor([ratio ** (i + 1) for i in range(num_heads)], dtype=torch.float32)


def get_inputs(seed: int, device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    b, h, seq_len, d = 2, 8, 1024, 64
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    k = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    v = torch.randn((b, h, seq_len, d), generator=gen, dtype=torch.float32).to(device)
    slopes = _alibi_slopes(h).to(device)
    return (q, k, v, slopes), {"window_size": 128, "softcap": 50.0}


def reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alibi_slopes: torch.Tensor,
    *,
    window_size: int = 128,
    softcap: float = 50.0,
) -> torch.Tensor:
    _, h, seq_len, d = q.shape
    scale = 1.0 / math.sqrt(float(d))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    q_pos = torch.arange(seq_len, device=q.device, dtype=torch.float32).view(1, 1, seq_len, 1)
    k_pos = torch.arange(seq_len, device=q.device, dtype=torch.float32).view(1, 1, 1, seq_len)
    scores = scores + alibi_slopes.view(1, h, 1, 1) * (k_pos - q_pos)
    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    valid = (k_pos <= q_pos) & (k_pos >= (q_pos - float(window_size)))
    scores = scores.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
