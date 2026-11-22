"""Level 3 Spec L3-07: Chunked Fused Linear + Cross-Entropy Forward & Backward."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

TASK_NAME = "chunked_linear_cross_entropy"
SUPPORTS_BACKWARD = True
FWD_TOL = (1e-3, 1e-3)
BWD_TOL = (2e-3, 2e-3)

DEFAULT_BT = 2048
DEFAULT_H = 1024
DEFAULT_V = 32768


def get_inputs(
    seed: int,
    device: torch.device,
    *,
    bt: int = DEFAULT_BT,
    hidden: int = DEFAULT_H,
    vocab: int = DEFAULT_V,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, Any]]:
    """Generate deterministic LLM LM-head inputs (x, weight, targets) on target device."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((bt, hidden), generator=gen, dtype=torch.float32).to(device=device).requires_grad_(True)
    w_scale = 1.0 / math.sqrt(float(hidden))
    weight = (
        (torch.randn((vocab, hidden), generator=gen, dtype=torch.float32) * w_scale)
        .to(device=device)
        .requires_grad_(True)
    )
    targets = torch.randint(0, vocab, (bt,), generator=gen, dtype=torch.int64).to(device=device)
    return (x, weight, targets), {}


def reference_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch LM-head projection followed by mean cross-entropy loss."""
    logits = torch.matmul(x, weight.t())
    return F.cross_entropy(logits, targets, reduction="mean")
