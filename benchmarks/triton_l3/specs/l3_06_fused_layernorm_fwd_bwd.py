"""Level 3 Spec L3-06: Fused LayerNorm Forward + Backward with Atomic Parameter Gradients."""

from __future__ import annotations

import torch
import torch.nn.functional as F

TASK_NAME = "fused_layernorm_fwd_bwd"
SUPPORTS_BACKWARD = True
FWD_TOL = (1e-3, 1e-3)
BWD_TOL = (2e-3, 2e-3)


def get_inputs(
    seed: int = 0,
    device: torch.device | str = "mps",
    *,
    m: int = 2048,
    n: int = 4096,
    eps: float = 1e-5,
) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    """Create deterministic LLM Transformer hidden LayerNorm forward/backward inputs."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    x = torch.randn((m, n), generator=gen, dtype=torch.float32).to(device).requires_grad_(True)
    weight = (1.0 + 0.1 * torch.randn((n,), generator=gen, dtype=torch.float32)).to(device).requires_grad_(True)
    bias = (0.1 * torch.randn((n,), generator=gen, dtype=torch.float32)).to(device).requires_grad_(True)
    return (x, weight, bias), {"eps": float(eps)}


def reference_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch reference LayerNorm over trailing feature dimension."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
