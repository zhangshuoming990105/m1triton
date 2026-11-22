"""Level 3 Spec L3-08: Mamba / Selective State-Space Scan (SSM Forward)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

TASK_NAME = "mamba_ssm_scan"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)

DEFAULT_B = 8
DEFAULT_L = 512
DEFAULT_D = 512
DEFAULT_N_STATE = 16


def get_inputs(
    seed: int = 0,
    device: torch.device | str = "mps",
    *,
    B: int = DEFAULT_B,
    L: int = DEFAULT_L,
    D: int = DEFAULT_D,
    N_STATE: int = DEFAULT_N_STATE,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Generate deterministic Mamba-1 / Mamba-2 selective state-space scan inputs."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    dev = torch.device(device)

    u = (torch.randn((B, L, D), generator=gen, dtype=torch.float32) * 0.5).to(dev).contiguous()
    dt = torch.randn((B, L, D), generator=gen, dtype=torch.float32).to(dev).contiguous()

    # Negative S4D-Real multi-scale log-decay spectrum across state slots 1..N_STATE.
    a_scales = torch.arange(1, N_STATE + 1, dtype=torch.float32).unsqueeze(0)
    a_jitter = 0.5 + 0.5 * torch.rand((D, N_STATE), generator=gen, dtype=torch.float32)
    A = (-(a_scales * a_jitter)).to(dev).contiguous()

    B_proj = (torch.randn((B, L, N_STATE), generator=gen, dtype=torch.float32) * 0.5).to(dev).contiguous()
    C_proj = (torch.randn((B, L, N_STATE), generator=gen, dtype=torch.float32) * 0.5).to(dev).contiguous()
    D_skip = torch.randn((D,), generator=gen, dtype=torch.float32).to(dev).contiguous()
    dt_bias = (-3.0 + 2.0 * torch.rand((D,), generator=gen, dtype=torch.float32)).to(dev).contiguous()

    return (u, dt, A, B_proj, C_proj, D_skip, dt_bias), {}


def reference_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    D_skip: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference for selective state-space scan (Mamba SSM forward)."""
    B, L, D = u.shape
    N = A.shape[1]

    delta = F.softplus(dt + dt_bias)
    h = torch.zeros((B, D, N), dtype=u.dtype, device=u.device)
    y = torch.empty_like(u)

    for t in range(L):
        dA = torch.exp(delta[:, t, :, None] * A[None, :, :])
        dB_u = delta[:, t, :, None] * B_proj[:, t, None, :] * u[:, t, :, None]
        h = dA * h + dB_u
        y[:, t, :] = (h * C_proj[:, t, None, :]).sum(dim=-1) + D_skip * u[:, t, :]

    return y


__all__ = [
    "DEFAULT_B",
    "DEFAULT_D",
    "DEFAULT_L",
    "DEFAULT_N_STATE",
    "FWD_TOL",
    "SUPPORTS_BACKWARD",
    "TASK_NAME",
    "get_inputs",
    "reference_forward",
]
