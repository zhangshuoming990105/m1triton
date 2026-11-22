"""Level 3 Spec L3-05: Fused QK-RMSNorm + Half-Split RoPE + In-Place KV-Cache Scatter."""

from __future__ import annotations

import torch

TASK_NAME = "fused_rmsnorm_rope_kv"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)


def get_inputs(
    seed: int = 0,
    device: torch.device | str = "mps",
    *,
    n_tok: int = 4096,
    h_q: int = 32,
    h_kv: int = 8,
    d: int = 128,
    num_slots: int = 8192,
    eps: float = 1e-6,
) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    """Create deterministic Qwen3/Gemma-2/vLLM QK-RMSNorm + RoPE + KV-cache scatter inputs."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    q = torch.randn((n_tok, h_q, d), generator=gen, dtype=torch.float32).to(device)
    k = torch.randn((n_tok, h_kv, d), generator=gen, dtype=torch.float32).to(device)
    v = torch.randn((n_tok, h_kv, d), generator=gen, dtype=torch.float32).to(device)
    q_weight = (1.0 + 0.1 * torch.randn((d,), generator=gen, dtype=torch.float32)).to(device)
    k_weight = (1.0 + 0.1 * torch.randn((d,), generator=gen, dtype=torch.float32)).to(device)

    positions = torch.arange(n_tok, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d)))
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles).contiguous().to(device)
    sin = torch.sin(angles).contiguous().to(device)

    slot_mapping = (
        torch.randperm(num_slots, generator=gen, dtype=torch.int64)[:n_tok]
        .to(torch.int32)
        .contiguous()
        .to(device)
    )
    k_cache = torch.randn((num_slots, h_kv, d), generator=gen, dtype=torch.float32).to(device)
    v_cache = torch.randn((num_slots, h_kv, d), generator=gen, dtype=torch.float32).to(device)

    inputs = (q, k, v, q_weight, k_weight, cos, sin, slot_mapping, k_cache, v_cache)
    return inputs, {"eps": float(eps)}


def reference_forward(
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
    """PyTorch reference: QK-RMSNorm -> half-split RoPE -> in-place KV-cache scatter."""
    half_d = q.shape[-1] // 2

    q_inv_rms = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    q_norm = q * q_inv_rms * q_weight

    k_inv_rms = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
    k_norm = k * k_inv_rms * k_weight

    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)

    q0, q1 = q_norm[..., :half_d], q_norm[..., half_d:]
    q_out = torch.cat((q0 * cos_e - q1 * sin_e, q0 * sin_e + q1 * cos_e), dim=-1)

    k0, k1 = k_norm[..., :half_d], k_norm[..., half_d:]
    k_rope = torch.cat((k0 * cos_e - k1 * sin_e, k0 * sin_e + k1 * cos_e), dim=-1)

    slots = slot_mapping.long()
    k_cache[slots] = k_rope
    v_cache[slots] = v
    return q_out, k_cache, v_cache
