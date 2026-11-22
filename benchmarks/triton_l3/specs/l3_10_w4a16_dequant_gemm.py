"""L3-10 Reference Spec: W4A16 4-bit Group-Quantized Inference GEMM."""

from __future__ import annotations

import torch

TASK_NAME = "w4a16_dequant_gemm"
SUPPORTS_BACKWARD = False
FWD_TOL = (5e-3, 5e-3)

M_TOKENS = 256
K_DIM = 4096
N_DIM = 4096
GROUP_SIZE = 128


def get_inputs(seed: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = (torch.randn((M_TOKENS, K_DIM), generator=gen, dtype=torch.float32) * 0.25).to(torch.float16).to(device)

    nibbles = torch.randint(0, 16, (K_DIM // 8, 8, N_DIM), generator=gen, dtype=torch.int64)
    packed = torch.zeros((K_DIM // 8, N_DIM), dtype=torch.int64)
    for j in range(8):
        packed |= nibbles[:, j, :] << (4 * j)
    qweight = packed.to(torch.int32).to(device)

    num_groups = K_DIM // GROUP_SIZE
    scales = (
        torch.empty((num_groups, N_DIM), dtype=torch.float32).uniform_(0.005, 0.02, generator=gen)
    ).to(torch.float16).to(device)
    zeros = (
        torch.empty((num_groups, N_DIM), dtype=torch.float32).uniform_(6.5, 8.5, generator=gen)
    ).to(torch.float16).to(device)
    return (a, qweight, scales, zeros), {"group_size": GROUP_SIZE}


def reference_forward(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    k_packed, n = qweight.shape
    k = k_packed * 8
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32).view(1, 8, 1)
    unpacked = ((qweight.unsqueeze(1) >> shifts) & 0xF).reshape(k, n).to(a.dtype)
    num_groups = k // group_size
    w = ((unpacked.view(num_groups, group_size, n) - zeros.unsqueeze(1)) * scales.unsqueeze(1)).reshape(k, n)
    return torch.matmul(a, w)
