"""L3-09 Reference Spec: Mixtral-style 8-expert ragged MoE Grouped GEMM."""

from __future__ import annotations

import torch

TASK_NAME = "moe_grouped_gemm"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)

NUM_EXPERTS = 8
TOTAL_TOKENS = 2048
HIDDEN_K = 1024
OUT_N = 2048

_EXPERT_BUCKET_PATTERNS: tuple[tuple[int, ...], ...] = (
    (413, 127, 519, 37, 285, 91, 443, 133),
    (512, 0, 389, 17, 613, 129, 307, 81),
    (257, 319, 63, 481, 149, 223, 509, 47),
)


def get_inputs(seed: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    counts = _EXPERT_BUCKET_PATTERNS[seed % len(_EXPERT_BUCKET_PATTERNS)]
    assert len(counts) == NUM_EXPERTS and sum(counts) == TOTAL_TOKENS

    offsets_cpu = [0]
    for c in counts:
        offsets_cpu.append(offsets_cpu[-1] + c)

    a = (torch.randn((TOTAL_TOKENS, HIDDEN_K), generator=gen, dtype=torch.float32) * 0.25).to(device)
    w = (torch.randn((NUM_EXPERTS, OUT_N, HIDDEN_K), generator=gen, dtype=torch.float32) * 0.25).to(device)
    offsets = torch.tensor(offsets_cpu, dtype=torch.int32, device=device)
    return (a, w, offsets), {}


def reference_forward(a: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    total_tokens = a.shape[0]
    num_experts, n, _ = w.shape
    out = torch.empty((total_tokens, n), device=a.device, dtype=a.dtype)
    offs = offsets.tolist()
    for e in range(num_experts):
        s = int(offs[e])
        end = int(offs[e + 1])
        if end > s:
            out[s:end] = torch.matmul(a[s:end], w[e].t())
    return out
