"""L3-03 Reference Spec: vLLM PagedAttention Decode (B=8, H_q=16, H_kv=4, D=64, block_size=16, max_ctx=1024)."""

from __future__ import annotations

import math
from typing import Any

import torch

TASK_NAME = "paged_attn_decode_vllm"
SUPPORTS_BACKWARD = False
FWD_TOL = (1e-3, 1e-3)


def get_inputs(seed: int, device: torch.device) -> tuple[tuple[Any, ...], dict[str, Any]]:
    b, h_q, h_kv, d = 8, 16, 4, 64
    block_size = 16
    max_context = 1024
    max_blocks = max_context // block_size
    num_total_blocks = b * max_blocks + 32

    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((b, h_q, d), generator=gen, dtype=torch.float32).to(device)
    k_cache = torch.randn((num_total_blocks, h_kv, block_size, d), generator=gen, dtype=torch.float32).to(device)
    v_cache = torch.randn((num_total_blocks, h_kv, block_size, d), generator=gen, dtype=torch.float32).to(device)

    # Ragged context lengths spanning short to max_context, including non-multiples of block_size.
    base_lens = torch.tensor([137, 256, 389, 512, 643, 768, 895, 1024], dtype=torch.int32)
    jitter = torch.randint(-31, 32, (b,), generator=gen, dtype=torch.int32)
    context_lens = torch.clamp(base_lens + jitter, min=32, max=max_context).to(torch.int32)
    context_lens[-1] = max_context

    # Shuffled physical block allocation table with unique blocks per sequence.
    perm = torch.randperm(num_total_blocks, generator=gen, dtype=torch.int32)
    block_tables = perm[: b * max_blocks].view(b, max_blocks).contiguous()

    return (q, k_cache, v_cache, block_tables.to(device), context_lens.to(device)), {"block_size": block_size}


def reference_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    b, h_q, d = q.shape
    h_kv = k_cache.shape[1]
    gqa_ratio = h_q // h_kv
    scale = 1.0 / math.sqrt(float(d))
    ctx_list = context_lens.to("cpu").tolist()

    outs: list[torch.Tensor] = []
    for idx in range(b):
        ctx_len = int(ctx_list[idx])
        num_blks = (ctx_len + block_size - 1) // block_size
        phys_blks = block_tables[idx, :num_blks].to(torch.int64)
        k_seq = k_cache.index_select(0, phys_blks).permute(1, 0, 2, 3).reshape(h_kv, num_blks * block_size, d)
        v_seq = v_cache.index_select(0, phys_blks).permute(1, 0, 2, 3).reshape(h_kv, num_blks * block_size, d)
        k_seq = k_seq[:, :ctx_len, :]
        v_seq = v_seq[:, :ctx_len, :]
        if gqa_ratio > 1:
            k_seq = k_seq.repeat_interleave(gqa_ratio, dim=0)
            v_seq = v_seq.repeat_interleave(gqa_ratio, dim=0)
        q_b = q[idx].unsqueeze(1)
        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out_b = torch.matmul(probs, v_seq).squeeze(1)
        outs.append(out_b)
    return torch.stack(outs, dim=0)
