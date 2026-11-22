# M1Triton Level 3 Complex Triton Kernel Suite Scoreboard (`v2.0`)

- Harness: SHA-256-locked [evals/eval_triton_l3.py](../evals/eval_triton_l3.py) (`--seeds 3 --warmup 10 --trials 30`)
- Passed: **10/10** | Beats Baseline (`>1.00x`): **10/10** | Geomean Speedup: **3.9235x**
- Raw Artifact: [.audit/eval_final_l3_10.json](../.audit/eval_final_l3_10.json)

## Complete 10-Kernel Suite (`[measured]`)

| ID | Kernel | Fwd+Bwd | Base MPS (ms) | M1Triton (ms) | Speedup | Max Fwd Err | Max Bwd Err | Highlights |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |
| `L3-01` | `flash_attn2_fwd_gqa_causal` | Fwd | 9.1243 | 2.8112 | **3.25x** | 1.07e-06 | - | Causal GQA (`H_q=16, H_kv=4, L=1024, D=64`), zero `N×N` DRAM materialization, `@autotune` |
| `L3-02` | `flash_attn2_fwd_bwd_causal` | Yes | 1.9734 | 1.8797 | **1.05x** | 1.01e-06 | 1.91e-06 | Causal Fwd + Bwd (`dQ, dK, dV`) via `torch.autograd.Function`, SRAM recomputation, concurrent `command_batch()` |
| `L3-03` | `paged_attn_decode_vllm` | Fwd | 23.4686 | 1.8899 | **12.42x** | 2.09e-07 | - | vLLM block-table indirect gather (`block_size=16`, ragged `context_lens`), 2-stage Split-KV merge |
| `L3-04` | `sliding_window_alibi_softcap_attn` | Fwd | 5.1920 | 0.7640 | **6.80x** | 4.29e-06 | - | Banded sliding window (`W=128`) + per-head ALiBi slopes + `tanh(50.0)` softcap inside inner loop |
| `L3-05` | `fused_rmsnorm_rope_kv` | Fwd | 9.5170 | 1.2390 | **7.68x** | 1.43e-06 | - | Single-pass Llama/Qwen prelude: 1-read SRAM RMSNorm + rotary embedding + in-place KV-cache scatter |
| `L3-06` | `fused_layernorm_fwd_bwd` | Yes | 1.9132 | 1.6259 | **1.18x** | 9.54e-07 | 7.25e-05 | Fused LayerNorm Fwd + Bwd (`dx, dgamma, dbeta`) using Metal 3.0 `atomic_float` parameter reduction |
| `L3-07` | `chunked_linear_cross_entropy` | Yes | 71.8625 | 57.3470 | **1.25x** | 9.54e-07 | 9.78e-09 | Chunked `V=32768` Linear + Cross-Entropy Fwd + Bwd (`[BT, V]` logit tensor never materialized in UMA DRAM) |
| `L3-08` | `mamba_ssm_scan` | Fwd | 55.6889 | 0.8365 | **66.58x** | 3.34e-06 | - | Chunked parallel associative selective scan (`B, L, D, N_state`) in threadgroup SRAM + SIMD registers |
| `L3-09` | `moe_grouped_gemm` | Fwd | 2.2455 | 1.7187 | **1.31x** | 0.00e+00 | - | Ragged expert grouped GEMM with dynamic persistent tile scheduler across 32 Apple M1 Max cores |
| `L3-10` | `w4a16_dequant_gemm` | Fwd | 4.6854 | 1.5357 | **3.05x** | 0.00e+00 | - | Packed `uint32` 4-bit weights + group-128 scales/zeros (`8x` vs FP32 / `4x` vs FP16 footprint) fused SGMM |

## Memory Footprint & Runtime Verification Highlights (`[measured]`)

- **FlashAttention-2 (`L3-01`, `L3-02`)**: Online softmax tiling + SRAM backward recomputation avoids materializing `(B, H, L, L)` attention scores in UMA DRAM.
- **Chunked Linear Cross-Entropy (`L3-07`)**: Vocabulary chunking computes exact loss and `(dX, dW)` gradients without allocating `[B*T, V]` FP32 logit buffers (`1.25x` faster end-to-end fwd+bwd).
- **W4A16 Group-Quantized GEMM (`L3-10`)**: Packs 8 `int4` weights per `uint32` word (`64 MB FP32 / 32 MB FP16 -> 8.00 MB W4 + 0.50 MB scales/zeros`, `3.05x` faster than PyTorch MPS reference).
- **MetalGraph Compute ICB vs. Concurrent `command_batch()`**: On `L3-02`'s 3-kernel backward (`delta -> dkdv || dq`), `MTLIndirectCommandBuffer` replay runs in **`0.921 ms` wall / `0.646 ms` GPU** vs. `1.354 ms` wall / `0.874 ms` GPU for concurrent `command_batch()` (**`1.471x`** faster wall-clock) while `command_batch()` handles dynamic per-call PyTorch autograd buffers cleanly with automatic RAW/WAW/WAR barriers.
