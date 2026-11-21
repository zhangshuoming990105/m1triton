# M6 Target-First FlashAttention-M1 Microkernel Race (Apple M1 Max)

Date: 2026-11-21
Device: Apple M1 Max (`32768` bytes max threadgroup memory, `32`-wide SIMD-group, `32` GPU cores, `48 MB` SLC, `10.171 TFLOPS` FP32 SGMM peak / `373.83 GB/s` DRAM in `.audit/roofline_m1max.json`)
Target Task: KernelBench L1 `#97 ScaledDotProductAttention` (`Q, K, V: (32, 32, 512, 1024)`, FP32, non-causal, `scale = 1/sqrt(1024) = 0.03125`, `1099.51 GFLOPs`, `108.1 ms` physical floor, `258.75 ms` pinned baseline in `.audit/baseline_l1_mps.json`, strict `atol = rtol = 1e-4` since `#97` is outside `DEEP_FP32_ACCUM_TASK_IDS`).

All measurements below are `[measured]` on M1 Max via `m1triton.runtime` GPU timestamps (`MTLMathModeFast`) and interleaved end-to-end wall timings.

## Key Hardware Discoveries on M1 Max

1. **SGMM Register Allocation & Spill Cliff (Microbenchmark)**:
   - Indexing `simdgroup_float8x8 acc[4][4]` inside a `for` loop (even with `#pragma unroll`) or calling `.thread_elements()` on an array element forces thread-local scratch spilling (`0.15 TFLOPS`).
   - Explicitly codegen-unrolled `simdgroup_float8x8` variables or `float2` scalar accumulators sustain **`9.71–10.17 TFLOPS`** up to `20–24` `float2` accumulators (`40–48` FP32 regs/thread), then hit a sharp spill cliff at `>= 28` `float2` (`1.03 TFLOPS`).
2. **SLC Head-Locality Grid Mapping (`1.88x` Speedup)**:
   - One head `(512, 1024)` of `K + V` is `4.0 MB` (`<< 48 MB` SLC).
   - Mapping `tgid.x = q_tile` (fast) and `tgid.y = bh` (slow) with `BM = 16` (`32` Q-tiles per head across `32` GPU cores) keeps all 32 cores sharing the **same single head** in SLC: `6.56 ms` vs `12.36 ms` when grid axes are swapped (`1.88x`).
3. **Eliminating Inner-D Barriers via Direct `simdgroup_load` + Hardware Transpose**:
   - Staging `Q` and `K` in SRAM across `D = 1024` requires `64` `threadgroup_barrier`s per KV tile and limits `BN`.
   - Loading `Q` and `K` directly with `simdgroup_load(..., transpose_matrix=true)` eliminates all barriers inside the `D`-reduction loop (`32x128` dropped from `9.70 ms` -> `7.67 ms`; `16x128` dropped to `6.58 ms`), while `P @ V` with `P` resident in SRAM and direct `V` loads reaches `7.35 TFLOPS` (`2.33 ms` on medium).
4. **Disjoint Register Lifetimes (`SinglePassSgmm`, `BM=16, BN=512, NSG=16`, `32.0 KB` Exact SRAM)**:
   - When `L_kv <= 512` (multiples of 32) and `D <= 1024` (multiples of 64), `S[16, 512]` fits in **`16 * 512 * 4 = 32,768` bytes** exact with zero extra SRAM arrays (`m_s/l_s/alpha_s` not needed).
   - Keeping `Q @ K^T` (`2x4 = 8` `simdgroup_float8x8` accs, `D_GRP = 1`, zero Split-D reduction) and `P @ V` (`2x8 = 16` `simdgroup_float8x8` accs) in **disjoint lexical scopes** eliminates simultaneous `S + O` register pressure, reads `Q, K, V` once each and writes `O` once (theoretical minimum `8.59 GB` DRAM traffic on `#97`).

## Medium-Shape Race Results: `(4, 8, 512, 1024)` FP32 (`34.36 GFLOPs`)

| Racer / Schedule | Architecture | SRAM | `max_abs_err` | Median (ms) | TFLOPS | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `A: 64x64x32 (4 SG)` | SLC-O, staged `Q/K/V`, `4x4` SGMM | `26.2 KB` | `9.83e-7` | `14.932` | `2.30` | 1 TG/core (4 SGs/core); 7 intermediate `O` R/W passes |
| `B: 32x128x32 (4 SG)` | SLC-O, staged `Q/K/V`, `4x4` SGMM | `25.1 KB` | `1.04e-6` | `15.591` | `2.20` | 4 SGs/core under-occupies 4 SIMD units across barriers |
| `C: 64x64x32 (8 SG, BDO=128)` | SLC-O, staged `Q/K/V`, `2x4`/`4x4` | `26.8 KB` | `9.83e-7` | `11.042` | `3.11` | Better occupancy; still pays 7 `O` passes + epilogue barriers |
| `D1: 32x128x32 (32 SG)` | InReg-O (`4x4`), staged `Q/K`, direct `V` | `22.9 KB` | `1.37e-6` | `8.708` | `3.95` | Zero device-`O` traffic; `P@V` hits `7.35 TF` solo |
| `D2: 32x128 (32 SG, dirQK)` | InReg-O (`4x4`), direct `Q/K/V`, `2x2` `S` | `16.9 KB` | `1.40e-6` | `7.674` | `4.48` | Eliminated 64 D-loop barriers |
| `E1: 16x128 (16 SG, dirQK)` | InReg-O (`2x8`), direct `Q/K/V`, `2x2` `S` | `8.4 KB` | `1.30e-6` | `6.576` | `5.23` | 2 TGs/core + 32 Q-tiles/head SLC hit |
| `E2: 16x256 (16 SG, dirQK)` | InReg-O (`2x8`), `2x4` `S`, `D_GRP=2` | `16.4 KB` | `1.30e-6` | `6.299` | `5.45` | 2 KV-steps; selected as general multi-tile SGMM path (`L_kv > 512`) |
| **`F: 16x512 (16 SG, Disjoint)`** | **Single-pass disjoint `S`/`O`, `D_GRP=1`, `BD=16`** | **`32.0 KB`** | **`5.36e-7`** | **`5.688–6.074`** | **`5.66–6.04`** | **Winner**: 1 pass over `Q/K/V`, zero Split-D, zero `O` spill |

## Full-Shape Verification: `#97 (32, 32, 512, 1024)` FP32 (`1099.51 GFLOPs`)

| Implementation | `max_abs_err` (`atol=1e-4`) | Median (ms) | Min (ms) | TFLOPS | Speedup vs Pinned MPS (`258.75 ms`) |
| --- | --- | --- | --- | --- | --- |
| Pinned MPS baseline (`.audit/baseline_l1_mps.json`) | ref | `258.75` | — | `4.25` | `1.00x` |
| Live solo PyTorch MPS (`F.scaled_dot_product_attention`) | ref | `150.84` | `150.06` | `7.29` | `1.72x` |
| Racer E2 (`MultiTileSgmm`, `BM=16, BN=256, 16 SG`, raw kernel) | `1.3e-6` | `219.97` | `214.52` | `5.00` | `1.18x` |
| Racer F (`SinglePassSgmm`, `BM=16, BN=512, 16 SG`, raw kernel) | `5.4e-7` | `190.93` | `189.58` | `5.76` | `1.36x` |
| **Production `task_097.ModelNew` (end-to-end incl. 2.15 GB alloc + fallback)** | **`5.96e-7`** | **`199.61`** | **`195.31`** | **`5.51`** | **`1.30x`** |

## Selected Production Architecture (`flash_attn_lowering.cpp` + `flash_attn_emitter.cpp`)

1. **`FlashAttentionLoweringPass`**: validates operands/dtypes/finite scale, marks `out_ptr` writable, honors `TileSpecAttr` and `BLOCK_M`/`BLOCK_N`/`BLOCK_D`/`NUM_SG` constexprs, and clamps tile shapes strictly within `target.max_threadgroup_memory_bytes` and `target.max_threads_per_threadgroup` (defaulting to `BM=16, BN=256, BD=16, NUM_SG=16`, and allowing exact-fit `BN=512` when `BM=16, NUM_SG=16, F32`).
2. **`try_emit_flash_attn_kernel` (3-tier generality in 1 fused kernel, zero N×N device allocation)**:
   - **Tier 1 (`SinglePassSgmm`, `BM=16, BN=512, 16 SG`, `32.0 KB`)**: Disjoint-register `simdgroup_float8x8` `Q@K^T` (`2x4`, `D_GRP=1`) -> `simd_max`/`simd_sum` softmax -> `simdgroup_float8x8` `P@V` (`2x8`), covering `L_kv <= 512` (mult-32), `D <= 1024` (mult-64), causal & non-causal.
   - **Tier 2 (`MultiTileSgmm`, `BM=16, BN in {64,128,256}, 16 SG`, `<= 16.5 KB`)**: In-register `O` streaming online softmax for arbitrarily long `L_kv > 512` with causal early-exit (`m1_kv_lim`).
   - **Tier 3 (Zero-barrier Row-Wise Online Softmax Fallback)**: Built into both SGMM kernels (and standalone for `FP16`/non-standard tile configs), using 1 simdgroup per query row with private 256-float scratch so arbitrary/odd `(BH, L_q, L_kv, D)`, `D > 1024`, tail query rows, negative `causal_offset` all-masked rows (zero NaNs), and `FP16` all execute accurately in a single dispatch.
3. **M7 Perf Follow-Up**: Notice live solo PyTorch MPS reached `150.84 ms` (`7.29 TF`) vs `199.61 ms` (`5.51 TF`) for our combined kernel (`+8.7 ms` vs standalone `190.93 ms` due to the in-kernel row-wise fallback adding register pressure and `torch.empty` allocation). Separating the exact-shape fast specialization from the fallback and exploring XOR-swizzled `shm` / wider `Q@K^T` schedules in M7 can push toward the `~130–145 ms` target.
