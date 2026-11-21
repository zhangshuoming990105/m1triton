# M7 Cluster D — FlashAttention-M1 Performance Optimization (`#097 ScaledDotProductAttention`)

Date: 2026-11-21
Device: Apple M1 Max (`32` GPU cores, `32,768` bytes max threadgroup memory/TG, `64 KB` SRAM/core, `32`-wide SIMD-group, `10.171 TFLOPS` FP32 SGMM peak / `373.83 GB/s` DRAM)
Target Task: KernelBench L1 `#097 ScaledDotProductAttention` (`(32, 32, 512, 1024)` FP32 non-causal, `1099.51 GFLOPs`, strict `atol = rtol = 1e-4`, `disp == 1`, zero `N×N` device allocation)

All measurements below are `[measured]` on Apple M1 Max using Metal GPU timestamps (`MTLMathModeFast`) and interleaved E2E timings on UMA-safe medium shapes (`(4, 8, 512, 1024)` = `268 MB` total, `34.36 GFLOPs`; and `(8, 8, 512, 1024)` = `537 MB` total, `68.72 GFLOPs`).

## 1. Baseline & Component Phase Breakdown (`(4, 8, 512, 1024)` FP32, `34.36 GFLOPs`)

| Measurement | GPU Median (ms) | Wall Median (ms) | TFLOPS | Notes |
| --- | ---: | ---: | ---: | --- |
| Live PyTorch 2.15.0 MPS (`F.scaled_dot_product_attention`) | — | `5.319–5.409` | `6.35–6.46` | Component BMMs: `Q@K^T = 2.312ms (7.43 TF)`, `P@V = 2.345ms (7.33 TF)` |
| M6 Production `task_097.ModelNew` (with in-kernel fallback) | — | `6.667` | `5.15` | `198.54 ms (0.77x)` on full `#097` |
| M6 `SinglePassSgmm` raw (`16x512, 16 SG`, **NO fallback**) | `5.897` | `6.296` | `5.83` | Eliminating fallback immediately recovers `~0.37 ms` medium / `~8.7 ms` full |
| └ Phase 1: `Q @ K^T` solo (`17.18 GFLOPs`) | **`3.616`** | `3.950` | **`4.75`** | **Primary bottleneck (`61.3%` of kernel time!)** |
| └ Phase 2: `Softmax` solo (difference) | `0.017` | `0.025` | — | `0.3%` of kernel time (negligible) |
| └ Phase 3: `P @ V` solo (`17.18 GFLOPs`) | **`2.174`** | `2.485` | **`7.90`** | Already faster than PyTorch MPS `P @ V` (`2.345 ms`) |

## 2. Root-Cause Hardware Discoveries on M1 Max

1. **In-Kernel Fallback Register Bloat**: Including `emit_general_rowwise` in the same Metal function inflated register count and dropped occupancy (`6.667 ms -> 6.296 ms` wall on medium). Adding `FAST_ONLY: tl.constexpr = 0|1` compiles a razor-thin fast kernel with zero fallback code when preconditions are satisfied at dispatch.
2. **SRAM Stride-512 Bank Conflicts (Disproved for `simdgroup_load`!)**: Testing XOR-swizzled `float2` SRAM (`5.936 ms` vs `5.897 ms`) and K-blocked `P[64][16][8]` / `P[32][16][16]` (`2.177 / 2.195 ms` vs `2.161 ms` linear stride-512) proved empirically that M1 Max `simdgroup_load` from SRAM suffers **zero bank-conflict penalty** on stride `512`. By contrast, stride `32` (`pad=0` on `Q^T[bq][32]`) *did* show a measurable penalty (`2.333 ms` vs `2.279 ms` at `ldqt=36`) due to scalar scatter + 32-bank aliasing on narrow rows.
3. **Why `Q @ K^T` Was `3.62 ms (4.75 TF)` vs `P @ V` `2.17 ms (7.90 TF)`**:
   - **Small `BM=16` (`2x4` SGMM)**: only `2` MMAs per `K` load (`3.856 ms`). Doubling to `BM=32` (`4x4` SGMM = `16` accs, matching `roofline_probe.py`) cut `K` traffic in half (`3.856 -> 2.719 ms`, **`-1.14 ms`**).
   - **`transpose_matrix = true` on Device `K`**: Removing transpose on device `K` saved `0.89 ms` on `2x4` and `0.22 ms` on `4x4`.
   - **Computing `S^T = K @ Q^T` with Double-Buffered Staged `Q^T` in Free `m1_shm`**: During `Q @ K^T`, `m1_shm` (`32 KB`) is 100% idle. Transposing the formulation to **`S^T = K @ Q^T`** loads `A = K` directly from device with **`transpose_matrix = false`** and loads `B = Q^T` from double-buffered SRAM `m1_shm[2][64][36]` (`18.4 KB <= 32 KB`) with **`transpose_matrix = false`** (`2.719 -> 2.262 ms`, **`7.60 TFLOPS`** — beating PyTorch MPS `bmm(Q, K^T)` `2.312 ms`!).
   - **Compile-Time Constant Stride (`D_STATIC`, `LKV_STATIC`)**: Passing `D_STATIC` (`64, 128, 256, 512, 1024`) turns 64-bit runtime row-stride multiplies in every `simdgroup_load`/`simdgroup_store` into constant shift/add addressing, pushing `K @ Q^T` (`-0.18 ms`) and `P @ V` (`2.161 -> 2.056 ms`, **`8.35 TFLOPS`**).
   - **Single Base-Pointer Addressing Rule on M1 Max**: Holding multiple 64-bit device pointers (`kp_0..kp_3` or `vp, vp1`) or splitting `kb + 8192 + kk` into hoisted temporaries added `+0.12–0.28 ms` of register pressure at `512` threads/TG. Keeping a single `kb` / `vp` base pointer with `kb + (i+1)*8L * D + ks + sub` preserves peak occupancy.
4. **Two-Wave `BM=32, BN=512, 16 SG` Disjoint-Memory Architecture (`32.0 KB` Exact SRAM, Zero Intermediate `O` Writes)**:
   - After `K @ Q^T` (`4x4` = `16` tiles per SG), rows `0..15` (`8` tiles) are stored transposed into `m1_shm[16][512]` (`32.0 KB`) for Wave 1 Softmax + `P @ V` (`2x8`), while rows `16..31` (`8` tiles) remain resident in registers (`8 + 16 = 24` tiles, zero spill overhead: `4.349 ms` actual vs `2.262 + 0.030 + 2.056 = 4.348 ms` sum of isolated parts!). Wave 2 then stores rows `16..31` into `m1_shm` for Softmax + `P @ V`.

## 3. Systematic Hypothesis Race Log (`(4, 8, 512, 1024)` FP32)

| Racer / Hypothesis | Scope | GPU (ms) | TFLOPS | `max_abs_err` | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `R0`: M6 `16x512 16SG` raw (NO fallback) | Full FA | `5.897` | `5.83` | `5.36e-7` | Baseline fast-only |
| `R1`: `16x512 16SG` + XOR-swizzled `m1_shm` | Full FA | `5.936` | `5.79` | `5.36e-7` | Rejected (`simdgroup_load` has 0 bank conf on 512) |
| `R2a`: `16x512 16SG` + Staged `Q[16, 256+4]` | Full FA | `5.823` | `5.90` | `5.36e-7` | Minor (`-0.07 ms`); `K` was bottleneck, not `Q` |
| `R3a`: `16x512 8SG (2x8)` direct + XOR | Full FA | `6.638` | `5.18` | `5.36e-7` | Rejected (8 SG + 32KB `2x8` K-bound) |
| `R5a`: `32x256 8SG (4x4)` 2-pass SLC-O | Full FA | `6.634` | `5.18` | `6.56e-7` | Rejected (direct `K` `T=true` + SLC-O overhead) |
| `QK2a`: `16x512 16SG` `j`-outer `BD=32` | `Q@K^T` | `3.164` | `5.43` | `0.00e+0` | Better CL reuse (`-0.69 ms`), superseded by `BM=32` |
| `QK2d`: `32x512 16SG 4x4` direct `Q/K` | `Q@K^T` | `2.719` | `6.32` | `0.00e+0` | Major win (`-1.14 ms`); `4x4` halves `K` loads |
| `QK3d`: `32x512 16SG` StagedQ + PerSG `K^T` | `Q@K^T` | `2.902` | `5.92` | `0.00e+0` | Proved manual `K^T` > SRAM `T=true` (`6.24 ms`) |
| `SGMM_SLC 64x128 8SG DB` (`manT`) | `Q@K^T` | `2.653` | `6.48` | `0.00e+0` | 64 barriers + `O` SLC passes (`7.15 ms` full FA) |
| `DIR 4x4 float4 + simd_shuffle K` | `Q@K^T` | `11.363` | `1.51` | `0.00e+0` | Rejected (shuffle ALU bottleneck) |
| `FA_2W_dir_2x8`: Two-Wave `32x512` direct `Q/K` | Full FA | `5.062` | `6.79` | `5.36e-7` | First racer to beat live PyTorch MPS (`5.32 ms`)! |
| `SWAP K@Q^T` `bq=64 ld=36 DB` overlapped + `D=1024` | `K@Q^T` | **`2.261`** | **`7.60`** | `0.00e+0` | Beats PyTorch MPS `bmm(Q, K^T)` (`2.312 ms`)! |
| `PV 16x512 2x8` `cont_ku` + `D=1024` | `P@V` | **`2.056`** | **`8.35`** | — | Beats PyTorch MPS `bmm(P, V)` (`2.345 ms`)! |
| `Design D` (`8 SG` `32x512`, `m1_Ob` SLC `S`-stage) | Full FA | `5.332` | `6.44` | `5.36e-7` | Rejected (2x `Q` reads + `S` SLC bounce) |
| `8 SG` Single-Wave `16x512` (`8x2 QK + 2x8 PV`) | Full FA | `5.890` | `5.83` | `5.36e-7` | Rejected (`8x2` doubles `K` traffic) |
| **`FA5` Production `TwoWaveSgmm` (`32x512, 16 SG`)** | **Full FA** | **`4.355`** | **`7.89`** | **`5.36e-7`** | **WINNER**: `SWAP K@Q^T` DB + `cont_sub`/`cont_ku` + `FAST_ONLY=1` + `D_STATIC` |

## 4. Production Verification & Scaling Results (`task_097.ModelNew`)

Interleaved E2E benchmark vs live PyTorch 2.15.0 MPS (`torch.empty` + sync included):
- **Medium `(4, 8, 512, 1024)` (`268 MB`)**:
  - Live PyTorch 2.15.0 MPS: `5.409 ms` (`6.35 TF`)
  - M6 Production `task_097`: `6.667 ms` (`5.15 TF`, `0.81x`)
  - **M7 Production `task_097`**: **`4.992 ms` E2E (`6.88 TF`, `1.084x` vs live MPS, `1.34x` vs M6)**, **`4.355 ms` pure GPU (`7.89 TF`)**, `max_abs_err = 5.36e-7` (`disp == 1`).
- **Scaled `(8, 8, 512, 1024)` (`537 MB`)**:
  - Live PyTorch 2.15.0 MPS: `10.059 ms` (`6.83 TF`)
  - **M7 Production `task_097`**: **`9.321 ms` E2E (`7.37 TF`, `1.079x` vs live MPS)**, **`8.670 ms` pure GPU (`7.93 TF`, projects to `~138.7 ms` GPU on full `#097` vs `151.82 ms` live PyTorch MPS)**, `max_abs_err = 6.56e-7`.

All 22 production correctness/edge-case tests + 8 `TwoWaveSgmm FAST_ONLY=0` fallback tests pass with `disp == 1`, `TorchComputeGuard`, `static_ast_audit`, and strict `atol = rtol = 1e-4` (FP16 `1e-2`):
- Two-Wave `D in {64, 128, 192, 256, 512, 1024}`, non-square `L_q != L_kv`, `randn` inputs, causal & negative `causal_offset` (all-masked zero-NaN rows), unaligned `storage_offset=1` views (`packed_float4` loads), Single-Pass `L_q=48`, Multi-Tile `L_kv=768`, odd shapes `(2,3,37x53,73)`, `D=1088 > 1024`, non-contiguous views, `TwoWaveSgmm FAST_ONLY=0` fallback (`odd`, `D=1088`, `L_kv=576`, `FP16`), and FP16.
- Full local CI parity (`bash scripts/ci_local.sh`): `Guardrails OK`, `Feature Map audit OK`, `15/15` C++ CTests (`test_flash_attn_emitter` passed), `52/52` Python pytests passed.
