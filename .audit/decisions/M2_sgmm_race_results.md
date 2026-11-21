# M2 Target-First SGMM Microkernel Race Results (Apple M1 Max)

Date: 2026-11-21
Device: Apple M1 Max (`32768` bytes max threadgroup memory, `32`-wide SIMD-group, `32` GPU cores, `10.171 TFLOPS` FP32 SGMM peak in `.audit/roofline_m1max.json`)
Method: Native `MTLDevice::newLibraryWithSource` (`MTLMathModeFast`) + GPU command-buffer timestamps (`[cb GPUEndTime] - [cb GPUStartTime]`) via `/tmp/sgmm_race*.mm` compiled with `/usr/bin/clang++ -O3`. All values below are `[measured]`.

## Key Hardware Discoveries on M1 Max

1. **16 KB Occupancy Threshold vs Double-Buffering**:
   - `64x64x32` single-buffered (`16.0 KB` at `PAD=0`, `17.5 KB` at `PAD=4`) halved barrier frequency vs `BK=16` and consistently beat `64x64x16` double-buffered (`18.5 KB` at `PAD=4`).
   - `128x64x16` double-buffered (`28.5 KB`) dropped to 1 TG/core occupancy and regressed to `29.55 ms` on `#1` (`4.65 TFLOPS`), whereas `128x64x32` single-buffered (`26.5 KB` at `PAD=4`) with interleaved `simdgroup_load` + `simdgroup_multiply_accumulate` reached **`7.54–7.68 TFLOPS`**.
   - `2`-SG (`64x64x32` with `32x32` tiles -> 32 accumulators/SG) spilled registers catastrophically (`139.7 ms`, `0.98 TFLOPS`); keeping `<= 16` `simdgroup_float8x8` accumulators (`4x4` grid per SG, `32x32` output/SG) avoided spilling across all schedules.
2. **Odd-Stride 128-bit Vector Loads (`#8`: `K=2949`, `N=5921`)**:
   - Requiring `(lda & 3) == 0` for `*(device float4*)` forced `#8` onto the scalar loop (`93.1–152.6 ms`).
   - Using `*(const device packed_float4*)` (4-byte alignment requirement, hardware 128-bit vector load) fixed `#8` to **`42.91–43.15 ms` (`6.64–6.68 TFLOPS`)** with `err = 0.000000`.
3. **Zero-Copy Transposed A/B Staging (`#16, #17, #18`)**:
   - Coalesced `packed_float4` loads along the contiguous dimension -> coalesced `float4` stores into SRAM -> hardware `simdgroup_load(..., transpose_matrix=true)` (`tm1`) eliminated 16-way bank conflicts on `128x64x32` (`#18` dropped from `29.74 ms` scatter-transpose to **`19.93 ms`**, beating MPS `28.25 ms` by **`1.42x`**).
4. **2D SLC Swizzle (`swizzle_w=8`) + `float4` Epilogue**:
   - `swizzle_w=8` improved `#2 (2048,8192)@(8192,4096)` on `64x64x16` from `33.42 ms` -> `20.63 ms` (`1.62x`).
   - For write-bound small-K GEMMs (`#7`, `#9`, `C = 4.295 GB`), SRAM staging + `float4` coalesced stores + row-major N traversal cut `#9` from `60.02 ms` -> **`14.41 ms`**.
5. **Triangular K-Tightening & Dead-Tile Pruning (`#14, #15`)**:
   - `64x64x32` outperformed `128x64x32` (`3.81 ms` vs `5.56 ms` on `#14`) because finer `64x64` diagonal tiles preserve tighter K ranges and better 32-core load balance (`4.34x` vs MPS on `#14`, `4.63x` vs MPS on `#15`).

## Race Results Across KernelBench L1 Shapes (FP32)

| Task / Shape | MPS Baseline (ms) | Racer A (`64x64x16` 2-buf) | Racer B1 (`64x64x32` 1-buf) | Racer B2 (`128x64x16` 1-buf) | Racer C1 (`128x64x32` 1-buf, `sw8N`, `ilv`) | Racer C2 (`64x128x32` 1-buf, `sw8M`) | Winner / Speedup vs MPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `#1` `(4096,4096)@(4096,4096)` | `16.354` | `20.291 ms` (`6.774 TF`) | `18.879 ms` (`7.280 TF`) | `20.921 ms` (`6.569 TF`) | **`17.905 ms` (`7.676 TF`)** | `18.079 ms` (`7.602 TF`) | `128x64x32` (`0.91x`) |
| `#2` `(2048,8192)@(8192,4096)` | `16.444` | `33.424 ms` (`sw1`) / `20.630 ms` (`sw8N`) | `18.937 ms` (`7.258 TF`) | `21.080 ms` (`6.520 TF`) | **`18.139 ms` (`7.577 TF`)** | `18.812 ms` (`7.306 TF`) | `128x64x32` (`0.91x`) |
| `#3` `B=128 (512,1024)@(1024,2048)` | `31.339` | `41.038 ms` (`6.698 TF`) | `38.203 ms` (`7.195 TF`) | `42.217 ms` (`6.511 TF`) | **`36.678 ms` (`7.494 TF`)** | `37.538 ms` (`7.323 TF`) | `128x64x32` (`0.85x`) |
| `#8` `(8205,2949)@(2949,5921)` | `39.245` | `46.711 ms` (`6.134 TF`) | `44.839 ms` (`6.390 TF`) | `49.270 ms` (`5.816 TF`) | `43.152 ms` (`6.640 TF`) | **`42.913 ms` (`6.677 TF`)** | `64x128x32` (`0.91x`) |
| `#14` `triu(4096,4096)` | `16.541` | `4.069 ms` (`5.629 TF`) | **`3.814 ms` (`6.007 TF`)** | `4.566 ms` (`5.017 TF`) | `5.556 ms` (`4.123 TF`) | `5.083 ms` (`4.506 TF`) | **`64x64x32` (`4.34x`)** |
| `#15` `tril(4096,4096)` | `16.492` | `3.726 ms` (`6.148 TF`) | **`3.560 ms` (`6.435 TF`)** | `4.135 ms` (`5.539 TF`) | `4.953 ms` (`4.625 TF`) | `5.782 ms` (`3.961 TF`) | **`64x64x32` (`4.63x`)** |
| `#16` `transA (8192,2048).T@(8192,4096)` | `19.880` | `20.596 ms` (`6.673 TF`) | `20.042 ms` (`6.858 TF`) | `25.570 ms` (`5.375 TF`) | **`19.545 ms` (`7.032 TF`, `tm1`)** | `19.542 ms` (`7.033 TF`, `tm1`) | **`128x64x32 tm1` (`1.02x`)** |
| `#17` `transB (2048,8192)@(4096,8192).T` | `24.736` | `28.912 ms` (`sw1`) / `22.460 ms` (`sw8N`) | `19.499 ms` (`7.049 TF`) | `22.118 ms` (`6.214 TF`) | **`18.202 ms` (`7.551 TF`, `tm0`)** / `18.628 ms` (`tm1`) | `18.727 ms` (`7.339 TF`, `tm1`) | **`128x64x32` (`1.36x`)** |
| `#18` `transAB (8192,2048).T@(4096,8192).T` | `28.250` | `20.683 ms` (`6.645 TF`) | `20.617 ms` (`6.666 TF`) | `25.626 ms` (`5.363 TF`) | **`19.932 ms` (`6.895 TF`, `tm1`)** | `29.739 ms` (`tm0`) / `30.484 ms` | **`128x64x32 tm1` (`1.42x`)** |
| `#6` `split_k=64 (256,524288)@(524288,256)` | `517.017` | `44.880 ms` (`sk64`) | `11.152 ms` (`sk64`) | `20.268 ms` (`sk64`) | `10.532 ms` (`sk64`) | **`10.332 ms` (`6.651 TF`, `sk64` atomic)** / `10.279 ms` (`sk32` part) | **`64x128x32` (`50.04x–50.30x`)** |
| `#7` `(32768,64)@(64,32768)` | `18.299` | — | **`24.166 ms` (`rowmajor f4epi`)** / `26.697 ms` (`sw8N f4epi`) | — | `36.815 ms` | — | `64x64x32` (`0.76x`) |
| `#9` `(32768,32)@(32,32768)` | `11.578` | — | **`14.410 ms` (`rowmajor f4epi`)** / `19.164 ms` (`sw8N f4epi`) | — | `31.175 ms` | — | `64x64x32` (`0.80x`) |

Note on `#6` Split-K: `64x128x32` single-kernel `atomic_fetch_add_explicit` (`10.332 ms`, `50.04x` vs MPS) and two-kernel partial buffer + vectorized reduce (`10.279 ms`, `50.30x` vs MPS) perform within `0.05 ms` of each other; single-kernel atomic avoids allocating a `16 MB` workspace and extra dispatch.

## Production Emitter Policy (`cpp/codegen/metal/sgmm_emitter.cpp`)

1. **Schedule Honor & Default Selection**:
   - Honors `TileSpecAttr` (`bm, bn, bk, num_simdgroups, double_buffer, split_k, swizzle_w`) whenever SRAM fits `<= target.max_threadgroup_memory_bytes` (`32768` bytes).
   - Default buffering when `double_buffer` is unset or checking `TileSpecAttr`: `128x64x32`, `64x128x32`, and `64x64x32` use single-buffered `PAD=4` SRAM with interleaved `simdgroup_load` + `simdgroup_multiply_accumulate` (`7.28–7.68 TFLOPS`); `64x64x16` double-buffered (`18.5 KB` with `PAD=4`) uses 1-barrier register prefetching.
   - `128x64x32` / `64x64x32` / `64x64x16` / `32x32x32` / `32x32x16` support runtime coalesced `packed_float4` + `simdgroup_load(transpose=true)` (`tm1`) whenever `stride_am == 1` or `stride_bk == 1` (and SRAM fits), falling back cleanly to `tm0` scatter-transpose if `tm1` SRAM exceeds 32 KB (e.g. `64x128x32`) or scalar loads for general non-unit strides.
2. **Preserved Runner-Ups**:
   - `64x64x32_1buf_p4` (`4` SGs): optimal for `#14/#15` triangular GEMMs (`3.56–3.81 ms`) and write-bound small-K `#7/#9`.
   - `64x64x16_2buf_p4` (`4` SGs): preserved when `BK=16` and `double_buffer=true` is selected via `TileSpecAttr` or autotuning.
   - `64x128x32_1buf` (`8` SGs): optimal for `#6` Split-K (`10.33 ms`, `50x` vs MPS) and `#8` irregular (`42.91 ms`).
   - `128x64x32_1buf_p4` (`8` SGs): optimal for dense square/rectangular (`#1, #2, #3`) and transposed (`#16, #17, #18`) GEMMs (`7.03–7.68 TFLOPS`).
