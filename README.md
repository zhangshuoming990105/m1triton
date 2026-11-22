# M1Triton

Apple M1 Max specialized Triton-compatible and compiler-DSL co-designed compiler (`Python AST -> C++20 Typed Block SSA IR -> Pure C++20 Passes -> MSL Codegen -> Objective-C++ Metal UMA Runtime`).

## Verified KernelBench Level 1 Results (`v1.0-kernelbench-l1`, `[measured]`)

Evaluated end-to-end on **Apple M1 Max** (32 GPU cores, 32 GB UMA, 400 GB/s theoretical / 373.83 GB/s measured STREAM) using the unmodified SHA-256-locked harness ([evals/eval_kernelbench.py](evals/eval_kernelbench.py), `--seeds 3 --warmup 10 --trials 30`, pinned `ScalingIntelligence/KernelBench` commit `423217d9fda9`) against PyTorch 2.15.0 MPS eager:

- **Correctness**: **100 / 100 passed** (`0` failed, `0` missing; strict `rtol=1e-4, atol=1e-4`, or `1e-3` on deep-K FP32 accumulators; strict IEEE-754 NaN/Inf matching)
- **Zero Framework Fallbacks**: Every task executes compiled M1Triton Metal kernels (`min compiled_kernel_dispatch_count = 3` across 3 seeds), enforced statically via AST guardrails and dynamically via `TorchDispatchMode` ATen compute tripwires
- **Overall Speedup**: **1.9268x geometric mean** vs PyTorch MPS (`71/100` strictly faster `> 1.00x`, `85/100` at parity or faster `>= 0.98x`, `95/100 >= 0.90x`)

| Category | Tasks | Passed | Beats Baseline (`>1.0x`) | Parity+ (`>=0.98x`) | Geomean Speedup | Highlights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Matmul & GEMM (`#1–#18`) | 18 | 18/18 | 9 | 13 | **1.45x** | `#6` Large-K Split-K **8.36x**, `#15` LowerTri **3.96x**, `#14` UpperTri **3.77x**, `#12` Diag **2.20x** |
| Activations & Softmax (`#19–#32, #88`) | 15 | 15/15 | 7 | 15 | **1.41x** | `#88` MinGPTNewGelu **7.66x**, `#30` Softsign **3.51x**, `#25` Swish **2.51x**, `#23` Softmax **1.35x** |
| Normalization (`#33–#40`) | 8 | 8/8 | 6 | 6 | **1.59x** | `#38` L1Norm **3.69x**, `#39` L2Norm **2.99x**, `#40` LayerNorm **2.23x**, `#35` GroupNorm **1.51x** |
| Pooling (`#41–#46`) | 6 | 6/6 | 6 | 6 | **2.79x** | `#46` AvgPool3D **9.60x** (native Metal vs CPU), `#41` MaxPool1D **7.50x**, `#44` AvgPool1D **4.22x** |
| Axis Reductions (`#47–#49, #51–#53`) | 6 | 6/6 | 4 | 4 | **1.56x** | `#52` Argmin **3.75x**, `#51` Argmax **3.22x**, `#53` Min **1.09x**, `#49` Max **1.06x** |
| Standard / Pointwise / Depthwise Conv | 18 | 18/18 | 14 | 16 | **2.07x** | `#66` Conv3D **10.41x**, `#67` Conv1D **9.98x**, `#60` Conv3D **9.54x**, `#59` Conv3D **8.96x**, `#87` PW2D **2.12x** |
| Transposed Conv | 17 | 17/17 | 14 | 14 | **2.83x** | `#72` ConvT3D **23.54x**, `#77` ConvT3D **19.30x**, `#73` ConvT3D **13.72x**, `#81` ConvT2D **2.51x** |
| Prefix Scans (`#89–#93`) | 5 | 5/5 | 5 | 5 | **1.84x** | `#92` Exclusive **3.99x**, `#91` Reverse **2.91x**, `#93` Masked **1.82x**, `#89/#90` **1.00x** (371 GB/s) |
| Fused Losses & FlashAttention (`#94–#100`) | 7 | 7/7 | 6 | 6 | **2.89x** | `#100` Hinge **16.65x**, `#98` KLDiv **4.93x**, `#99` Triplet **3.71x**, `#97` FlashAttention-M1 **1.04x** |
| **Total** | **100** | **100/100** | **71** | **85** | **1.9268x** | Full 100-task table in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |

Full raw JSON artifact: [.audit/eval_final_100.json](.audit/eval_final_100.json).

## Hardware Envelope (Apple M1 Max, `[measured]`)

Measured via frozen [evals/roofline_probe.py](evals/roofline_probe.py) ([.audit/roofline_m1max.json](.audit/roofline_m1max.json)):

| Metric | Theoretical | Measured |
| --- | ---: | ---: |
| Memory Bandwidth (`stream_copy`, 512 MB) | 400.0 GB/s | **373.83 GB/s** (IQR 0.011 ms) |
| Memory Bandwidth (`stream_triad`, 805 MB) | 400.0 GB/s | **368.86 GB/s** (IQR 0.048 ms) |
| FP32 SGMM (`simdgroup_float8x8`) | 10.4 TFLOPS | **10.171 TFLOPS** |
| FP16 SGMM (`simdgroup_half8x8`) | 10.4 TFLOPS | **9.634 TFLOPS** |
| Threadgroup SRAM / SIMD Width | 32 KB / 32 | 32,768 B / 32 threads |

## Architecture

```text
python/m1triton/          Thin Python AST frontend (@jit, tl.*, tlx.*, autotuner)
        |  (pybind11 _C IRBuilder)
        v
cpp/ir/                   Pure C++20 Typed Block SSA IR, TargetSpec, Verifier, Printer/Parser
        |  (TargetSpec-only IR-to-IR transforms, zero Metal/runtime coupling)
        v
cpp/passes/               15 passes: Canonicalize, CSE, LICM, EpilogueFusion, StructurePruning,
        |                 OnlineSoftmaxWelford, PoolLowering, ConvLowering, LossLowering,
        |                 FlashAttentionLowering, Contiguity, TileLayout, ThreadgroupStaging,
        |                 SimdgroupMatrixLowering, DCE
        v
cpp/codegen/metal/        Shared msl_common + 11 specialized MSL emitters (SGMM, Winograd/Implicit-GEMM
        |                 Conv, DW/Pool, Online Softmax/Norm, Row Scan, Loss, FlashAttention, Generic)
        v
cpp/runtime/              Objective-C++ (.mm) Metal UMA runtime: zero-copy MPS/CPU buffers,
                          newLibraryWithSource + MTLBinaryArchive disk cache, PSO threadgroup
                          limit validation, async command queue, dispatch counter
```

Architectural boundaries (`ARCH-IR-PURE`, `ARCH-PASSES-NO-RUNTIME`, `ARCH-FRONTEND-NO-TORCH-CHEAT`, `FROZEN-EVAL`, `COMMENT-HYGIENE`) are enforced mechanically on every CI run by [tools/check-guardrails.mjs](tools/check-guardrails.mjs) against [guardrails.json](guardrails.json).

## Quickstart

Requirements: macOS on Apple Silicon with CommandLineTools (`/usr/bin/clang++`, `Metal.framework` — no Xcode / `xcrun metal` needed), CMake, Ninja, Python 3.11+, `pybind11`, `torch`, `pytest`, Node.js.

```bash
# 1. Full local CI parity (guardrails, feature map audit, C++ build + 15 CTests, 62 Pytests)
bash scripts/ci_local.sh

# 2. Standalone C++ IR pass inspection and MSL codegen CLI
./build/m1triton-opt --pipeline=m2 --dump-ir --emit-msl path/to/kernel.m1ir

# 3. Run the SHA-256-frozen KernelBench Level 1 evaluator on any task slice
PYTHONPATH=python:build python3 evals/eval_kernelbench.py \
  --tasks 1..18 --seeds 3 --warmup 10 --trials 30 \
  --roofline .audit/roofline_m1max.json --out .audit/eval_slice.json

# 4. Verify frozen eval integrity
node tools/check-guardrails.mjs
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Layered compiler architecture, mechanically enforced boundaries, UMA runtime design
- [docs/PASS_PIPELINE.md](docs/PASS_PIPELINE.md) — All 15 pure C++20 SSA optimization passes and `m1triton-opt` usage
- [docs/DSL_CODESIGN.md](docs/DSL_CODESIGN.md) — Compiler-DSL Co-design primitives (`tlx.*`), hardware mechanisms, and verified speedups
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — Complete 100-task KernelBench Level 1 official frozen scoreboard
- [.audit/experiments.tsv](.audit/experiments.tsv) — M7 performance sprint experiment ledger (kept and rejected hypotheses)
