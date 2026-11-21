# GOAL: M1Triton (Apple M1 Max Specialized Triton-Compatible + Co-Designed Compiler)

## 1. Mission Statement

Build **M1Triton**, a production-grade engineering compiler written with a **thin Python frontend** and a **C++20 / Objective-C++ core + Metal/AGX backend**, capable of compiling and running Triton-compatible kernels and Co-designed DSL kernels natively on **Apple M1 Max**.

Success means proving the advantage of **hardware-specialized Compiler + DSL Co-design** on a concrete, tamper-proof benchmark suite:
1. **100/100 Correctness on KernelBench Level 1**: Every single task (`1` through `100` from `ScalingIntelligence/KernelBench` Level 1) compiles through M1Triton's own frontend, C++ SSA IR, optimization passes, MSL codegen, and Metal runtime, and matches reference PyTorch outputs within strict numerical tolerances.
2. **Zero Cheating / Zero Framework Fallbacks**: Every evaluated operator MUST execute compiled M1Triton Metal kernels (`compiled_kernel_dispatch_count > 0`). Hidden fallbacks to `torch.*`, `MPSGraph`, `Accelerate/BLAS`, or external kernel libraries inside candidate kernels are strictly forbidden.
3. **Peak M1 Max Performance via Compiler-DSL Co-Design**: Push memory-bound kernels toward M1 Max's 400 GB/s unified memory bandwidth and compute-bound kernels toward M1 Max's `simdgroup_matrix_multiply_accumulate` throughput, beating the M1 Max PyTorch MPS baseline (and tracking geomean speedup + beats-baseline count across all 100 tasks). When vanilla Triton syntax hides hardware optimization structure (e.g., triangular/symmetric/diagonal sparsity, explicit SIMD-group tiling, split-K, online softmax fusion, hardware prefix scan), co-design reusable, composable DSL primitives/hints while preserving exact mathematical semantics.
4. **Exemplary Compiler Engineering Hygiene**: Clean layered C++/Python repository, mechanical architecture guardrails, GitHub PR/CI workflow, zero AI slop, zero redundant mock-only tests, and synchronized technical documentation.

---

## 2. Target Hardware and Verified Local Environment (`[measured]`)

All facts below were measured directly on this machine before launch:

- **Hardware**: MacBook Pro (`MacBookPro18,2`), **Apple M1 Max**
  - CPU: 10 cores (8 Performance + 2 Efficiency)
  - GPU: **32 Apple GPU cores**, Metal 4 support, `simdWidth = 32`, `maxThreadsPerThreadgroup = 1024`, `maxThreadgroupMemoryLength = 32768` bytes (32 KB)
  - Memory: **32 GB Unified Memory (UMA)**, **400 GB/s** theoretical peak bandwidth (512-bit LPDDR5)
  - Compute ceiling: ~10.4 TFLOPS FP32 / ~20.8 TFLOPS FP16 via cooperative `simdgroup_matrix_multiply_accumulate` (8x8 fragments)
- **Toolchain**:
  - Apple Clang 21.0.0 (`/usr/bin/clang++`), macOS SDK with `Metal.framework/Headers/Metal.h`
  - **CRITICAL TOOLCHAIN FACT**: This machine has CommandLineTools only (no `/Applications/Xcode.app`, so `xcrun metal` CLI is absent). **Native runtime compilation via Objective-C++ `MTLDevice::newLibraryWithSource` + `MTLBinaryArchive` disk caching is verified working (`METAL_OK`).** Do NOT waste time looking for `xcrun metal`.
  - CMake (`/Users/zhangshuoming/miniconda3/bin/cmake`), Ninja (`/Users/zhangshuoming/miniconda3/bin/ninja`)
  - LLVM / MLIR 18.1.8 (`/opt/homebrew/opt/llvm/bin/clang++`, `llvm-config`, `mlir-opt`), optional reference/utility; prefer a lean custom C++20 Block SSA IR unless a concrete pass demands MLIR
  - Python 3.11.4 (`/Users/zhangshuoming/miniconda3/bin/python3`), `pybind11 2.13.6`, `torch 2.4.1` (`mps: True`), `numpy 2.3.2`, `pytest 9.0.2`, `datasets 4.6.1`, `scipy 1.16.1`, `uv`
  - GitHub CLI `gh` authenticated as `zhangshuoming990105` (scopes: `repo`, `workflow`)
- **Permitted References (Read-Only Inspiration)**:
  - `/Users/zhangshuoming/workspace/triton` (Triton frontend syntax and semantics reference)
  - `/Users/zhangshuoming/workspace/msl_tutorial` (Metal performance testing reference)
  - Any public literature, Apple Metal docs, MLX Metal kernel idioms, FlashAttention, and CUTLASS/Triton tiling papers. All shipped code must be cleanly authored to fit M1Triton's own layered architecture.

---

## 3. System and Compiler Architecture

Design and enforce strict layer separation from Day 1 via `guardrails.json`:

```text
python/m1triton/          # Thin Python frontend: @triton.jit compat, tl.* & tlx.* Co-design DSL, AST->IR builder, autotuner
cpp/
  ir/                     # Pure C++20 Typed Block SSA IR, TargetSpec, Layout/Structure attrs, Text Printer/Parser, Verifier
  passes/                 # Pure C++20 IR-to-IR optimization pipeline (zero Metal/runtime headers)
  codegen/metal/          # Lower optimized IR -> compact, human-readable MSL kernels + IR/MSL dump inspection
  runtime/                # Objective-C++ (.mm) Metal runtime: zero-copy UMA tensor bridge, MTLBinaryArchive cache, async queue batching, GPU timestamps, dispatch counter
  bindings/               # Thin pybind11 module exposing C++ compiler + runtime to Python
benchmarks/
  kernelbench_l1/         # Pinned real ScalingIntelligence/KernelBench Level 1 tasks (1..100) + manifest SHA
  kernels/                # M1Triton implementations for KB L1 tasks (using standard tl.* and co-designed tlx.*)
evals/
  eval_kernelbench.py     # FROZEN (SHA-256 locked in guardrails.json) correctness + performance + anti-cheat harness
  roofline_probe.py       # FROZEN M1 Max empirical bandwidth & TFLOPS probe
tests/                    # C++ CTest unit/pass tests + Python behavior/smoke/IR-roundtrip tests
docs/                     # ARCHITECTURE.md, DSL_CODESIGN.md, PASS_PIPELINE.md, BENCHMARKS.md
scripts/                  # ci_local.sh, fetch_kernelbench.py, report_scoreboard.py
.github/workflows/ci.yml  # GitHub Actions CI mirroring scripts/ci_local.sh
```

### Mandatory Architecture Boundaries (`guardrails.json`)
- `ARCH-IR-PURE`: `cpp/ir/` MUST NOT `#include` `cpp/runtime`, `cpp/codegen`, `cpp/bindings`, or `Metal`.
- `ARCH-PASSES-NO-RUNTIME`: `cpp/passes/` MUST NOT `#include` `cpp/runtime`, `cpp/bindings`, or `Metal`; hardware facts flow in through `TargetSpec`.
- `ARCH-FRONTEND-NO-TORCH-CHEAT`: `benchmarks/kernels/` MUST NOT call `torch.nn.functional`, `torch.matmul`, `torch.bmm`, `torch.einsum`, `torch.conv`, `torch.softmax`, `torch.layer_norm`, etc., in candidate forward execution (enforce mechanically via `bannedPatterns` in `guardrails.json` + runtime dispatch verification).

### Core Compiler Principles to Embody
1. **Real Compiler Pipeline (Not String-Template Cheating)**: Python parses `@triton.jit` / `@m1.jit` AST into C++ SSA IR operations. Every optimization is an inspectable, unit-tested C++ pass over IR (`--dump-ir`, `--dump-msl`), with a strict `IRVerifier` running between passes in debug/test builds.
2. **Core Optimization Passes (Add Incrementally and Test Individually)**:
   - `Canonicalize`, `CSE`, `DCE`, `LoopInvariantCodeMotion`
   - `ContiguityAndAlignmentAnalysis` (proves 128-bit `float4`/`half4` vectorized loads/stores)
   - `TileAndLayoutAssignment` (maps block tiles to M1 Max 32-thread SIMD-groups and 32 KB threadgroup memory)
   - `ThreadgroupStagingAndSwizzlePass` (bank-conflict-free shared memory staging + double-buffering)
   - `SimdgroupMatrixLoweringPass` (lowers `tl.dot`, implicit-GEMM conv, and attention matmuls to Apple `simdgroup_matrix_multiply_accumulate` 8x8 register tiles)
   - `SimdReductionAndScanLoweringPass` (lowers reductions/scans to hardware `simd_sum`, `simd_max`, `simd_shuffle_xor`, `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`)
   - `StructureAwareDeadTilePruningPass` (prunes zero/redundant tiles for triangular, diagonal, symmetric, and causal-masked workloads)
   - `EpilogueFusionAndBufferEliminationPass` (fuses bias/activation/norm/loss epilogues to eliminate global memory round-trips)
   - `RooflineGuidedAutotuner` (prunes tile configs using M1 Max 32 KB threadgroup + register occupancy model; caches best configs on disk)
3. **Zero-Copy UMA Runtime Design (System + Compiler Co-Design)**:
   - On Apple Silicon Unified Memory, CPU and GPU share physical DRAM. Bridge PyTorch MPS tensors (via native `MTLBuffer` handle / storage offset extraction) and contiguous CPU tensors (via `newBufferWithBytesNoCopy` page-aligned zero-copy) without extra host-device copies.
   - Minimize dispatch latency via persistent `MTLComputePipelineState` caching, `MTLBinaryArchive` disk caching, command buffer batching, and avoiding per-kernel synchronous waits during multi-kernel execution.
4. **Compiler-DSL Co-Design Discipline**:
   - Standard Triton subset (`@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, `tl.dot`, `tl.sum`, `tl.max`, `tl.min`, `tl.where`, `tl.constexpr`, `triton.autotune`, `kernel[grid](...)`) MUST work cleanly without custom hints.
   - Extend with clean, reusable Co-design primitives (`tlx.*` / `@m1.jit` hints) when hardware specialization yields real gains across operator families, such as:
     - `tlx.structure("upper_triangular" | "lower_triangular" | "symmetric" | "diagonal")`
     - `tlx.tile_spec(bm=..., bn=..., bk=..., double_buffer=True, split_k=...)`
     - `tlx.online_softmax(...)` / `tlx.welford_norm(...)`
     - `tlx.simd_scan(op="add"|"mul", exclusive=..., reverse=...)`
     - `tlx.implicit_gemm_conv(...)` / `tlx.layout_hint("nhwc" | "nchw")`
     - `tlx.persistent_grid(...)` / `tlx.vec_width(4)`
   - **Rule**: No single-problem hacks. Every DSL extension must lower through general C++ IR/passes, be documented in `docs/DSL_CODESIGN.md`, and include standalone compiler tests.

---

## 4. Frozen Evaluation & Anti-Cheat Protocol

1. **Authentic Benchmark Source**: Fetch all 100 tasks of `ScalingIntelligence/KernelBench` `KernelBench/level1/` into `benchmarks/kernelbench_l1/` and record the exact upstream Git commit SHA in `benchmarks/kernelbench_l1/MANIFEST.json`.
2. **Baseline Definition**:
   - Primary baseline: Reference `Model` from each KernelBench task running on **PyTorch MPS eager** on this M1 Max (using exact task `get_init_inputs()` / `get_inputs()`).
   - If a specific PyTorch op is unimplemented or crashes on PyTorch 2.4.1 MPS (e.g., certain 3D ops), record `baseline_backend = "cpu_fallback_mps_unsupported"` explicitly in the report, compare against PyTorch CPU, and NEVER silently drop the task.
   - Also run `evals/roofline_probe.py` to log hardware utilization (`% of measured M1 Max STREAM bandwidth` and `% of measured FP32/FP16 SGMM TFLOPS`).
3. **Strict Correctness Gate**:
   - At least 3 independent randomized input seeds per task plus shape-edge tests in `tests/`.
   - Strict tolerances matching KernelBench (default `rtol=1e-2, atol=1e-2` for FP16, `rtol=1e-4, atol=1e-4` for FP32, or `1e-3` only for deep K>=4096 FP32 accumulations, clearly declared in the frozen evaluator and uniform across all tasks).
   - Anti-cheat assertions inside `eval_kernelbench.py`: verify `m1triton.runtime.get_dispatch_count()` incremented strictly during candidate `forward()` AND verify candidate module contains zero forbidden `torch` compute op calls.
4. **Measurement Rigor**:
   - Separate cold compile time from steady-state GPU execution time.
   - Warmup (`>= 10` runs), timed trials (`>= 30` runs with `torch.mps.synchronize()` / Metal GPU timestamps), report median and IQR; compute geometric mean speedup across all 100 tasks.
5. **Freeze Immediately**: Once `evals/eval_kernelbench.py` and `evals/roofline_probe.py` pass both a valid smoke test and a deliberately degraded/cheating negative control (confirming the evaluator rejects wrong answers and catches fallback cheats), freeze both files via `lock-eval.mjs` into `guardrails.json`. Never edit frozen eval files afterward.

---

## 5. Milestone Roadmap (M0 -> M8)

Track every milestone in `.audit/progress.json` via `/cstack:mission-control` (`progress-ledger.mjs`). Never mark a milestone `verified` without executable proof.

- **`M0`: Repo Foundation, Guardrails, GitHub PR/CI & Frozen Eval**:
  - Initialize Git repo (if not yet initialized), `.gitignore`, CMake + Ninja + Apple Clang + pybind11 + Metal build skeleton, `guardrails.json` (`cstack project-init --mode systems`), `AGENTS.md` Rule Ledger, `.github/workflows/ci.yml` + `scripts/ci_local.sh` (`/cstack:git-pr-ci`), `/cstack:env-verify` verification driver.
  - Fetch all 100 KernelBench Level 1 tasks, build and vet `evals/eval_kernelbench.py` + `evals/roofline_probe.py` (with positive & negative anti-cheat tests), and SHA-256 freeze them via `lock-eval.mjs`.
  - Initialize `.audit/progress.json` with `M0..M8` and record initial M1 Max empirical roofline numbers + PyTorch MPS baseline timing snapshot across Level 1.
- **`M1`: End-to-End Vertical Slice (Python Frontend -> C++ SSA IR -> MSL -> Metal Runtime)**:
  - Implement core Block SSA IR + IR Verifier + text dumper, basic canonicalization/vectorization passes, MSL codegen, Objective-C++ Metal runtime with UMA zero-copy bridge + binary cache + dispatch counter, and thin Python `@triton.jit` subset.
  - Prove end-to-end execution and correctness on `vector_add`, `saxpy`, `fused_add_relu`, and naive `matmul`, plus C++ CTest and Python pytest green in CI.
  - Verify `vector_add` reaches `>= 75%` of measured M1 Max STREAM bandwidth.
- **`M2`: High-Performance SGMM GEMM Engine & KernelBench Batch 1 (`#1-#18`, 18 tasks)**:
  - Build `SimdgroupMatrixLoweringPass`, `ThreadgroupStagingAndSwizzlePass` (double-buffered 32 KB threadgroup tiling), `split_k` for Large-K (`#6`), specialized SIMD-reduced GEMV (`#4`), batched/3D/4D/transposed GEMM (`#3, #10, #11, #16-#18`), irregular/tall-skinny tiling (`#7-#9`), and `StructureAwareDeadTilePruningPass` + `tlx.structure` Co-design hints for diagonal/symmetric/triangular matmuls (`#12-#15`).
  - Gate: `18/18` passing strict correctness; track baseline speedup on every task.
- **`M3`: Vectorized Elementwise, Online Softmax & Normalization (`#19-#40, #88`, 23 tasks)**:
  - Build `ContiguityAndAlignmentAnalysis` (128-bit `float4`/`half4` vectorization), `EpilogueFusionPass`, single-pass online Softmax/LogSoftmax (`#23, #24`), fused Welford & SIMD reductions for BatchNorm/InstanceNorm/GroupNorm/RMSNorm/Frobenius/L1/L2/LayerNorm (`#33-#40`), and all activations including MinGPTNewGelu (`#19-#32, #88`).
  - Gate: `41/100` cumulative passing; memory-bound elementwise/norm kernels approaching M1 Max memory bandwidth ceiling.
- **`M4`: Pooling, Reductions & Hardware Prefix Scans (`#41-#49, #51-#53, #89-#93`, 17 tasks)**:
  - Implement 1D/2D/3D MaxPool & AvgPool (`#41-#46`), generic axis reductions Sum/Mean/Max/Min/Argmax/Argmin (`#47-#49, #51-#53`), and `SimdReductionAndScanLoweringPass` using Metal hardware `simd_prefix_inclusive_sum` / `simd_prefix_exclusive_sum` + decoupled/hierarchical block scan for `cumsum`, `cumprod`, `cumsum_reverse`, `cumsum_exclusive`, `masked_cumsum` (`#89-#93`).
  - Gate: `58/100` cumulative passing.
- **`M5`: Convolution Compiler Lowering & Specialized Microkernels (`#50, #54-#87`, 35 tasks)**:
  - Implement compiler-level Implicit-GEMM lowering reusing the M2 `simdgroup_matrix` engine for standard 1D/2D/3D convolutions (symmetric/asymmetric, strided, padded, dilated, grouped), specialized 1x1 pointwise GEMM lowering (`#87`), dedicated depthwise / depthwise-separable 2D microkernels (`#82-#86`), and strided/dilated/grouped transposed 1D/2D/3D convolution kernels (`#57, #58, #61, #64, #65, #68-#75, #77-#79, #81`).
  - Gate: `93/100` cumulative passing.
- **`M6`: Fused Losses & FlashAttention-M1 (`#94-#100`, 7 tasks) -> 100/100 Coverage**:
  - Implement fused MSE, CrossEntropy, Huber, KLDiv, TripletMargin, Hinge losses (`#94-#96, #98-#100`), and FlashAttention-style tiled online softmax + `simdgroup_matrix` fused `ScaledDotProductAttention` (`#97`) inside M1 Max's 32 KB threadgroup SRAM without materializing N×N attention matrices.
  - Gate: **100/100 KernelBench Level 1 passing strict frozen correctness**.
- **`M7`: Compiler-DSL Co-Design Performance Sprint (Hillclimbing Loop)**:
  - Rank all 100 tasks by `our_ms / baseline_ms` and hardware roofline gap.
  - Run disciplined `/cstack:research-lab` optimization loops (interleaved A/B timing, noise-floor gate, mechanism explanation, 2-strike premise circuit breaker, logged in `.audit/experiments.tsv`) targeting the slowest clusters first:
    - Dispatch overhead elimination & command-buffer fusion
    - Register tiling / occupancy tuning & threadgroup bank-conflict elimination
    - Co-design DSL syntax/hint upgrades (`tlx.*`) where semantics allow skipping work or fusing memory traffic
    - Roofline-guided autotuner sweeps with cached best schedules
  - Target: Maximize `beats_baseline` count and overall geometric mean speedup across all 100 tasks (stretch goal: geomean speedup `> 1.5x` vs PyTorch MPS and `>= 85/100` tasks faster than or at parity with baseline, while keeping 100/100 strictly correct).
- **`M8`: Full-Repo Unslop, Documentation Synthesis, Adversarial Interrogation & Final Release**:
  - Run full `/cstack:unslop` pass across Python, C++, MSL, tests, and docs.
  - Run multi-angle `/cstack:interrogate` (`slop-critic` on `pro`) checking correctness edges, anti-cheat compliance, memory safety/leaks, and architecture cleanliness.
  - Complete clear, evidence-backed docs: `README.md`, `docs/ARCHITECTURE.md`, `docs/PASS_PIPELINE.md`, `docs/DSL_CODESIGN.md`, and `docs/BENCHMARKS.md` (with full 100-task table).
  - Run final clean CI (`bash scripts/ci_local.sh`), frozen-eval verification (`lock-eval.mjs --check`), and full 100/100 `evals/eval_kernelbench.py` run; record final verified checkpoint in `.audit/progress.json`.

---

## 6. Autonomous 24h+ Execution & Multi-Agent Orchestration Playbook

This mission is authorized to run unattended for 24h+ with `cstack` preset `autonomous-systems` (`budget=autonomous`). Do NOT pause to ask routine questions on reversible engineering choices.

### Dynamic Specialist Team (Define via `define_subagent` / `swarm` / `run_workflow`)
Allocate model tiers intelligently to match task complexity:
- **`compiler-architect` (`pro`)**: Designs SSA IR invariants, pass interfaces, roofline performance models, and Co-design DSL syntax extensions; diagnoses subtle correctness or performance regressions.
- **`frontend-dsl-eng` (`inherit`)**: Owns `python/m1triton/` AST lowering, Triton API compatibility, `tlx.*` Co-design hints, and `cpp/bindings/` pybind11 glue.
- **`cpp-ir-pass-eng` (`inherit`)**: Owns `cpp/ir/` and `cpp/passes/` plus C++ CTest pass unit tests.
- **`metal-kernel-eng` (`inherit`, escalate to `pro` on hard SGMM/Conv/FlashAttention tuning)**: Owns `cpp/codegen/metal/`, `cpp/runtime/` Objective-C++ UMA/Metal engine, and micro-architecture tuning.
- **`evidence-scout` (`flash`)**: Reads reference repos (`/Users/zhangshuoming/workspace/triton`, `msl_tutorial`), extracts KernelBench task shapes/specs, and triages bulk build/test logs.
- **`slop-critic` (`pro`)**: Adversarial read-only verifier hunting fake fallbacks, weak tests, precision cheats, dead code, and unverified claims.

### Batch Orchestration (`run_workflow` and `swarm`)
- Use `swarm` (`workspace: "share"`) when multiple specialists implement disjoint layers of a milestone concurrently (strictly disjoint file ownership; never edit the same file in two parallel workers).
- Use `run_workflow` Python scripts for batched operator bringup, batched KernelBench failure triage, and multi-angle `slop-critic` audits across operator families.

### GitHub PR / CI & Branch Workflow (`/cstack:git-pr-ci`)
- Never commit unverified code directly to `main`. Every milestone or major pass lands via a scoped feature branch (`feat/m1-vertical-slice`, `perf/m2-sgmm-double-buffer`, `chore/hygiene-m2`, etc.).
- Run `bash scripts/ci_local.sh` before every merge/PR.
- If a GitHub `origin` remote is configured, push branch -> `gh pr create` -> `gh pr checks --watch` -> fix any CI failures -> `gh pr merge --squash --delete-branch`.
- If no `origin` remote exists yet, do NOT block: run full local CI parity + adversarial check and merge cleanly into `main` with `git merge --no-ff`.

### Compaction Resilience & Anti-Slop Cadence (`/cstack:mission-control`)
- Update `.audit/progress.json` after every milestone and benchmark sweep; record git commit checkpoints before any compaction or major phase shift.
- Every 2 verified milestones (`Hygiene: DUE`), execute a mandatory cleanup cycle: `node tools/check-guardrails.mjs` + `lock-eval.mjs --check` + `/cstack:unslop` + docs sync + `slop-critic` audit, then record `checkpoint --hygiene-clean`.
- Respect the 2-strike premise circuit breaker in performance tuning: if two experiments under one optimization hypothesis fail to beat the noise floor, stop tweaking knobs, profile the actual hardware limiter (GPU occupancy, register spill, bandwidth, dispatch latency), and open a new hypothesis.
