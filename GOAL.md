# GOAL: M1Triton v2.0 — KernelBench Level 2 (Multi-Operator Fusion & Graph-to-Tile Compiler-DSL Co-Design on Apple M1 Max)

> **Baseline Archive**: M1Triton v1.0 (`git tag v1.0-kernelbench-l1`) achieved **100/100** official SHA-256-frozen KernelBench Level 1 correctness with **1.9268x geomean speedup** (`71/100 > 1.00x`, `85/100 >= 0.98x`) against PyTorch 2.15.0 MPS on Apple M1 Max. All L1 tests and guardrails remain mandatory regression gates in v2.0.

---

## 1. Mission Statement

Evolve **M1Triton** from a single-operator kernel compiler (v1.0) into a **fusion-capable, multi-level Compiler + Co-designed DSL** targeting **KernelBench Level 2** (`ScalingIntelligence/KernelBench` `KernelBench/level2/`, 100 multi-operator fusion workloads) natively on **Apple M1 Max**.

In Level 2, individual tasks chain 2–6 operators (`Conv/ConvTranspose/Matmul/Gemm` + `Bias/Scale/Divide/Residual` + `BatchNorm/GroupNorm/InstanceNorm/LayerNorm` + `Activations` + `Pooling/LogSumExp/Softmax/Reductions`). Running Level 2 as unfused L1 kernels hits Apple Silicon's **memory bandwidth wall** (374 GB/s DRAM) due to repeated multi-GB intermediate tensor round-trips. True compiler-DSL co-design eliminates those intermediate round-trips by keeping activations inside **SIMD-group registers**, **32 KB threadgroup SRAM**, and **single-command-buffer fused schedules**.

### Success Criteria
1. **100/100 Correctness on KernelBench Level 2**: Every task (`1..100`) compiles through M1Triton's Python frontend, C++20 Fusion & SSA IR, pure C++20 optimization passes, MSL codegen, and Metal UMA runtime, matching PyTorch reference outputs within strict frozen numerical tolerances across 3 randomized seeds (including strict NaN/Inf matching and stateful weight/BN synchronization).
2. **Zero Cheating & Genuine Operator Fusion**:
   - Zero PyTorch/MPSGraph/SciPy compute fallbacks (`ARCH-FRONTEND-NO-TORCH-CHEAT` + dynamic `TorchDispatchMode` ATen tripwire + `compiled_kernel_dispatch_count > 0`).
   - **Measurable Fusion Discipline**: Track per-task `fused_dispatch_count` vs unfused op count; chains eligible for epilogue/elementwise/reduction/pool fusion MUST fuse through general C++ IR passes rather than falling back to unfused L1 dispatch sequences or bespoke single-task string templates.
3. **Peak Multi-Op Fusion Performance on Apple M1 Max**:
   - Beat PyTorch MPS eager across KernelBench Level 2 via DRAM traffic elimination, register epilogue fusion, SRAM producer-consumer tiling, and single-command-buffer batching.
   - **Target**: **100/100 passing**, **geomean speedup `>= 2.0x`** vs PyTorch MPS eager, and **`>= 85/100` tasks strictly faster (`> 1.00x`)** than baseline (with `>= 92/100` at parity or faster `>= 0.98x`).
4. **Non-Regression & Exemplary Compiler Engineering**:
   - Preserve 100% of v1.0 CI (`bash scripts/ci_local.sh`, `15+` CTests, `62+` Pytests, L1 frozen lock intact).
   - Follow **Hardware-Target-First IR Design**: for every fusion family, prototype and hardware-verify the target fused MSL schedule (`src -> dst` IR golden cases) before writing the general C++ lowering pass.
   - Keep `main` ruthlessly unslopped: clean human docs, zero dead code, zero redundant abstractions, periodic `slop-critic` audits every 2 milestones.

---

## 2. Target Hardware Envelope (`[measured]`, Apple M1 Max)

| Resource | Apple M1 Max Envelope | v2.0 Fusion Implication |
| --- | --- | --- |
| GPU Cores / SIMD Width | 32 cores / 32 threads | Prefer 4–8 SG (128–256 threads) for fused compute epilogues; guard every PSO via `maxTotalThreadsPerThreadgroup` |
| Threadgroup SRAM | 32,768 bytes (32 KB) | Hard ceiling for producer-consumer SRAM tiles (`Conv/Matmul -> Pool/Norm/Reduce`) and double-buffered SGMM staging |
| Unified Memory Bandwidth | 400 GB/s peak / 373.83 GB/s measured | Every eliminated intermediate tensor write+read saves `2 * bytes / 374 GB/s` — primary lever on Level 2 |
| FP32 / FP16 SGMM Compute | 10.171 / 9.634 TFLOPS measured | Fused register epilogues (`bias/act/residual/scale`) hide completely behind SGMM/Conv compute latency |
| Command Submission | Private `MTLCommandQueue` | Multi-stage fused pipelines (e.g. `Conv+Act -> GlobalPool/Reduce`) should batch inside one `MTLCommandBuffer` without host stalls |

---

## 3. v2.0 Compiler Architecture & Co-Design Roadmap

All existing architectural boundaries (`ARCH-IR-PURE`, `ARCH-PASSES-NO-RUNTIME`, `ARCH-FRONTEND-NO-TORCH-CHEAT`, `FROZEN-EVAL`, `COMMENT-HYGIENE`) remain mechanically enforced in `guardrails.json`.

### 3.1 Multi-Level IR Extensions (`cpp/ir/`, pure C++20)
- **Composable Epilogue DAG IR (`EpilogueDAGAttr` / `Region`)**:
  - Attach typed, verifier-checked SSA epilogue subgraphs (elementwise unary/binary/clamp/select, channel-broadcast bias/scale/BN-affine, residual tensor add/mul) directly onto `OpKind::Dot`, `OpKind::ImplicitGemmConv`, and Winograd output transforms.
- **Fused Producer-Consumer Tail Attributes (`FusedTailAttr`)**:
  - Represent local pooling (`MaxPool`/`AvgPool` over spatial output tile), channel/spatial reductions (`Sum`/`Mean`/`Max`/`Min`/`LogSumExp`/`Softmax`), and normalization tails (`GroupNorm`/`InstanceNorm`/`LayerNorm`/`BatchNorm`) cleanly in IR so passes can inspect, fuse, or split them based on `TargetSpec`.
- **IR Verifier & Roundtrip Printer/Parser**:
  - Every new fusion op/attr must roundtrip through `IRPrinter`/`IRParser` and pass `IRVerifier` before and after every pass.

### 3.2 New Pure C++20 Optimization & Fusion Passes (`cpp/passes/`)
1. **`AlgebraicNormFoldingPass`**: Folds inference `BatchNorm` (running mean/var + affine weight/bias) and constant `Scale`/`Divide`/`Multiply` into producer `Conv`/`Linear` weights & biases or epilogue affine constants when mathematically exact (verified within strict FP32 tolerance).
2. **`EpilogueDAGFusionPass`**: Fuses chains of `Bias + Residual + Scale/Div/Sub/Add + Activation (ReLU/LeakyReLU/GELU/Swish/Mish/Sigmoid/Tanh/HardSwish/HardSigmoid/HardTanh/Clamp/ELU/SELU/Softplus/Tanh-GELU)` directly into SGMM (`sgmm_emitter`), ConvStd (`conv_std_emitter`), ConvTranspose (`conv_transpose_emitter`), Depthwise (`conv_dw_emitter`), and Winograd (`F(2x2)` / `F(4x4)`) register-tile store epilogues.
3. **`ProducerConsumerSRAMFusionPass`**: Fuses `Producer(Conv/ConvTranspose/Matmul) -> LocalPool(Max/Avg)` and `Producer -> Row/Spatial Reduction (Mean/Sum/Max/Min/LogSumExp/Softmax)` inside 32 KB threadgroup SRAM or SIMD-group register reductions without materializing full 4D intermediates in global memory.
4. **`ElementwiseReductionDAGFusionPass`**: Fuses multi-op elementwise + broadcast + residual + reduction DAGs (including multi-input/forked graphs) into a single `float4`/`packed_float4` kernel with strict IEEE-754 NaN preservation.
5. **`FusionOccupancyCostModelPass`**: Evaluates register pressure, 32 KB SRAM budget, and arithmetic intensity against `TargetSpec` to choose:
   - Single-kernel register epilogue fusion,
   - On-chip SRAM producer-consumer fusion, or
   - Safe 2-stage split inside a single `MTLCommandBuffer` when register pressure would breach Apple M1 Max's occupancy cliff.

### 3.3 Composable Compiler-DSL Co-Design (`python/m1triton/tlx.py`)
Design reusable, composable primitives (zero single-problem hacks; every primitive lowers through general C++ passes and ships with standalone C++/Python compiler tests):
- `tlx.epilogue(ops=[...])` / `@tlx.epilogue` builder: composable post-op DAG attached to `tl.dot`, `tlx.implicit_gemm_conv`, Winograd, and transposed conv emitters.
- `tlx.fuse_reduce(kind=..., axis=..., keepdim=...)`: fuses trailing reductions (`sum`, `mean`, `max`, `min`, `logsumexp`, `softmax`) directly into producer epilogues via SIMD-group + partial-buffer reduction.
- `tlx.fuse_pool(kind="max"|"avg", kernel_size=..., stride=..., padding=...)`: fuses spatial pooling over producer output tiles.
- `tlx.bn_fold(weight, bias, running_mean, running_var, gamma, beta, eps)`: declarative inference BN folding lowered by `AlgebraicNormFoldingPass` (dynamically synced with PyTorch reference state — never stale across `.data` mutations!).
- `m1triton.runtime.command_batch()` context manager: batches multi-stage fused kernels into a single `MTLCommandBuffer` in `cpp/runtime/metal_runtime.mm` to eliminate inter-kernel CPU/GPU synchronization bubbles.

### 3.4 Compiler Utility & Diagnostics Upgrade
- Extend `m1triton-opt` with `--dump-ir-before-all`, `--dump-ir-after-all`, `--pass=<name>`, `--verify-fusion`, and `--stats` (printing fused op counts, SRAM bytes, estimated occupancy).
- Add lightweight Python `m1triton.profile()` helper reporting per-kernel GPU timestamps, dispatch count, PSO `maxTotalThreadsPerThreadgroup`, and achieved GB/s or TFLOPS against M1 Max roofline.
- Add `src -> dst` golden IR transformation tests under `tests/cpp/fusion/` so every fusion pass has readable input-IR -> expected-IR -> emitted-MSL regression coverage.

---

## 4. Frozen Level 2 Evaluation & Anti-Cheat Protocol

1. **Authentic Benchmark Source**: Fetch all 100 tasks of `ScalingIntelligence/KernelBench` `KernelBench/level2/` into `benchmarks/kernelbench_l2/` and record upstream commit SHA in `benchmarks/kernelbench_l2/manifest.json`.
2. **Candidate Directory**: `benchmarks/kernels_l2/task_001.py` .. `task_100.py` (cleanly separated from L1 `benchmarks/kernels/` so L1 stays untouched and regression-verified!).
3. **Frozen Evaluator (`evals/eval_kernelbench_l2.py`)**:
   - Same 3 anti-cheat layers as L1: static AST check, `TorchDispatchMode` ATen compute tripwire, native Metal `compiled_kernel_dispatch_count > 0`.
   - Additional **L2 Fusion Telemetry**: logs `dispatches_per_forward`, `cold_compile_ms`, `baseline_median_ms`, `candidate_median_ms`, `speedup`, `max_abs_err`, and `baseline_backend`.
   - Memory-safe chunk/process-isolated evaluation support (built cleanly into CLI without altering single-task semantics) so multi-GB 3D/4D L2 tasks never trigger macOS swap bloat.
   - Vet with positive + negative anti-cheat tests (wrong answer, hidden `torch.nn.functional` fallback, stale weight mutation), then **SHA-256 freeze** `evals/eval_kernelbench_l2.py` in `guardrails.json` alongside L1's frozen files.

---

## 5. Milestone Roadmap (`L2-M0` -> `L2-M7`)

Track every milestone in `.audit/progress_l2.json` via `/cstack:mission-control`. Every 2 milestones execute a mandatory unslop + docs + `slop-critic` (`pro`) hygiene checkpoint.

- **`L2-M0`: Level 2 Foundation, Corpus Audit, Frozen Eval & Baseline Snapshot**:
  - Fetch all 100 KB L2 tasks (`benchmarks/kernelbench_l2/`), classify all 100 tasks into concrete fusion families (GEMM+Epilogue, GEMM+Reduce/Norm, Conv2D/3D+Epilogue+Pool, ConvTranspose+Epilogue+Norm, Multi-Branch/Residual/Attention-like chains).
  - Build, vet, and SHA-256 freeze `evals/eval_kernelbench_l2.py`; add `benchmarks/kernels_l2` to `ARCH-FRONTEND-NO-TORCH-CHEAT` in `guardrails.json`.
  - Capture full 100-task PyTorch MPS eager baseline snapshot (`.audit/baseline_l2_mps.json`) and identify any PyTorch MPS unsupported L2 ops.
  - Upgrade `m1triton-opt` diagnostics (`--dump-ir-before-all`, `--stats`) and `MetalRuntime` single-command-buffer batching (`command_batch()`).

- **`L2-M1`: Epilogue DAG IR, `EpilogueDAGFusionPass`, `AlgebraicNormFoldingPass` & GEMM Fusion Family (~25–30 tasks)**:
  - Design & verify `EpilogueDAGAttr` in `cpp/ir/`, `EpilogueDAGFusionPass` + `AlgebraicNormFoldingPass` in `cpp/passes/`, and register-level fused epilogue codegen in `sgmm_emitter.cpp` (with `src -> dst` C++ IR tests).
  - Implement and verify all GEMM/Matmul/Linear + Bias/Scale/Activation/Residual/Clamp/Sub/Div/Norm/Reduce chains in KB L2.
  - Gate: 100% pass on GEMM fusion cluster; single-dispatch register epilogue fusion verified (`dispatches_per_forward == 1` for pure epilogue chains); beat baseline across cluster.

- **`L2-M2`: Conv2D / Conv3D Register Epilogue & BN-Fold Fusion Family (~25–30 tasks)**:
  - Extend `EpilogueDAGFusionPass` + `conv_std_emitter.cpp` (both Implicit-GEMM and Winograd `F(2x2)`/`F(4x4)` output transforms!) + `conv_dw_emitter.cpp` to emit fused `Bias + BN-fold + Activation + Scale/Clamp/Residual` inside register/tile epilogues.
  - Implement and verify all standard 2D/3D Conv + Epilogue chains in KB L2.
  - Mandatory Hygiene Checkpoint (`L2-M1` + `L2-M2`): unslop + docs + `slop-critic` (`pro`).

- **`L2-M3`: ConvTranspose2D / 3D Epilogue & Norm Fusion Family (~20–25 tasks)**:
  - Extend `conv_transpose_emitter.cpp` (including `HALO=1` and `FUSE_SW=1` active-phase schedules!) with fused register epilogues (`Bias + Activation + Residual + Scale/Clamp`) and fused BN/GN/IN/LN tails.
  - Implement and verify all ConvTranspose chains in KB L2.

- **`L2-M4`: Producer-Consumer SRAM Fusion (`Conv/Matmul -> Pool / Reduce / Softmax / LogSumExp`) -> 100/100 Coverage**:
  - Build `ProducerConsumerSRAMFusionPass` and `ElementwiseReductionDAGFusionPass` in `cpp/passes/` + SRAM tile-fusion emitters for `Conv/Matmul + MaxPool/AvgPool/GlobalAvgPool`, `Conv/Matmul + Mean/Sum/Min/Max/LogSumExp/Softmax`, and remaining multi-op chains.
  - Gate: **100/100 KernelBench Level 2 passing strict frozen correctness** (plus 100% v1.0 L1 CI non-regression!).
  - Mandatory Hygiene Checkpoint (`L2-M3` + `L2-M4`): unslop + docs + `slop-critic` (`pro`).

- **`L2-M5`: Fusion Occupancy Cost Model & `/cstack:research-lab` Performance Sprint**:
  - Rank all 100 L2 tasks by `our_ms / baseline_ms`, fusion dispatch count, and DRAM traffic vs M1 Max roofline.
  - Run disciplined `/cstack:research-lab` hillclimbing loops (logged in `.audit/experiments_l2.tsv`, 2-strike circuit breaker):
    - Convert remaining 2-dispatch chains into 1-dispatch SRAM/register fused kernels where profitable,
    - Tune `FusionOccupancyCostModelPass` tile sizes & register caps against M1 Max 32 KB SRAM / 32 cores,
    - Eliminate host-side dispatch overhead with single-`MTLCommandBuffer` batching for multi-stage reductions/norms.
  - Target: **geomean speedup `>= 2.0x`** and **`>= 85/100` beats baseline** while preserving **100/100 correctness**.

- **`L2-M6`: Full-Repo Unslop, Final Adversarial Interrogation & v2.0 Release Archive**:
  - Full-repo `/cstack:unslop` deduplicating shared epilogue/fusion codegen across emitters.
  - Adversarial `slop-critic` (`pro`) interrogation covering NaN propagation in fused epilogues, non-power-of-2 shapes, dynamic `.data` weight/BN mutation staleness, PSO `maxTotalThreadsPerThreadgroup` limits, cross-queue MPS sync, and zero anti-cheat violations.
  - Full 100/100 official unmodified frozen `evals/eval_kernelbench_l2.py` verification (`--seeds 3 --warmup 10 --trials 30`), updated `README.md`, `docs/BENCHMARKS_L2.md`, `docs/FUSION_ARCHITECTURE.md`, `docs/DSL_CODESIGN.md`, `bash scripts/ci_local.sh` green, and annotated release tag `v2.0-kernelbench-l2`.

---

## 6. Execution Discipline & Model Policy

- **Hardware-First Design Workflow**: For every new fusion pattern, design the optimal M1 Max kernel schedule first, validate `src -> dst` IR transforms on concrete examples, and then implement the general C++ pass + emitter.
- **Parallel Multi-Agent Orchestration**: Use `swarm` (`workspace: "share"`) with strictly disjoint file ownership for parallel fusion clusters and design races (recording tradeoffs in `.audit/decisions/`), `run_workflow` for batch verification/triage, and `slop-critic` (`pro`) at every 2-milestone checkpoint.
- **Model Quota Discipline**: Default to **Gemini** (`gemini-4-neon-xhigh` / `pro` / `flash`) across workers, architects, and critics; avoid Claude models unless explicitly requested by the user to conserve 5h/7d Claude quota.
- **Autonomous Watchdogs**: Attach `schedule` timers (`TimerCondition: "any"`) to every long build, swarm, or benchmark run; run heavy multi-GB GPU evaluations process-isolated and sequentially to avoid macOS swap thrashing or GPU contention.
