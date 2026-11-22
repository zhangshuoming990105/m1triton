# GOAL: M1Triton v2.0 — "Better Triton on MPS" (KernelBench L2 Fusion, L3 Complex Triton Kernels, `torch.compile`, Autotune, Metal Graphs & End-User Packaging)

> **v1.0 Baseline Archive**: `git tag v1.0-kernelbench-l1` (`e26e3d8`) achieved **100/100** official SHA-256-frozen KernelBench Level 1 correctness with **1.9268x geomean speedup** (`71/100 > 1.00x`, `85/100 >= 0.98x`) against PyTorch 2.15.0 MPS on Apple M1 Max. All L1 tests and guardrails remain mandatory regression gates in v2.0.

---

## 1. Vision & Architectural Positioning ("Better Triton on MPS")

Evolve **M1Triton** from a single-operator compiler (v1.0) into a **complete, installable, production-usable Triton compiler & graph-fusion backend for Apple Silicon MPS** that simultaneously solves:
1. **KernelBench Level 2 (100 multi-operator fusion workloads)** — eliminating multi-GB intermediate DRAM round-trips via register epilogue fusion, 32 KB SRAM producer-consumer tiling, and single-command-buffer / Metal Graph execution.
2. **Level 3 Complex Triton Kernel Suite (10 real-world production Triton algorithms)** — proving M1Triton compiles genuinely hard Triton code (FlashAttention-2 fwd+bwd, PagedAttention, Sliding-Window/ALiBi Attention, Fused RMSNorm+RoPE+KV-Cache, LayerNorm fwd+bwd with atomics, Chunked Linear Cross-Entropy fwd+bwd, Mamba SSM Scan, Persistent MoE Grouped GEMM, W4A16 Dequant-GEMM) directly via `@m1triton.jit` + `@m1triton.autotune`.
3. **`torch.compile(backend="m1triton")` + First-Class `@m1triton.autotune` + Metal Graph Execution** — integrating cleanly with PyTorch 2.15.0 MPS instead of reinventing PyTorch's frontend/allocator, while building our own compiler/runtime layers everywhere MPS falls short.
4. **Zero-Friction End-User Installation & UX** — `pip install -e .` (PEP 517 CMake build), `m1triton doctor` environment verifier, runnable `examples/`, and reproducible setup docs verified in a clean virtualenv.

---

## 2. MPS Ecosystem Analysis: What to Reuse vs. Build Ourselves (`[measured]` on macOS 26.6 + PyTorch 2.15.0)

Direct probing of this Apple M1 Max (`Apple7`, `Metal3`, PyTorch 2.15.0 MPS) established exact system boundaries:

| Capability | What Exists on MPS Today (`[measured]`) | Reuse vs. Build Decision |
| --- | --- | --- |
| **Tensor Storage, Allocator & Autograd** | PyTorch MPS UMA `MTLBuffer` allocator, `nn.Module`, `torch.autograd`, `torch.library.custom_op` | **REUSE directly** via zero-copy `data_ptr()` -> `id<MTLBuffer>` bridging and `torch.library.custom_op` (for L3 custom fwd+bwd autograd). |
| **Python Graph Capture** | PyTorch `TorchDynamo` + `AOTAutograd` FX graph capture works on MPS (`torch.compile` active) | **REUSE directly**: register `torch.compile(model, backend="m1triton")` via `torch._dynamo.register_backend` instead of building a custom Python tracer. |
| **PyTorch Inductor MPS Backend** | Works on simple pointwise/reduction chains via `torch.mps.compile_shader`, **but** keeps Conv/GEMM in `extern_kernels` (cannot fuse epilogues into Conv/SGMM, emits scalar non-SGMM MSL, no Triton frontend, no `MTLBinaryArchive` disk cache) | **DO NOT wrap as compute backend**; use as a **secondary benchmark reference** alongside PyTorch MPS eager to prove M1Triton beats both! |
| **Apple `MPSGraph`** | Closed-source black-box graph compiler behind PyTorch MPS ops | **DO NOT wrap** (forbids custom Triton algorithms, slower by `2x–23x` on 3D/transposed convs, scans, pooling, large-K GEMM; violates anti-cheat). |
| **CUDA-Graph-like Replay on Metal** | `torch.mps` exposes `Stream`/`Event`, **NO `CUDAGraph` API**; M1 Max Metal 3 natively supports **`MTLIndirectCommandBuffer` (Compute ICB = `YES`)** and single-`MTLCommandBuffer` batching | **BUILD ourselves** in `cpp/runtime/`: (1) `command_batch()` single-`MTLCommandBuffer` encoder batching, (2) `MetalGraph` capture/replay over Compute ICB (`MTLIndirectCommandBuffer`) for static multi-kernel pipelines, (3) `MTLSharedEvent` / stream-safe interop with `torch.mps.current_stream()`. Benchmark ICB vs. command-buffer batching empirically and keep what wins. |
| **Triton Compiler + Autotuner + Fusion IR** | Zero official Triton MPS support | **BUILD ourselves**: general C++20 SSA + Fusion IR (`cpp/ir`), pure C++20 optimization/fusion passes (`cpp/passes`), MSL codegen (`cpp/codegen/metal`), disk-cached PSO-aware `@m1triton.autotune` & `@m1triton.heuristics`. |

### Relationship Between `torch.compile` and M1Triton
Just like **PyTorch Inductor + Triton** on CUDA, M1Triton operates cleanly at **two complementary layers**:
1. **Kernel-Author Layer (`@m1triton.jit` / `@m1triton.autotune` / `@m1triton.heuristics`)**: Developers write custom algorithmic GPU kernels (`tl.*` + `tlx.*`) such as L3 Attention/SSM/MoE/W4A16 kernels, optionally wrapped with `torch.library.custom_op` so `torch.compile` treats them as opaque fusible/autograd-aware nodes.
2. **Model-Compiler Layer (`torch.compile(model, backend="m1triton")`)**: Standard PyTorch `nn.Module` code (zero manual kernel rewriting) is traced by TorchDynamo/AOTAutograd into an ATen FX graph, partitioned by M1Triton's FX-to-M1IR graph fusion compiler into fused register-epilogue / SRAM-tiled Metal kernels, and dispatched via batched command buffers / `MetalGraph` (with safe eager fallback only in general user `torch.compile` mode for unsupported non-tensor/CPU control-flow ops — **never** inside frozen KernelBench L1/L2/L3 evals!).

---

## 3. Core Technical Upgrades (v2.0 Architecture)

All v1.0 mechanical boundaries (`ARCH-IR-PURE`, `ARCH-PASSES-NO-RUNTIME`, `ARCH-FRONTEND-NO-TORCH-CHEAT`, `FROZEN-EVAL`, `COMMENT-HYGIENE`) remain enforced by `node tools/check-guardrails.mjs` and extend to `benchmarks/kernels_l2/` and `benchmarks/triton_l3/`.

### 3.1 Multi-Level Fusion IR & Pure C++20 Passes (`cpp/ir/`, `cpp/passes/`)
Follow **Hardware-Target-First IR Design**: prototype and verify optimal target MSL schedules and `src -> dst` golden IR test cases (`tests/cpp/fusion/`) before generalizing each pass.
- **IR Extensions (`cpp/ir/`)**:
  - `EpilogueDAGAttr` / composable post-op regions attached to `Dot`, `ImplicitGemmConv`, and Winograd transforms.
  - `FusedTailAttr` for producer-consumer local pooling and spatial/channel reductions.
  - General Triton IR ops needed by L3 complex kernels: `AtomicRMW` (`add`, `max`, `min`, `xchg`, `cas`), `Gather`/`IndirectLoad` (for PagedAttention & MoE pointer tables), bitwise unpack ops (`Shr`, `And`, `BitCast` for W4A16), `StaticRange`/unrolled multi-`Dot` loop regions, `PhiloxRand` / counter-based RNG, multi-output kernel returns.
  - Full `IRVerifier` + `IRPrinter`/`IRParser` roundtrip support.
- **New Pure C++20 Passes (`cpp/passes/`)**:
  1. `AlgebraicNormFoldingPass`: Exact inference folding of `BatchNorm` (running mean/var + affine) & constant `Scale`/`Div`/`Mul` into producer weights/epilogues (dynamically synced with PyTorch weights, zero stale caches).
  2. `EpilogueDAGFusionPass`: Fuses arbitrary `Bias + Residual + Scale/Div/Sub/Add + Activation` chains directly into SGMM, ConvStd, ConvTranspose, Depthwise, and Winograd register-tile store epilogues.
  3. `ProducerConsumerSRAMFusionPass`: Fuses `Conv/Matmul -> Pool` and `Conv/Matmul -> Reduce (Sum/Mean/Max/Min/LogSumExp/Softmax)` inside 32 KB threadgroup SRAM / SIMD-group registers without materializing full intermediate tensors to DRAM.
  4. `ElementwiseReductionDAGFusionPass`: Fuses multi-op elementwise + broadcast + residual + reduction DAGs into single `float4`/`packed_float4` passes with strict IEEE-754 NaN propagation.
  5. `FusionOccupancyCostModelPass`: Evaluates M1 Max 32 KB SRAM, register pressure, PSO threadgroup limits, and arithmetic intensity to pick single-kernel register fusion, SRAM tile fusion, or batched 2-stage split.

### 3.2 Production `@m1triton.autotune` & `@m1triton.heuristics` (`python/m1triton/autotune.py`)
Upgrade v1.0's in-memory `Autotuner` into a full Triton-compatible autotuning engine:
- **Triton API Parity**: `@m1triton.autotune(configs=[Config(...)], key=[...], prune_configs_by=..., reset_to_zero=..., restore_value=..., warmup=..., rep=...)` + `@m1triton.heuristics(values={...})`, mapping `num_warps` cleanly to Apple M1 Max `num_simdgroups` (`32` threads/SG).
- **Roofline + Live PSO Pre-Pruning**: Filter out configs violating 32 KB threadgroup SRAM or compiled `[pso maxTotalThreadsPerThreadgroup]` before benchmarking (zero crashes, zero silent drops).
- **Persistent Disk Cache**: Store winning configs under `~/.cache/m1triton/autotune/` (configurable via `M1TRITON_CACHE_DIR`) keyed by `(kernel_ir_sha256, gpu_name, key_values, dtypes)` so cold restarts skip retuning.
- **Accurate GPU Timing**: Measure candidate configs via native `MTLCommandBuffer` `GPUEndTime - GPUStartTime` timestamps with median filtering and deterministic buffer reset (`reset_to_zero` / `restore_value`).

### 3.3 Metal Runtime Execution Graphs & Stream Interop (`cpp/runtime/metal_runtime.mm`)
- **`command_batch()`**: Encodes multi-kernel sequences into a single `MTLCommandBuffer` (zero intermediate host-GPU sync bubbles).
- **`MetalGraph` (Compute ICB Replay)**: Captures static multi-dispatch kernel graphs into an Apple M1 Max `MTLIndirectCommandBuffer` (`MTLIndirectCommandTypeConcurrentDispatch`) and replays them with single-call overhead; expose `m1triton.runtime.MetalGraph` with capture/replay API and benchmark honestly against `command_batch()`.
- **Safe MPS Stream Interop**: Clean synchronization with `torch.mps.current_stream()` (investigating `MTLSharedEvent` / native stream sync to minimize cross-queue latency while guaranteeing zero data races).

### 3.4 `torch.compile(backend="m1triton")` (`python/m1triton/compile_backend.py`)
- Register `m1triton` backend with `torch._dynamo.register_backend` + AOTAutograd decomposition.
- Implement **FX Graph -> M1Triton Fusion Plan Partitioner**: matches fusible chains (`Conv/Linear/Matmul + BN/Bias/Act/Residual/Pool/Reduce`, pointwise+reduction subgraphs), lowers matched subgraphs through M1Triton's C++ Fusion IR -> Metal kernels, and supports both **strict zero-fallback mode** (`M1TRITON_STRICT_COMPILE=1`, used for KernelBench L2 verification) and **general user mode** (clean graph partitioning around unsupported host/sparse ops).
- Register custom L3 kernels via `torch.library.custom_op` + `register_autograd` so they compose seamlessly inside `torch.compile`.

### 3.5 End-User Packaging, Installation & CLI Tools (First-Class Human UX)
Ensure any external user on Apple Silicon macOS can install, verify, and run M1Triton in minutes with zero `PYTHONPATH` workarounds:
- **PEP 517 Packaging ([pyproject.toml](pyproject.toml))**: Switch build backend to `scikit-build-core` driving `CMakeLists.txt` + Ninja + `/usr/bin/clang++` (`Metal.framework`), supporting:
  ```bash
  git clone <repo> && cd ai-compiler
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e ".[dev]"   # or: uv pip install -e ".[dev]"
  ```
- **Environment & Health CLI (`m1triton doctor` / `python -m m1triton doctor`)**:
  - Checks macOS version, Apple Silicon GPU name & Metal 3 / Compute ICB / 32 KB SRAM capabilities, `/usr/bin/clang++` + `Metal.framework`, PyTorch MPS availability, `MTLBinaryArchive` & autotune disk cache directories (`M1TRITON_CACHE_DIR`), and runs a 2-second GPU vector-add + SGMM correctness/bandwidth check.
- **Compiler Developer CLI (`m1triton-opt`)**:
  - Installed alongside the package; upgraded with `--dump-ir-before-all`, `--dump-ir-after-all`, `--pass=<name>`, `--stats` (SRAM bytes, fused ops, occupancy), and `src -> dst` golden transform verification.
- **Runnable Examples (`examples/`)**:
  - `01_vector_add.py` (standard `@triton.jit` drop-in compatibility)
  - `02_autotuned_gemm.py` (`@m1triton.autotune` + disk cache inspection)
  - `03_torch_compile_fusion.py` (`torch.compile(model, backend="m1triton")` vs eager vs Inductor)
  - `04_flash_attention_fwd_bwd.py` (L3 FlashAttention-2 forward + backward with `torch.autograd`)
  - `05_w4a16_quant_gemm.py` (L3 4-bit quantized inference GEMM)
- **Clean-Venv Verification Gate**: CI / release checks verify fresh-venv `pip install --no-build-isolation -e .` (or `pip install -e .`) -> `m1triton doctor` -> running all `examples/` without setting `PYTHONPATH` manually.

---

## 4. Benchmark Suites & Frozen Anti-Cheat Evaluation

### 4.1 KernelBench Level 2 (100 Multi-Op Fusion Tasks)
- **Source**: Fetch all 100 tasks of `ScalingIntelligence/KernelBench` `KernelBench/level2/` into `benchmarks/kernelbench_l2/` with pinned upstream SHA in `benchmarks/kernelbench_l2/manifest.json`.
- **Candidates**: `benchmarks/kernels_l2/task_001.py .. task_100.py` (compiled via M1Triton's Fusion IR / `torch.compile(backend="m1triton", strict=True)` / composable `tlx` fusion combinators; zero PyTorch/MPSGraph/Inductor compute fallbacks!).
- **Frozen Evaluator (`evals/eval_kernelbench_l2.py`, SHA-256 locked)**:
  - Same 3 anti-cheat layers as L1 + fusion telemetry (`dispatches_per_forward`, `cold_compile_ms`, `baseline_eager_ms`, optional `baseline_inductor_ms`, `candidate_ms`, `speedup`, `max_abs_err`) + built-in process-isolated chunk mode to avoid macOS swap bloat on multi-GB tensors.
- **Targets**: **100/100 passing**, **geomean speedup `>= 2.0x`** vs PyTorch MPS eager, **`>= 85/100` strictly faster (`> 1.00x`)**, `>= 92/100 >= 0.98x`.

### 4.2 Level 3 Complex Triton Kernel Suite (10 Real-World Algorithms)
- **Location**: `benchmarks/triton_l3/` (each kernel written cleanly in Triton-compatible `@m1triton.jit` + `@m1triton.autotune`, verified against PyTorch reference including backward `torch.autograd.grad` where applicable!).
- **Frozen Evaluator (`evals/eval_triton_l3.py`, SHA-256 locked)**: Correctness (fwd + bwd gradients across 3 seeds & edge shapes), strict zero-fallback anti-cheat, memory footprint verification (e.g., no `N×N` attention or `[BT, V]` logit materialization), and speed vs PyTorch MPS eager + `torch.compile(inductor)`.
- **The 10 L3 Kernels**:
  1. **`L3-01` FlashAttention-2 Forward (Causal + Non-Causal + GQA, `D=64/128`)** — Multi-block online softmax + `tl.dot` inner loop + `@m1triton.autotune`.
  2. **`L3-02` FlashAttention-2 Backward (`dQ, dK, dV` with SRAM Recomputation)** — `delta` pre-pass + recomputed `P` in registers/SRAM + full gradient check vs `torch.autograd.grad`.
  3. **`L3-03` PagedAttention Decode (vLLM Block-Table KV Cache + Split-KV)** — Indirect `block_tables` gather, ragged `context_lens`, 2-stage split-KV online merge.
  4. **`L3-04` Sliding-Window + ALiBi + Logit-Softcap Attention (Mistral/Gemma-2 style)** — Dynamic banded K-loop bounds, per-head ALiBi slopes, `tanh` softcap inside inner loop.
  5. **`L3-05` Fused RMSNorm + RoPE + In-Place KV-Cache Scatter** — Single-kernel Llama/Qwen transformer block prelude.
  6. **`L3-06` Fused LayerNorm Forward + Backward (`dx, dgamma, dbeta`)** — Triton tutorial `05-layer-norm` style with atomic/staged parameter gradient reduction verified vs `torch.autograd.grad`.
  7. **`L3-07` Chunked Fused Linear + Cross-Entropy Forward & Backward (Liger / Cut-CE style)** — Huge vocab (`V=32768..128256`) without materializing `[B*T, V]` logits in UMA DRAM.
  8. **`L3-08` Mamba / Selective State-Space Scan (SSM Forward)** — Chunked parallel associative scan over `(B, L, D, N_state)` in threadgroup SRAM + SIMD registers.
  9. **`L3-09` MoE Grouped GEMM + Persistent Tile Scheduler** — Ragged expert shapes (`08-grouped-gemm` / `09-persistent-matmul` style) dynamically scheduled across 32 Apple GPU cores + `@m1triton.autotune`.
  10. **`L3-10` W4A16 Group-Quantized Dequant-GEMM (`uint32` packed `int4` + scales/zeros)** — Bitwise on-the-fly register dequantization feeding `simdgroup_matrix_multiply_accumulate`.
- **Targets**: **10/10 L3 kernels passing strict fwd (+ bwd) correctness**, autotuned with persistent disk cache, beating PyTorch MPS eager (and proving real memory savings on FlashAttn/Chunked-CE/W4A16).

---

## 5. Milestone Roadmap (`V2-M0` -> `V2-M7`)

Track every milestone in `.audit/progress_v2.json` via `/cstack:mission-control`. Run mandatory unslop + docs + `slop-critic` (`pro`) hygiene checkpoints every 2 milestones.

- **`V2-M0`: Packaging, `m1triton doctor`, Runtime Graphs, Autotuner v2 & Frozen L2/L3 Harnesses**:
  - Migrate `pyproject.toml` to `scikit-build-core` (`pip install -e .` works cleanly in fresh venv without `PYTHONPATH`), implement `m1triton doctor`, upgrade `m1triton-opt` (`--dump-ir-before-all`, `--stats`).
  - Upgrade `python/m1triton/autotune.py` (persistent disk cache, live PSO pruning, `@m1triton.heuristics`, GPU-timestamp timing) + C++ runtime `command_batch()` & `MetalGraph` (Compute ICB prototype + latency benchmark).
  - Fetch all 100 KB L2 tasks (`benchmarks/kernelbench_l2/`), classify fusion families, build & SHA-256 freeze `evals/eval_kernelbench_l2.py` and `evals/eval_triton_l3.py`, capture L2 MPS eager (+ Inductor spot) baseline snapshot (`.audit/baseline_l2_mps.json`).

- **`V2-M1`: Epilogue DAG IR, `EpilogueDAGFusionPass`, `AlgebraicNormFoldingPass` & L2 GEMM Fusion (~25–30 tasks)**:
  - Implement `EpilogueDAGAttr`, `EpilogueDAGFusionPass`, `AlgebraicNormFoldingPass` + `src -> dst` golden IR tests + fused register epilogues in `sgmm_emitter.cpp`.
  - Pass 100% of KB L2 GEMM/Linear fusion cluster (`dispatches_per_forward == 1` on pure epilogue chains).

- **`V2-M2`: L2 Conv2D/3D & ConvTranspose Register Epilogue + BN-Fold Fusion (~45–50 tasks)**:
  - Extend `EpilogueDAGFusionPass` to `conv_std_emitter.cpp` (Implicit-GEMM + Winograd `F(2x2)`/`F(4x4)`), `conv_dw_emitter.cpp`, and `conv_transpose_emitter.cpp` (`HALO=1` & `FUSE_SW=1`).
  - Pass 100% of KB L2 Conv + ConvTranspose epilogue clusters.
  - **Hygiene Checkpoint (`V2-M1` + `V2-M2`)**: unslop + docs + `slop-critic` (`pro`).

- **`V2-M3`: Producer-Consumer SRAM Fusion (`Pool`/`Reduce`/`Norm`) + `torch.compile(backend="m1triton")` -> 100/100 KB L2 Coverage**:
  - Implement `ProducerConsumerSRAMFusionPass` & `ElementwiseReductionDAGFusionPass` for `Conv/Matmul -> Pool / Reduce / Softmax / LogSumExp / Norm`.
  - Build `python/m1triton/compile_backend.py` (`torch.compile(model, backend="m1triton")`) with FX-to-M1IR fusion partitioner and verify end-to-end compilation of standard PyTorch `nn.Module`s (including L2 tasks & multi-layer ResNet/MLP blocks!).
  - Gate: **100/100 KB L2 passing strict frozen correctness** + 100% v1.0 L1 CI non-regression.

- **`V2-M4`: General Triton IR/Codegen Upgrade & L3 Kernels Part 1 (`L3-01`..`L3-05`: Attention & LLM Core)**:
  - Extend pure C++20 IR & generic/specialized Metal emitters for multi-`tl.dot` loops, indirect gather/paged loads, dynamic loop pruning, and multi-output backward passes.
  - Implement, autotune, and verify **`L3-01` FlashAttention-2 Fwd (GQA)**, **`L3-02` FlashAttention-2 Bwd (`dQ/dK/dV` + `torch.autograd` custom op)**, **`L3-03` PagedAttention Decode (Split-KV)**, **`L3-04` Sliding-Window + ALiBi + Softcap Attention**, and **`L3-05` Fused RMSNorm + RoPE + KV-Cache Scatter**.
  - **Hygiene Checkpoint (`V2-M3` + `V2-M4`)**: unslop + docs + `slop-critic` (`pro`).

- **`V2-M5`: L3 Kernels Part 2 (`L3-06`..`L3-10`: Backward Atomics, Huge-Vocab CE, SSM, MoE & W4A16)**:
  - Add `AtomicRMW` & bitwise dequant lowering in C++ IR / Metal codegen.
  - Implement, autotune, and verify **`L3-06` Fused LayerNorm Fwd+Bwd**, **`L3-07` Chunked Linear Cross-Entropy Fwd+Bwd** (zero `[BT, V]` materialization), **`L3-08` Mamba SSM Selective Scan**, **`L3-09` Persistent MoE Grouped GEMM**, and **`L3-10` W4A16 Group-Quantized Dequant-GEMM**.
  - Gate: **10/10 L3 complex Triton kernels passing strict fwd (+ bwd) frozen eval**.

- **`V2-M6`: Fusion Cost Model, `MetalGraph`/ICB Tuning & `/cstack:research-lab` Performance Sprint**:
  - Implement `FusionOccupancyCostModelPass`; run disciplined `/cstack:research-lab` optimization loops (logged in `.audit/experiments_v2.tsv`, 2-strike circuit breaker) across slowest L2 chains and L3 kernels (comparing `command_batch()` vs Compute ICB `MetalGraph`, register tiling, SRAM swizzle, autotune sweeps).
  - Target: **KB L2 geomean `>= 2.0x` (`>= 85/100` beats baseline)** + strong speedups across all 10 L3 kernels vs PyTorch MPS eager & Inductor.

- **`V2-M7`: Full-Repo Unslop, Fresh-Venv User Verification, Final `slop-critic` Audit & `v2.0` Release Archive**:
  - Full-repo `/cstack:unslop` + adversarial `slop-critic` (`pro`) audit (NaNs in fused epilogues, backward gradient accuracy, `.data` mutation staleness, PSO limits, ICB safety, zero anti-cheat violations).
  - Official unmodified frozen runs: **100/100 KB L2** + **10/10 L3** + **v1.0 L1 CI regression** (`bash scripts/ci_local.sh` green).
  - Clean fresh-venv user install test (`pip install -e .` -> `m1triton doctor` -> all `examples/` green).
  - Update [README.md](README.md) (complete installation, `m1triton doctor`, `torch.compile`, `@m1triton.autotune`, `examples/`, L1+L2+L3 verified tables), `docs/FUSION_ARCHITECTURE.md`, `docs/BENCHMARKS_L2.md`, `docs/BENCHMARKS_L3.md`, and archive annotated tag `v2.0-kernelbench-l2-l3`.

---

## 6. Execution Discipline & Model Policy

- **Hardware-First Design**: Prototype target fused MSL + `src -> dst` golden IR transforms before generalizing C++ passes.
- **Multi-Agent Orchestration**: Use `swarm` (`workspace: "share"`) for disjoint fusion/kernel clusters and design races (recording tradeoffs in `.audit/decisions/`), `run_workflow` for batched verification, and `slop-critic` (`pro`) every 2 milestones.
- **Model Quota Discipline**: Default strictly to **Gemini** (`gemini-4-neon-xhigh` / `pro` / `flash`) across workers, architects, and critics; avoid Claude models unless explicitly requested to preserve 5h/7d Claude quota.
- **Autonomous Watchdogs & Memory Isolation**: Attach `schedule` watchdogs (`TimerCondition: "any"`) to every long build, swarm, or benchmark; isolate multi-GB evaluation tasks per process to prevent macOS swap thrashing.
