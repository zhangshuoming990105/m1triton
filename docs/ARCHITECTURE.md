# M1Triton Architecture

M1Triton is an Apple Silicon (M1 Max) specialized Triton-compatible compiler, autotuner, Metal execution graph runtime, and `torch.compile` fusion backend with strict separation between the thin Python frontend, pure C++20 SSA + Fusion IR, pure C++20 optimization passes, Metal Shading Language (MSL) code generator, and Objective-C++ Unified Memory (UMA) Metal runtime.

## Layer Diagram

```text
python/m1triton/          Thin Python frontend (@jit, @autotune, @heuristics, tl.*, tlx.*,
        |                 doctor, torch.compile(backend="m1triton") FX fusion partitioner)
        v  (pybind11 _C IRBuilder)
cpp/ir/                   Pure C++20 Typed Block SSA + Fusion IR (EpilogueDAGAttr, FusedTailAttr),
        |                 TargetSpec, Verifier, Printer/Parser
        v  (IR-to-IR transforms, TargetSpec only)
cpp/passes/               18 passes: Canonicalize, CSE, LICM, EpilogueFusion, EpilogueDAGFusion,
        |                 AlgebraicNormFolding, StructurePruning, OnlineSoftmaxWelford, PoolLowering,
        |                 FusionOccupancyCostModel, ConvLowering, LossLowering, FlashAttentionLowering,
        |                 Contiguity, TileLayout, ThreadgroupStaging, SimdgroupMatrixLowering, DCE
        v
cpp/codegen/metal/        Shared msl_common (IEEE-754 NaN-safe helpers + emit_epilogue_dag_msl) +
        |                 11 specialized MSL emitters + generic metal_codegen + LaunchSpec
        v
cpp/runtime/              Objective-C++ (.mm) Metal UMA runtime: zero-copy MPS/CPU buffers,
                          MTLDevice::newLibraryWithSource + MTLBinaryArchive disk cache,
                          concurrent command_batch() with automatic RAW/WAW/WAR barriers,
                          MetalGraph Compute ICB capture/replay, PSO threadgroup-limit guard,
                          GPU timestamps, compiled dispatch counter
```

## Mechanically Enforced Architecture Boundaries

All boundaries are checked on every CI run by `node tools/check-guardrails.mjs` against `guardrails.json`:

1. `ARCH-IR-PURE`: `cpp/ir/` forbids includes from `cpp/runtime`, `cpp/codegen`, `cpp/bindings`, `Metal`, `Foundation`, or `pybind11`.
2. `ARCH-PASSES-NO-RUNTIME`: `cpp/passes/` forbids includes from `cpp/runtime`, `cpp/bindings`, `Metal`, `Foundation`, or `pybind11`. Hardware parameters flow exclusively through `m1triton::ir::TargetSpec`.
3. `ARCH-FRONTEND-NO-TORCH-CHEAT`: `benchmarks/kernels/`, `benchmarks/kernels_l2/`, and `benchmarks/triton_l3/kernels/` forbid PyTorch/MPSGraph/Inductor/SciPy compute fallbacks. Reinforced dynamically in frozen evaluators via static AST inspection, `TorchDispatchMode` ATen compute tripwires, and native Metal `get_dispatch_count() > 0`.
4. `FROZEN-EVAL`: All 4 evaluators (`evals/eval_kernelbench.py`, `evals/roofline_probe.py`, `evals/eval_kernelbench_l2.py`, `evals/eval_triton_l3.py`) are SHA-256 locked in `guardrails.json`.
5. `COMMENT-HYGIENE`: Rejects narration comments, unstructured hack comments, and unapproved lint/type suppressions.

## Apple M1 Max Measured Hardware Envelope (`[measured]`)

Measured via frozen `evals/roofline_probe.py` on this MacBook Pro (`Apple M1 Max`, 32 GPU cores, 32 GB UMA):

| Metric | Theoretical | Measured (`roofline_probe.py`) |
| --- | --- | --- |
| Memory Bandwidth (`stream_copy`, 512 MB) | 400.0 GB/s | 373.83 GB/s (IQR 0.011 ms) |
| Memory Bandwidth (`stream_triad`, 805 MB) | 400.0 GB/s | 368.86 GB/s (IQR 0.048 ms) |
| FP32 SGMM (`simdgroup_float8x8`) | 10.4 TFLOPS | 10.171 TFLOPS |
| FP16 SGMM (`simdgroup_half8x8`) | 10.4 TFLOPS (Apple7 SGMM) | 9.634 TFLOPS |
| Threadgroup SRAM | 32,768 bytes | 32,768 bytes |
| SIMD-group Width | 32 threads | 32 threads |
| Max Threads per Threadgroup | 1,024 | 1,024 |

## Core Compiler Invariants

- **Zero String-Template Cheating**: Every `@triton.jit` / `@m1triton.jit` kernel lowers Python AST into C++ `m1triton::ir::Kernel` SSA blocks, passes `IRVerifier` before and after every optimization pass, and emits MSL through `cpp/codegen/metal/`.
- **Zero-Copy UMA Tensor Bridging**: PyTorch MPS tensors pass `tensor.untyped_storage().data_ptr()` as `id<MTLBuffer>` and `tensor.storage_offset() * tensor.element_size()` as the Metal buffer byte offset (with strided span tracking for hazard detection) with zero extra host-device copies.
- **Native Runtime Compilation**: Uses Apple Clang `/usr/bin/clang++` with `Metal.framework` `MTLDevice::newLibraryWithSource` backed by `MTLBinaryArchive` disk serialization (no `xcrun metal` CLI dependency).
