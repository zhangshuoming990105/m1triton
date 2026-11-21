# M1Triton Pass Pipeline

All passes in `cpp/passes/` are pure C++20 IR-to-IR transforms parameterized only by `m1triton::ir::TargetSpec` (`ARCH-PASSES-NO-RUNTIME`). `PassManager::run` executes `IRVerifier::verify` after every pass when `verify_each_pass` is enabled.

## Active M1 Pipeline (`PassManager::build_m1_pipeline()`)

1. **`CanonicalizePass`** (`cpp/passes/canonicalize.cpp`): Scalar constant folding, algebraic identities (`x+0`, `x*1`, `x*0`, `x-0`, `select(true/false)`), and nested `PtrAdd` folding.
2. **`CSEPass`** (`cpp/passes/cse.cpp`): Scoped value numbering over pure ops (`Constant`, `ProgramId`, `Arange`, `Splat`, commutative `Binary`, `Unary`, `PtrAdd`, `Select`). Never eliminates across memory ops or barriers.
3. **`LICMPass`** (`cpp/passes/licm.cpp`): Fixed-point hoisting of loop-invariant pure ops out of `ForLoop` regions into the parent block.
4. **`ContiguityAndAlignmentAnalysis`** (`cpp/passes/contiguity_analysis.cpp`): Propagates modular index facts (`step`, `contiguity`, `alignment`, `uniform`) across `Arange`, `ProgramId`, arithmetic, `PtrAdd`, `ForLoop` iter args, and pointer divisibility. Upgrades eligible `Load`/`Store` ops to `vec_width=4` (`16 B` alignment for `f32`, `8 B` for `f16`).
5. **`TileAndLayoutAssignmentPass`** (`cpp/passes/tile_layout.cpp`): Assigns `LayoutAttr` (`Blocked`, `tg_threads`, `elems_per_thread`, `vec_width`) against `TargetSpec` limits (`simd_width=32`, `max_threads_per_threadgroup=1024`, `max_threadgroup_memory_bytes=32768`).
6. **`DCEPass`** (`cpp/passes/dce.cpp`): Backward liveness removing dead pure ops and side-effect-free dead control-flow regions.

## Planned Milestone Passes (M2–M6)

- **M2**: `StructureAwareDeadTilePruningPass`, `ThreadgroupStagingAndSwizzlePass`, `SimdgroupMatrixLoweringPass`, Split-K & GEMV lowering.
- **M3–M4**: `SimdReductionAndScanLoweringPass` (online softmax, Welford norm, hardware `simd_sum`/`simd_max`/`simd_prefix_inclusive_sum`), `EpilogueFusionPass`.
- **M5–M6**: `ConvToImplicitGemmPass`, FlashAttention-M1 32 KB SRAM tile lowering.
