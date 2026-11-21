# M1Triton Pass Pipeline

All passes in `cpp/passes/` are pure C++20 IR-to-IR transforms parameterized only by `m1triton::ir::TargetSpec` (`ARCH-PASSES-NO-RUNTIME`). `PassManager::run` verifies SSA invariants via `IRVerifier::verify` after every pass when `verify_each_pass` is enabled.

## Active Pipeline (`PassManager::build_m2_pipeline()`)

1. **`CanonicalizePass`** (`cpp/passes/canonicalize.cpp`): Constant folding, algebraic identities (`x+0`, `x*1`, `x*0`, `x-0`, `select(true/false)`), and nested `PtrAdd` folding with 64-bit promotion for mixed-signedness offsets.
2. **`CSEPass`** (`cpp/passes/cse.cpp`): Scoped value numbering over pure ops without eliminating across memory operations or barriers.
3. **`LICMPass`** (`cpp/passes/licm.cpp`): Fixed-point hoisting of loop-invariant pure ops out of `ForLoop` regions.
4. **`EpilogueFusionPass`** (`cpp/passes/epilogue_fusion.cpp`): Folds activation idioms (`pow(x, 2/3) -> mul`, `1/(1+exp(-x)) -> sigmoid`, `select(x > c, x, c) -> max`) and promotes aligned 1D unit-stride elementwise `Load`/`Store` ops to `vec_width=4` when base pointer divisibility and offsets prove 16-byte alignment.
5. **`StructureAwareDeadTilePruningPass`** (`cpp/passes/structure_pruning.cpp`): Propagates `StructureAttr` (`upper_triangular`, `lower_triangular`, `diagonal`, `symmetric`, `causal_mask`) to prune zero `k`-tiles and dead output tiles (preserving general `Symmetric(A) @ Symmetric(B)` non-symmetry).
6. **`OnlineSoftmaxWelfordPass`** (`cpp/passes/online_softmax_welford.cpp`): Configures Apple M1 Max threadgroup occupancy (`tg_threads=256`, `vec_width=4`) for `OnlineSoftmax` and `WelfordReduce` ops, enabling 1-read `[64, 64]` threadgroup SRAM caching (`16 KB <= 32 KB`) for `RMSNorm`.
7. **`PoolLoweringPass`** (`cpp/passes/pool_lowering.cpp`): Assigns 1D/2D/3D pooling threadgroup geometry and enables SRAM halo staging when the input halo tile fits within half of M1 Max threadgroup memory (`<= 16 KB`).
8. **`ConvLoweringPass`** (`cpp/passes/conv_lowering.cpp`): Validates 1D/2D/3D `ConvAttr` geometry, marks output pointers writable, and schedules standard (`64x64x16` / `128x64x32` 8-SG), 1x1 pointwise, phase-split transposed, and SRAM-halo depthwise convolution tiles within M1 Max's `32 KB` SRAM limit.
9. **`LossLoweringPass`** (`cpp/passes/loss_lowering.cpp`): Validates two-stage fused reduction loss operands/hyperparameters (`MSE`, `CrossEntropy`, `Huber`, `KLDiv`, `TripletMargin`, `Hinge`), marks output pointers writable, and assigns `tg_threads=256`.
10. **`FlashAttentionLoweringPass`** (`cpp/passes/flash_attn_lowering.cpp`): Validates `FlashAttentionTile` operands/scale and selects M1 Max `32 KB` single-pass (`16x512x16`, 16-SG) or online-softmax multi-tile (`16x256x16`, 16-SG) schedules.
11. **`ContiguityAndAlignmentAnalysis`** (`cpp/passes/contiguity_analysis.cpp`): Propagates modular index divisibility (`step`, `contiguity`, `alignment`, `uniform`) across `Arange`, `ProgramId`, `PtrAdd`, and `ForLoop` iter args.
12. **`TileAndLayoutAssignmentPass`** (`cpp/passes/tile_layout.cpp`): Assigns `Blocked` / `SimdgroupFragment` layouts (`128x64x32`, `64x128x32`, `64x64x32`, `swizzle_w`) against `TargetSpec` (`simd_width=32`, `32 KB` SRAM, `1024` threads/TG).
13. **`ThreadgroupStagingAndSwizzlePass`** (`cpp/passes/threadgroup_staging.cpp`): Plans bank-conflict-free threadgroup SRAM staging (`PadBank4`/`PadBank8`/`Xor8`) and downgrades staging stages when SRAM exceeds `32 KB`.
14. **`SimdgroupMatrixLoweringPass`** (`cpp/passes/simdgroup_matmul.cpp`): Validates and lowers `Dot` ops to Apple `simdgroup_float8x8` / `simdgroup_half8x8` fragment schedules.
15. **`DCEPass`** (`cpp/passes/dce.cpp`): Backward liveness removing dead pure ops and side-effect-free dead control-flow regions.

## CLI Inspection Utility (`m1triton-opt`)

Run passes standalone on textual SSA IR (`--pipeline=m1|m2`, `--pass=<name>`, `--dump-ir`, `--emit-msl`):

```bash
./build/m1triton-opt --pipeline=m2 --emit-msl input.m1ir
```
