# M1Triton Pass Pipeline

All 18 passes in `cpp/passes/` are pure C++20 IR-to-IR transforms parameterized only by `m1triton::ir::TargetSpec` (`ARCH-PASSES-NO-RUNTIME`). `PassManager::run` verifies SSA invariants via `IRVerifier::verify` after every pass when `verify_each_pass` is enabled.

## Active Pipeline (`PassManager::build_m1_pipeline()` / `build_m2_pipeline()`)

1. **`CanonicalizePass`** (`cpp/passes/canonicalize.cpp`): Constant folding, algebraic identities (`x+0`, `x*1`, `x*0`, `x-0`, `select(true/false)`), and nested `PtrAdd` folding with 64-bit promotion for mixed-signedness offsets.
2. **`CSEPass`** (`cpp/passes/cse.cpp`): Scoped value numbering over pure ops without eliminating across memory operations or barriers.
3. **`LICMPass`** (`cpp/passes/licm.cpp`): Fixed-point hoisting of loop-invariant pure ops out of `ForLoop` regions.
4. **`EpilogueFusionPass`** (`cpp/passes/epilogue_fusion.cpp`): Folds activation idioms (`pow(x, 2/3) -> mul`, `1/(1+exp(-x)) -> sigmoid`, `select(x > c, x, c) -> max`) and promotes aligned 1D unit-stride elementwise `Load`/`Store` ops to `vec_width=4` when base pointer divisibility and offsets prove 16-byte alignment.
5. **`EpilogueDAGFusionPass`** (`cpp/passes/epilogue_dag_fusion.cpp`): Propagates explicit `EpilogueDAGAttr` / `FusedTailAttr` hints onto producer ops (`Dot`, `ImplicitGemmConv`, `OnlineSoftmax`, `WelfordReduce`, `Pool`, `Reduce`) and automatically fuses straight-line post-`Dot` Triton SSA elementwise chains (`bias`, `residual`, constants, activations) into register-level `EpilogueDAGAttr` sequences.
6. **`AlgebraicNormFoldingPass`** (`cpp/passes/algebraic_norm_fold.cpp`): Simplifies consecutive constant scaling/bias/clamp epilogue steps (`MulConst(a) -> MulConst(b)`, `AddConst(a) -> AddConst(b)`, `DivConst(c) -> MulConst(1/c)`, identity step elimination) in pure C++20 IR.
7. **`StructureAwareDeadTilePruningPass`** (`cpp/passes/structure_pruning.cpp`): Propagates `StructureAttr` (`upper_triangular`, `lower_triangular`, `diagonal`, `symmetric`, `causal_mask`) to prune zero `k`-tiles and dead output tiles (preserving general `Symmetric(A) @ Symmetric(B)` non-symmetry).
8. **`OnlineSoftmaxWelfordPass`** (`cpp/passes/online_softmax_welford.cpp`): Configures Apple M1 Max threadgroup occupancy (`tg_threads=256`, `vec_width=4`) for `OnlineSoftmax` and `WelfordReduce` ops, enabling 1-read `[64, 64]` threadgroup SRAM caching (`16 KB <= 32 KB`) for `RMSNorm`.
9. **`PoolLoweringPass`** (`cpp/passes/pool_lowering.cpp`): Assigns 1D/2D/3D pooling threadgroup geometry and enables SRAM halo staging when the input halo tile fits within half of M1 Max threadgroup memory (`<= 16 KB`).
10. **`FusionOccupancyCostModelPass`** (`cpp/passes/fusion_cost_model.cpp`): Evaluates Apple M1 Max 32 KB SRAM, register pressure, PSO threadgroup limits (`<= 1024`), resident threadgroups/core, and arithmetic intensity across `Dot`, `ImplicitGemmConv`, `GroupedGemm`, `QuantDot`, `FlashAttentionTile`, and pointwise kernels; idempotently clamps infeasible `num_simdgroups`, disables double buffering, or reduces tile dimensions along SGMM-aligned boundaries before downstream lowering.
11. **`ConvLoweringPass`** (`cpp/passes/conv_lowering.cpp`): Validates 1D/2D/3D `ConvAttr` geometry, marks output pointers writable, and schedules standard (`64x64x16` / `128x64x32` pipelined SGMM), Winograd `F(2x2, 3x3)` & chunked `F(4x4, 3x3)`, 1x1 pointwise, active-phase `FUSE_SW` transposed, and barrier-free `packed_float4` register-sliding-window depthwise convolutions within M1 Max's `32 KB` SRAM limit.
12. **`LossLoweringPass`** (`cpp/passes/loss_lowering.cpp`): Validates two-stage fused reduction loss operands/hyperparameters (`MSE`, `CrossEntropy`, `Huber`, `KLDiv`, `TripletMargin`, `Hinge`), marks output pointers writable, and assigns `tg_threads=256`.
13. **`FlashAttentionLoweringPass`** (`cpp/passes/flash_attn_lowering.cpp`): Validates `FlashAttentionTile` operands/scale across Forward, Backward (`BwdDKDV` / `BwdDQ`), PagedAttention Decode, GQA, Sliding-Window, ALiBi, and Softcap modes, selecting M1 Max `32 KB` SRAM schedules.
14. **`ContiguityAndAlignmentAnalysis`** (`cpp/passes/contiguity_analysis.cpp`): Propagates modular index divisibility (`step`, `contiguity`, `alignment`, `uniform`) across `Arange`, `ProgramId`, `PtrAdd`, and `ForLoop` iter args.
15. **`TileAndLayoutAssignmentPass`** (`cpp/passes/tile_layout.cpp`): Assigns `Blocked` / `SimdgroupFragment` layouts (`128x64x32`, `64x128x32`, `64x64x32`, up to 32 SIMD-groups, `swizzle_w`) against `TargetSpec` (`simd_width=32`, `32 KB` SRAM, `1024` threads/TG).
16. **`ThreadgroupStagingAndSwizzlePass`** (`cpp/passes/threadgroup_staging.cpp`): Plans bank-conflict-free threadgroup SRAM staging (`PadBank4`/`PadBank8`/`Xor8`) and downgrades staging stages when SRAM exceeds `32 KB`.
17. **`SimdgroupMatrixLoweringPass`** (`cpp/passes/simdgroup_matmul.cpp`): Validates and lowers `Dot` ops to Apple `simdgroup_float8x8` / `simdgroup_half8x8` fragment schedules.
18. **`DCEPass`** (`cpp/passes/dce.cpp`): Backward liveness removing dead pure ops and side-effect-free dead control-flow regions.

## CLI Inspection Utility (`m1triton-opt`)

Run full pipelines or individual named passes standalone on textual SSA IR (`--passes=m1|m2|<comma-list>`, `--dump-ir-before-all`, `--dump-ir-after-all`, `--stats`, `--emit-msl`, `--no-verify`):

```bash
# Installed via pip install -e . (or ./build/m1triton-opt)
m1triton-opt --passes=m2 --dump-ir-after-all --stats --emit-msl input.m1ir

# Inspect FusionOccupancyCostModelPass metrics (sram_est, regs_est, occ_est, fused_ops, ai)
m1triton-opt --passes=epilogue-dag-fusion,algebraic-norm-fold,fusion-cost-model --stats input.m1ir
```
