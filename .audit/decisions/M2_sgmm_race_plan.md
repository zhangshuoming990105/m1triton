# M2 (#1-#18) SGMM & Structure-Aware Racing Plan

Recorded before M2 execution so every competing worktree implementation, measured latency, SRAM/occupancy trade-off, and fallback retention choice is tracked on disk.

## Target Roofline Ceilings (`[measured]`, `.audit/roofline_m1max.json`)
- STREAM Copy: `373.83 GB/s`
- FP32 SGMM (`simdgroup_float8x8`): `10.171 TFLOPS`

## Parallel Worktree Racers Planned (Post-M1 Merge)
1. `race/m2-sgmm-64x64x16`: 4 SIMD-groups (128 threads, 2x2 SG grid, 4x4 frags/SG), double-buffered SRAM (`PAD=4`, 18.5 KB), `float4` coalesced staging.
2. `race/m2-sgmm-large-tile`: `128x64x16` (8 SIMD-groups, 2x B reuse, 14.25 KB 1-buf / 28.5 KB 2-buf) + `64x64x32` (4 SIMD-groups, halved barrier rate, 17.5 KB).
3. `race/m2-sgmm-swizzle-transpose`: 2D SLC super-tile swizzle (`w=8`), register prefetch (1x SRAM latency hiding), zero-copy transposed-A/B SRAM staging (`#16-#18`).
4. `feat/m2-structure-splitk-gemv`: `StructureAwareDeadTilePruningPass` (`#12` diag->mul, `#14/#15` triu/tril 6x FLOP cut + K-loop tightening, `#13` strict non-symmetric-product invariant), automatic/hinted `split_k` (`#6`), and 4-row `float4` SIMD-reduced GEMV (`#4`).

## Preservation Policy
- Keep winning SGMM schedules in `TileAndLayoutAssignmentPass` (shape-keyed / autotuner selectable).
- Preserve losing racer branches/benchmarks in `.audit/decisions/M2_sgmm_race_results.md` and `.audit/experiments.tsv` so alternative schedules can be recalled for specific shapes during M7 hillclimbing.
