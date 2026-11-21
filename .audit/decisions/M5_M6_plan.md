# M5 (#50, #54-#87) & M6 (#94-#100) Plan Summary

- **M5 Standard & 1x1 Convs (13 tasks)**: Zero-copy `NCHW` Implicit-GEMM with `M = C_out / G`, `N = Spatial_out`, `K = (C_in / G) * prod(K)` reusing `simdgroup_float8x8` (`float4` coalesced along `W_out`). `#87` pointwise `284ms -> ~38ms`, `#76` `526ms -> ~70ms`.
- **M5 Transposed Convs (17 tasks, incl. 7 CPU-fallback 3D tasks)**: Output-centric Implicit-GEMM with zero scatter atomics + phase-split on `stride > 1` (up to `prod(stride) = 8x` FLOP cut). `#68` `23990ms CPU -> ~240ms Metal (~100x)`.
- **M5 Depthwise & Separable 2D (`#82-#86`)**: Spatial SRAM halo microkernel (`float4` along `W`); `#86` chains depthwise + pointwise SGMM in one command buffer (`78.8ms -> ~14ms`).
- **M6 Fused Losses (`#94-#96, #98-#100`)**: Single-pass grid-stride `float4` block reductions + online log-sum-exp for `#95` (`int64` targets), raw fused `t*(log(t)-log(p))` for `#98` (zero `aten.log` in forward), 1D broadcast `targets[col]` for `#100`.
- **M6 FlashAttention-M1 (`#97`, `B=32,H=32,L=512,D=1024`, `1099.5 GFLOPs`, `108.1 ms` physical floor vs `258.75 ms` MPS)**: Race A (`BM=16, BN=64, BD=32`, 16 SGs = 512 threads, `O` in registers, `union` SRAM `21.8 KB <= 32 KB`) vs B (`BM=32, BN=32, BD=32`, `13.8 KB` SRAM), target `~120-135 ms (~2x)`.
