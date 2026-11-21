# M3 (#19-#40, #88) & M4 (#41-#49, #51-#53, #89-#93) Plan Summary

- **M3 Activations (`#19-#22, #25-#32, #88`)**: Single-pass fused `float4` kernels + NaN-safe `m1_tanh`, minimax `m1_erf`, threshold-20 `softplus`. Target `#25 Swish` `155ms -> ~35ms`, `#30 Softsign` `172ms -> ~35ms`, `#88 MinGPTNewGelu` `16.2ms -> ~1.6ms`.
- **M3 Online Softmax / LogSoftmax (`#23, #24`)**: 1 TG/row (`256` threads) two-pass streaming with Milakov-Gimelshein `(m, d)` `simd_shuffle_xor` monoid (`98.8ms -> ~55ms`).
- **M3 Norms (`#33-#40`)**: Two-stage `float4` Welford for `#33 BatchNorm` (train mode), `#34 InstanceNorm`, `#35 GroupNorm` (`370.5ms -> ~62ms`), `#40 LayerNorm` (`5.7ms -> ~2.8ms`); SRAM-cached `[64, 64]` single-read pass for `#36 RMSNorm` (`220.9ms -> ~41ms`); aligned-peel `float4` for odd-width `#38/#39 L1/L2` (`116.3ms -> ~69ms`); two-stage `float4` `#37 FrobeniusNorm`.
- **M4 Pooling (`#41-#46`)**: Unified 1D/2D/3D Max & Avg Pool (`PoolAttr`), SRAM halo tiling for 1D/2D (`#41 143.8ms -> ~7ms`, `#45 186.3ms -> ~18ms`), native Metal for MPS-unsupported 3D (`#43 1138.9ms -> ~12ms`, `#46 563.4ms -> ~32ms`).
- **M4 Strided Axis Reductions (`#47-#49, #51-#53`)**: Invert thread mapping along contiguous `dim2` (`64`/`128` threads) looping `dim1=0..4095` in registers with `I64` (`long`) outputs for `argmax`/`argmin` (`241-273ms -> ~26ms`).
- **M4 Hardware Prefix Scans (`#89-#93`)**: Single-pass 32-chunk×1024-float `float4` row scan using `simd_prefix_inclusive_sum` / `simd_prefix_exclusive_sum` + 5-step `simd_shuffle_up` for `cumprod` (`37-88ms -> ~25ms`).
