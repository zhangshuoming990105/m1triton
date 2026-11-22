# M1Triton

Apple Silicon (M1 Max) specialized Triton-compatible compiler, autotuner, Metal execution graph runtime, and `torch.compile` fusion backend (`Python AST / TorchDynamo FX -> C++20 Typed Block SSA + Fusion IR -> 18 Pure C++20 Passes -> MSL Codegen -> Objective-C++ Metal UMA Runtime`).

M1Triton operates at two complementary layers on top of PyTorch MPS's Unified Memory (`MTLBuffer`) allocator and autograd engine:
1. **Kernel-Author Layer**: Write custom Triton algorithms (`@m1triton.jit`, `@m1triton.autotune`, `@m1triton.heuristics`, `tl.*`, `tlx.*`) compiled directly to Metal Shading Language with `MTLBinaryArchive` + disk-cached autotuning.
2. **Model-Compiler Layer**: Compile standard `torch.nn.Module` models via `torch.compile(model, backend="m1triton")` without rewriting kernels, fusing ATen subgraphs into register-epilogue and 32 KB SRAM-tiled Metal kernels.

## Verified Official Frozen Benchmarks (`v2.0-kernelbench-l2-l3`, `[measured]`)

Evaluated end-to-end on **Apple M1 Max** (32 GPU cores, 32 GB UMA, 400 GB/s theoretical / 373.83 GB/s measured STREAM) against PyTorch 2.15.0 MPS eager using unmodified SHA-256-locked harnesses (`--seeds 3 --warmup 10 --trials 30`, zero PyTorch/MPSGraph/Inductor compute fallbacks, enforced via static AST guardrails, `TorchDispatchMode` ATen tripwires, and native Metal dispatch counters):

| Benchmark Suite | Harness (SHA-256 Locked) | Passed | Beats MPS (`>1.00x`) | Geomean Speedup | Scoreboard & Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| **KernelBench Level 2** (100 multi-op fusions) | [eval_kernelbench_l2.py](evals/eval_kernelbench_l2.py) | **100/100** | **100/100** | **3.0130x** | [docs/BENCHMARKS_L2.md](docs/BENCHMARKS_L2.md) ([JSON](.audit/eval_final_l2_100.json)) |
| **Level 3 Complex Triton Suite** (10 algorithms, fwd + bwd) | [eval_triton_l3.py](evals/eval_triton_l3.py) | **10/10** | **10/10** | **3.9235x** | [docs/BENCHMARKS_L3.md](docs/BENCHMARKS_L3.md) ([JSON](.audit/eval_final_l3_10.json)) |
| **KernelBench Level 1** (`v1.0-kernelbench-l1`, non-regression) | [eval_kernelbench.py](evals/eval_kernelbench.py) | **100/100** | **71/100** (`85 >= 0.98x`) | **1.9268x** | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) ([JSON](.audit/eval_final_100.json)) |

### KernelBench Level 2 Fusion Breakdown (`100/100`, `[measured]`)

Median dispatches per forward: **`2`** (`24/100` single-dispatch `dpf=1`; min speedup `1.02x`, max `421.37x`).

| Fusion Family | Tasks | Passed | Beats MPS | Median DPF | Geomean Speedup | Highlights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GEMM / Matmul / BMM Fusion | 37 | 37/37 | 37/37 | 2 | **1.69x** | `#37` **2.15x**, `#51` **1.94x**, `#66` **1.92x** |
| Conv2d Fusion | 21 | 21/21 | 21/21 | 3 | **2.09x** | `#82` **4.63x**, `#57` **4.45x**, `#21` **4.31x** |
| Conv3d Fusion | 12 | 12/12 | 12/12 | 3 | **3.98x** | `#27` **22.04x**, `#90` **6.47x**, `#48` **5.90x** |
| ConvTranspose2d Fusion | 11 | 11/11 | 11/11 | 2 | **3.28x** | `#42` **32.57x**, `#2` **5.39x**, `#16` **4.21x** |
| ConvTranspose3d Fusion | 19 | 19/19 | 19/19 | 2 | **11.13x** | `#77` **421.37x**, `#78` **20.39x**, `#34` **16.37x** |
| **Total** | **100** | **100/100** | **100/100** | **2** | **3.0130x** | Full table: [docs/BENCHMARKS_L2.md](docs/BENCHMARKS_L2.md) |

### Level 3 Complex Triton Kernel Suite (`10/10`, `[measured]`)

| ID | Kernel | Fwd+Bwd | Base MPS (ms) | M1Triton (ms) | Speedup | Key Mechanism |
| --- | --- | :---: | ---: | ---: | ---: | --- |
| `L3-01` | `flash_attn2_fwd_gqa_causal` | Fwd | 9.1243 | 2.8112 | **3.25x** | Causal GQA FlashAttention-2, zero `N×N` DRAM |
| `L3-02` | `flash_attn2_fwd_bwd_causal` | Yes | 1.9734 | 1.8797 | **1.05x** | SRAM recomputation + concurrent `command_batch()` (`dQ, dK, dV`) |
| `L3-03` | `paged_attn_decode_vllm` | Fwd | 23.4686 | 1.8899 | **12.42x** | Indirect block-table gather + 2-stage Split-KV merge |
| `L3-04` | `sliding_window_alibi_softcap_attn` | Fwd | 5.1920 | 0.7640 | **6.80x** | Banded window + ALiBi slopes + `tanh` logit softcap |
| `L3-05` | `fused_rmsnorm_rope_kv` | Fwd | 9.5170 | 1.2390 | **7.68x** | Single-kernel RMSNorm + RoPE + in-place KV scatter |
| `L3-06` | `fused_layernorm_fwd_bwd` | Yes | 1.9132 | 1.6259 | **1.18x** | Metal 3.0 `atomic_float` parameter gradient reduction |
| `L3-07` | `chunked_linear_cross_entropy` | Yes | 71.8625 | 57.3470 | **1.25x** | Vocab-chunked CE (`V=32768`), zero `[BT, V]` logit tensor |
| `L3-08` | `mamba_ssm_scan` | Fwd | 55.6889 | 0.8365 | **66.58x** | Parallel associative selective scan in SRAM + SIMD regs |
| `L3-09` | `moe_grouped_gemm` | Fwd | 2.2455 | 1.7187 | **1.31x** | Persistent tile scheduler over ragged MoE experts |
| `L3-10` | `w4a16_dequant_gemm` | Fwd | 4.6854 | 1.5357 | **3.05x** | Bitwise `int4` register dequant (`8x` vs FP32 weight footprint) |

## Installation & Getting Started

**Prerequisites**: Apple Silicon macOS with CommandLineTools (`/usr/bin/clang++` + `Metal.framework`, no Xcode/`xcrun metal` required), CMake, Ninja, Python 3.11+, `pybind11`, and PyTorch MPS (`torch`).

```bash
git clone https://github.com/zhangshuoming990105/m1triton.git && cd m1triton
python3 -m venv .venv && source .venv/bin/activate
pip install torch && pip install -e ".[dev]"   # builds _C extension + m1triton-opt CLI

# Verify GPU, Metal 3 / ICB, caches, and run built-in bandwidth & SGMM smoke checks
m1triton doctor
```

**Tutorials (Start Here)**:
1. [Installation, Verification & Troubleshooting](docs/tutorials/01_installation.md) — Clean virtualenv setup, `m1triton doctor`, caches, env vars, and FAQ
2. [Getting Started (`@jit`, `torch.compile`, `@autotune`)](docs/tutorials/02_getting_started.md) — Your first kernel, zero-rewrite `torch.compile`, and disk-cached autotuning
3. [Advanced Guide (`tlx`, Execution Graphs, Custom Autograd & `m1triton-opt`)](docs/tutorials/03_advanced.md) — Hardware-aware fusion, `command_batch()` vs. `MetalGraph` Compute ICB, backward kernels, and IR/MSL debugging

### Runnable Examples (Verified in Fresh Venv Without `PYTHONPATH`)

```bash
python examples/01_vector_add.py               # @triton.jit drop-in compatibility (321 GB/s)
python examples/02_autotuned_matmul.py         # @m1triton.autotune + persistent disk cache
python examples/03_torch_compile_fusion.py     # torch.compile(model, backend="m1triton") vs Eager & Inductor
python examples/04_flash_attention_fwd_bwd.py  # Differentiable FlashAttention-2 fwd+bwd (torch.autograd)
python examples/05_w4a16_quant_gemm.py         # W4A16 4-bit quantized GEMM (2.67x, 8x memory reduction)
```

### Quick Usage Snippets

```python
import torch, m1triton

# 1. Model-compiler mode: fuse standard PyTorch nn.Module graphs on MPS
compiled = torch.compile(model, backend="m1triton")
y = compiled(x)

# 2. Kernel-author mode: autotuned Metal kernels + execution graphs
@m1triton.autotune(
    configs=[m1triton.Config({"BLOCK": 256}, num_warps=4), m1triton.Config({"BLOCK": 512}, num_warps=8)],
    key=["n"],
)
@m1triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: m1triton.language.constexpr):
    offs = m1triton.language.program_id(0) * BLOCK + m1triton.language.arange(0, BLOCK)
    mask = offs < n
    m1triton.language.store(out_ptr + offs, m1triton.language.load(x_ptr + offs, mask=mask) + m1triton.language.load(y_ptr + offs, mask=mask), mask=mask)

# 3. Single-command-buffer concurrent batching & static Compute ICB replay
with m1triton.runtime.command_batch(wait=True, sync_mps=True):
    ...  # concurrent MTLDispatchTypeConcurrent encoder with automatic RAW/WAW/WAR barriers
```

## Architecture & Hardware Envelope

Measured via frozen [evals/roofline_probe.py](evals/roofline_probe.py) ([.audit/roofline_m1max.json](.audit/roofline_m1max.json)): **373.83 GB/s** STREAM copy (93.5% of 400 GB/s), **10.171 TFLOPS** FP32 SGMM (97.8% of 10.4 TFLOPS), **9.634 TFLOPS** FP16 SGMM, **32 KB** threadgroup SRAM, **32-thread** SIMD-group width.

```text
python/m1triton/          @jit, @autotune, @heuristics, tl.*, tlx.*, doctor,
        |                 torch.compile(backend="m1triton") FX fusion partitioner
        v  (pybind11 _C IRBuilder)
cpp/ir/                   Pure C++20 Typed Block SSA + Fusion IR (EpilogueDAGAttr, FusedTailAttr),
        |                 TargetSpec, Verifier, Printer/Parser
        v  (TargetSpec-only IR-to-IR transforms, zero Metal/runtime coupling)
cpp/passes/               18 passes: Canonicalize, CSE, LICM, EpilogueFusion, EpilogueDAGFusion,
        |                 AlgebraicNormFolding, StructurePruning, OnlineSoftmaxWelford, PoolLowering,
        |                 FusionOccupancyCostModel, ConvLowering, LossLowering, FlashAttentionLowering,
        |                 Contiguity, TileLayout, ThreadgroupStaging, SimdgroupMatrixLowering, DCE
        v
cpp/codegen/metal/        Shared msl_common (IEEE-754 NaN-safe helpers + emit_epilogue_dag_msl) +
        |                 11 specialized MSL emitters + generic metal_codegen
        v
cpp/runtime/              Objective-C++ (.mm) Metal UMA runtime: zero-copy MPS/CPU buffers,
                          MTLBinaryArchive disk cache, concurrent command_batch() with automatic
                          hazard barriers, MetalGraph Compute ICB replay, PSO limit validation
```

Architectural boundaries (`ARCH-IR-PURE`, `ARCH-PASSES-NO-RUNTIME`, `ARCH-FRONTEND-NO-TORCH-CHEAT`, `FROZEN-EVAL` across all 4 frozen evaluators, `COMMENT-HYGIENE`) are mechanically enforced on every CI run by [tools/check-guardrails.mjs](tools/check-guardrails.mjs).

## Developer & Verification Commands

```bash
# Full local CI parity (guardrails, feature map audit, C++ build + 17 CTests, 120 Pytests)
bash scripts/ci_local.sh

# Standalone C++ IR pass inspection, cost-model stats, and MSL codegen CLI
m1triton-opt --passes=m2 --dump-ir-after-all --stats --emit-msl path/to/kernel.m1ir

# Run SHA-256-frozen KernelBench L2 (100 tasks) or L3 (10 complex kernels)
PYTHONPATH=python:build python3 evals/eval_kernelbench_l2.py --tasks all --seeds 3 --warmup 10 --trials 30 --process-isolated --chunk-size 5 --roofline .audit/roofline_m1max.json --out .audit/eval_l2.json
PYTHONPATH=python:build python3 evals/eval_triton_l3.py --tasks all --seeds 3 --warmup 10 --trials 30 --out .audit/eval_l3.json

# Verify frozen evaluator SHA-256 locks and architectural guardrails
node tools/check-guardrails.mjs
```

## Documentation

- **Interactive Performance & Co-Design Report**: [docs/PERFORMANCE_REPORT.html](docs/PERFORMANCE_REPORT.html) — Self-contained offline HTML dashboard covering all 210 verified cases (`L1` + `L2` + `L3`), Apple M1 Max Roofline & `% SOL` audit, 16 Agent kernel deep-dives, and Compiler/DSL/Kernel Co-Design headroom analysis (`open docs/PERFORMANCE_REPORT.html`; regenerate via `python3 tools/generate_perf_report.py`)
- **User Tutorials**: [01 Installation](docs/tutorials/01_installation.md) · [02 Getting Started](docs/tutorials/02_getting_started.md) · [03 Advanced Guide](docs/tutorials/03_advanced.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Layered compiler architecture, mechanical boundaries, and UMA runtime
- [docs/FUSION_ARCHITECTURE.md](docs/FUSION_ARCHITECTURE.md) — v2.0 Epilogue DAG fusion, producer-consumer tails, `torch.compile`, concurrent batching & Compute ICB graphs
- [docs/PASS_PIPELINE.md](docs/PASS_PIPELINE.md) — All 18 pure C++20 SSA optimization passes and `m1triton-opt` CLI usage
- [docs/DSL_CODESIGN.md](docs/DSL_CODESIGN.md) — Compiler-DSL Co-design primitives (`tlx.*`) across L1, L2, and L3
- [docs/BENCHMARKS_L2.md](docs/BENCHMARKS_L2.md) — Complete 100-task KernelBench Level 2 official frozen scoreboard (`3.0130x` geomean)
- [docs/BENCHMARKS_L3.md](docs/BENCHMARKS_L3.md) — Complete 10-kernel Level 3 complex Triton suite scoreboard (`3.9235x` geomean)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — Complete 100-task KernelBench Level 1 (`v1.0-kernelbench-l1`) official scoreboard (`1.9268x` geomean)
- [.audit/experiments_v2.tsv](.audit/experiments_v2.tsv) — v2.0 performance sprint experiment ledger (kept, rejected, and circuit-broken hypotheses)
