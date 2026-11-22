# Tutorial 3: Advanced Guide (`tlx` Co-Design, Execution Graphs, Custom Autograd & Compiler Tooling)

This guide covers low-level Apple Silicon optimization and compiler workflows in M1Triton:
1. Expressing hardware-aware tile schedules and register-level fusion via `m1triton.tlx`.
2. Eliminating dispatch overhead with concurrent `command_batch()` and static `MetalGraph` (Compute ICB).
3. Integrating multi-kernel forward + backward operators with `torch.autograd.Function`.
4. Inspecting SSA IR, generated MSL, occupancy cost-model stats, and custom passes with `m1triton-opt`.

---

## 1. Apple Silicon Hardware Model & `tlx.*` Co-Design

Apple M1 Max GPUs have four architectural properties that govern peak performance:
- **32 KB threadgroup SRAM** (`32,768 B`): Keeping tile staging under `16 KB` allows 2+ resident threadgroups per core; exceeding `32 KB` fails compilation or spills.
- **32-thread SIMD-groups** and **`simdgroup_float8x8` / `simdgroup_half8x8` MMA**: Matrix tile shapes should be multiples of `32` in `M`/`N` and `8` in `K`.
- **16-byte aligned `float4` loads/stores**: Annotating unit-stride elementwise kernels with `tlx.vec_width(4)` unlocks ~96–99% of measured STREAM bandwidth (`373.8 GB/s`).
- **Unified Memory (UMA)**: Avoiding intermediate tensor DRAM round-trips via register epilogues (`tlx.epilogue`) and reduction tails (`tlx.fused_tail`) yields the largest speedups on multi-op chains (`3.01x` geomean across KernelBench L2).

### Example: Single-Dispatch Fused `Matmul + Bias + GELU` via `tlx.tile_spec` + `tlx.epilogue`

```python
import torch
import torch.nn.functional as F
import m1triton
import m1triton.language as tl
import m1triton.tlx as tlx

@m1triton.jit
def fused_linear_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
    NUM_SG: tl.constexpr = 4,
):
    tlx.tile_spec(BLOCK_M, BLOCK_N, BLOCK_K, num_simdgroups=NUM_SG, double_buffer=False, swizzle_w=8)
    tlx.epilogue((("add_bias", bias_ptr), ("gelu",)))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        b = tl.load(b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M, N, K = 512, 1024, 1024
a = torch.randn(M, K, device="mps", dtype=torch.float32)
b = torch.randn(K, N, device="mps", dtype=torch.float32)
bias = torch.randn(N, device="mps", dtype=torch.float32)
out = torch.empty(M, N, device="mps", dtype=torch.float32)

grid = (m1triton.cdiv(M, 64), m1triton.cdiv(N, 64))
fused_linear_gelu_kernel[grid](
    a, b, bias, out, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
    wait=True, sync_mps=True,
)
ref = F.gelu(a @ b + bias)
torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)
print("Fused Linear+Bias+GELU PASS (1 Metal dispatch)")
```

Other high-leverage `tlx.*` primitives (see [docs/DSL_CODESIGN.md](../DSL_CODESIGN.md) for the complete reference):
- `tlx.structure("upper_triangular" | "lower_triangular" | "diagonal" | "symmetric")`: Prunes dead tiles in structured matmuls.
- `tlx.fused_tail("row_sum" | "row_max" | "row_logsumexp" | "pool1d_max" | "spatial_gap", ...)`: Fuses producer-consumer reductions inside registers/SRAM.
- `tlx.flash_attention_tile` / `flash_attention_bwd_dkdv` / `flash_attention_bwd_dq` / `paged_attention_decode`: Attention forward, SRAM-recomputed backward, GQA, sliding window, ALiBi, softcap, and vLLM paged decode.
- `tlx.grouped_gemm` / `tlx.quant_dot`: Persistent MoE grouped GEMM and W4A16 in-register quantized matmul.

---

## 2. Metal Runtime Execution Graphs: `command_batch()` vs. `MetalGraph`

Metal does not expose CUDA Graphs through PyTorch MPS, so M1Triton provides two native execution-graph primitives in `m1triton.runtime`:

| Primitive | Underlying Metal API | Best For | Key Property |
| --- | --- | --- | --- |
| `rt.command_batch(wait=True, sync_mps=True)` | Single `MTLCommandBuffer` + `MTLDispatchTypeConcurrent` encoder | Dynamic shapes or per-call autograd buffers | Automatic `buffers_overlap` RAW/WAW/WAR barriers; independent branches run concurrently on GPU |
| `rt.MetalGraph(sync_mps=True)` | Metal 3 `MTLIndirectCommandBuffer` (Compute ICB) | Static repeated inference/decode pipelines with fixed buffer addresses | Lowest CPU & wall-clock replay latency (`0.015 ms` CPU submit; `1.47x` faster wall than `command_batch` on 3-kernel FA2 backward `[measured]`) |

### Example: Concurrent Diamond Pipeline in Both `command_batch()` and `MetalGraph`

```python
import torch
import m1triton
import m1triton.language as tl
import m1triton.runtime as rt

@m1triton.jit
def affine_kernel(x_ptr, out_ptr, scale, bias, n, BLOCK: tl.constexpr = 256):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * scale + bias, mask=mask)

@m1triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr = 256):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)

n = 65536
grid = (m1triton.cdiv(n, 256),)
x = torch.randn(n, device="mps", dtype=torch.float32)
b1, b2, out = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)

# 1. Concurrent command_batch(): b1 and b2 run concurrently; automatic RAW barrier before add_kernel
with rt.command_batch(wait=True, sync_mps=True) as batch:
    affine_kernel[grid](x, b1, 2.0, 1.0, n, sync_mps=False)
    affine_kernel[grid](x, b2, -0.5, 3.0, n, sync_mps=False)
    add_kernel[grid](b1, b2, out, n, sync_mps=False)
torch.testing.assert_close(out.cpu(), ((x * 2.0 + 1.0) + (x * -0.5 + 3.0)).cpu(), atol=1e-5, rtol=1e-5)

# 2. Static MetalGraph (Compute ICB): capture once, mutate x in place, replay repeatedly
with rt.MetalGraph(sync_mps=True) as graph:
    affine_kernel[grid](x, b1, 2.0, 1.0, n, sync_mps=False)
    affine_kernel[grid](x, b2, -0.5, 3.0, n, sync_mps=False)
    add_kernel[grid](b1, b2, out, n, sync_mps=False)

for step in range(3):
    x_new = torch.randn(n, device="mps", dtype=torch.float32) + float(step)
    x.copy_(x_new)
    gpu_ms = graph.replay(wait=True)
    ref = (x_new * 2.0 + 1.0) + (x_new * -0.5 + 3.0)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-5, rtol=1e-5)
graph.close()
print(f"command_batch={batch.gpu_ms:.3f}ms | MetalGraph replay={gpu_ms:.3f}ms (cmds=3)")
```

> [!IMPORTANT]
> Inside `command_batch()` and `MetalGraph`, pass `wait=False` and `sync_mps=False` to inner launches (`wait=True` is forbidden mid-batch). For autotuned kernels used inside `command_batch()`, warm up or pre-select configs before entering the batch so autotuning runs outside the active command buffer.

---

## 3. Custom Differentiable Operators (`torch.autograd.Function` + Metal 3 Atomics)

To integrate multi-kernel forward and backward passes cleanly with `torch.autograd.grad` and `.backward()`:
1. Dispatch the forward kernel with `wait=True, sync_mps=True` (or sync before returning tensors consumed by PyTorch MPS ops) and save minimal state (`ctx.save_for_backward(...)`, e.g., `out` and `lse` rather than an `O(N^2)` score matrix).
2. In `backward()`, allocate gradients on `"mps"` and launch dependent + parallel backward kernels inside a single `with rt.command_batch(wait=True, sync_mps=True):` block (for instance, `delta -> (bwd_dkdv || bwd_dq)`).
3. For parameter gradient reductions across threadgroups (like LayerNorm `dgamma`/`dbeta`), use `tl.atomic_add(ptr, val, mask=mask)`, which compiles directly to native Metal 3.0 `atomic_fetch_add_explicit` on `device atomic_float*`.

Complete end-to-end references in the repository:
- [examples/04_flash_attention_fwd_bwd.py](../../examples/04_flash_attention_fwd_bwd.py) — Differentiable causal FlashAttention-2 forward + backward.
- [examples/05_w4a16_quant_gemm.py](../../examples/05_w4a16_quant_gemm.py) — W4A16 group-quantized inference GEMM with `8x` FP32 memory reduction.
- [benchmarks/triton_l3/kernels/](../../benchmarks/triton_l3/kernels/) — All 10 complex L3 kernels (GQA, FA2 bwd, PagedAttention, Sliding-Window/ALiBi/Softcap, RMSNorm+RoPE+KV, LayerNorm bwd atomics, Chunked Linear Cross-Entropy, Mamba SSM scan, Persistent MoE Grouped GEMM, W4A16).

---

## 4. Compiler Inspection, Profiling & `m1triton-opt`

Every `@m1triton.jit` kernel exposes first-class IR and MSL introspection without launching on the GPU:

```python
import numpy as np
import m1triton
import m1triton.language as tl

@m1triton.jit
def saxpy(x_ptr, y_ptr, a, n, BLOCK: tl.constexpr = 256):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * a + tl.load(y_ptr + offs, mask=mask), mask=mask)

x = np.zeros(1024, dtype=np.float32)
y = np.zeros(1024, dtype=np.float32)
pre_ir, post_ir = saxpy.inspect_ir(x, y, 2.0, 1024, BLOCK=256)
msl_source = saxpy.inspect_msl(x, y, 2.0, 1024, BLOCK=256)
print(f"Pre-pass IR lines={len(pre_ir.splitlines())}, Post-pass IR lines={len(post_ir.splitlines())}, MSL bytes={len(msl_source)}")
```

### Standalone Pass Inspection & Cost-Model Telemetry (`m1triton-opt`)

Pipe textual SSA IR directly into `m1triton-opt` to run the full pipeline (`m1` or `m2`), test an individual pass sequence, dump IR before/after each pass, inspect `FusionOccupancyCostModelPass` estimates (`sram_est`, `regs_est`, `occ_est`, `fused_ops`, `ai`), or emit MSL:

```bash
# Save pre-pass IR from an existing kernel (e.g. examples/01_vector_add.py) and inspect pipeline stats + MSL
python -c "
import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('ex1', 'examples/01_vector_add.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
a = np.zeros(1024, np.float32)
pre_ir, _ = mod.vector_add_kernel.inspect_ir(a, a, a, 1024, BLOCK_SIZE=256)
print(pre_ir)
" | m1triton-opt --passes=m2 --stats --emit-msl

# Run a custom pass subset and dump IR after each pass
m1triton-opt --passes=canonicalize,cse,epilogue-dag-fusion,algebraic-norm-fold,fusion-cost-model --dump-ir-after-all --stats kernel.m1ir
```

### Live Runtime Profiling & Validation

```bash
# Print per-dispatch kernel name, grid, threadgroup size, vector width, and GPU ms
M1TRITON_PROFILE=1 python examples/01_vector_add.py

# Dump pre/post IR and generated MSL on compilation
M1TRITON_DUMP_IR=1 M1TRITON_DUMP_MSL=1 python examples/01_vector_add.py

# Validate Metal API usage, buffer lifetimes, and barriers via Apple's debug layer
MTL_DEBUG_LAYER=1 pytest -q tests/python/test_v2_runtime_graphs.py
```

For architectural deep-dives, see [docs/ARCHITECTURE.md](../ARCHITECTURE.md), [docs/FUSION_ARCHITECTURE.md](../FUSION_ARCHITECTURE.md), and [docs/PASS_PIPELINE.md](../PASS_PIPELINE.md).
