# Tutorial 2: Getting Started (`@jit`, `torch.compile`, and `@autotune`)

This tutorial covers the three core workflows for everyday M1Triton use:
1. Writing custom GPU kernels with `@m1triton.jit` (and drop-in `triton` compatibility).
2. Accelerating existing PyTorch models with zero kernel rewrites via `torch.compile(model, backend="m1triton")`.
3. Tuning block sizes and SIMD-group schedules automatically with `@m1triton.autotune` and `@m1triton.heuristics`.

---

## 1. Your First Kernel with `@m1triton.jit`

M1Triton follows Triton's block-programming model (`program_id`, `arange`, masked `load`/`store`, and `tl.constexpr` compile-time constants). Each kernel is parsed from Python AST into C++20 SSA IR, optimized across 18 passes, and compiled to Metal Shading Language.

```python
import torch
import m1triton
import m1triton.language as tl

@m1triton.jit
def fused_axpy_relu_kernel(x_ptr, y_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    z = x * alpha + y
    tl.store(out_ptr + offs, tl.maximum(z, 0.0), mask=mask)

n = 4_000_000
x = torch.randn(n, device="mps", dtype=torch.float32)
y = torch.randn(n, device="mps", dtype=torch.float32)
out = torch.empty_like(x)

grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)
compiled = fused_axpy_relu_kernel[grid](
    x, y, out, 2.5, n, BLOCK_SIZE=1024, wait=True, sync_mps=True
)

ref = torch.relu(x * 2.5 + y)
torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-5, rtol=1e-5)
print(f"PASS | gpu_ms={compiled.last_gpu_ms:.3f} ms")
```

### Key Launch Rules on Apple Silicon MPS
- **`sync_mps=True` (default)**: Synchronizes PyTorch's MPS command stream before M1Triton reads MPS tensors so prior PyTorch writes are visible. Set `sync_mps=False` when chaining multiple M1Triton kernels back-to-back.
- **`wait=True` vs `wait=False` (default)**: M1Triton dispatches onto its own `MTLCommandQueue`. Pass `wait=True` on the last kernel before reading the output tensor from PyTorch MPS or CPU (or call `m1triton.runtime.synchronize()`).
- **Drop-in Triton namespace**: Existing Triton code can switch imports with one line:

```python
import m1triton.triton_compat as triton
import m1triton.language as tl
# triton.jit, triton.autotune, triton.heuristics, triton.Config, triton.cdiv, triton.next_power_of_2
```

See [examples/01_vector_add.py](../../examples/01_vector_add.py) for a complete bandwidth benchmark (`~320 GB/s` on M1 Max).

---

## 2. Accelerating PyTorch Models with `torch.compile(backend="m1triton")`

Importing `m1triton` automatically registers the `"m1triton"` backend with PyTorch TorchDynamo + AOTAutograd. M1Triton reuses PyTorch's MPS Unified Memory allocator and FX graph capture while replacing unfused ATen op sequences with fused Metal kernels:
- **Pointwise & broadcast chains** -> single vectorized `float4` kernel (`dpf=1`).
- **Linear / Matmul / Conv2d / Conv3d / ConvTranspose + Bias + Residual + Activations + Reduction/Pool tails** -> register-level `EpilogueDAGAttr` and `FusedTailAttr` fused kernels dispatched inside a single `command_batch()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import m1triton
import m1triton.runtime as rt

class MLPBlock(nn.Module):
    def __init__(self, d_model: int = 1024, d_ff: int = 4096):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x))) + x

model = MLPBlock().to("mps").eval()
x = torch.randn(512, 1024, device="mps", dtype=torch.float32)

# Standard PyTorch API (or helper: m1triton.compile(model))
compiled_model = torch.compile(model, backend="m1triton")

with torch.no_grad():
    ref = model(x)
    _ = compiled_model(x)  # warm up FX capture + Metal binary cache
    rt.reset_dispatch_count()
    out = compiled_model(x)
    dispatches = rt.get_dispatch_count()

torch.testing.assert_close(out.cpu(), ref.cpu(), atol=1e-3, rtol=1e-3)
print(f"PASS | M1Triton dispatches={dispatches} | cache={m1triton.compile_backend.get_cache_stats()}")
```

### Strict Mode vs. General User Mode
- **Default mode** (`torch.compile(model, backend="m1triton")`): Fuses supported subgraphs into M1Triton Metal kernels and cleanly partitions around unsupported ops (e.g., host control flow or exotic sparse ops) with safe MPS boundaries.
- **Strict mode** (`m1triton.compile(model, strict=True)` or `M1TRITON_STRICT_COMPILE=1`): Raises an error if any op cannot be lowered directly to M1Triton Metal kernels—useful when verifying 100% native compilation coverage.

Run [examples/03_torch_compile_fusion.py](../../examples/03_torch_compile_fusion.py) to compare PyTorch MPS Eager, TorchInductor MPS, and M1Triton side by side.

---

## 3. Autotuning with `@m1triton.autotune` and `@m1triton.heuristics`

M1Triton's autotuner mirrors Triton's decorator API while tailoring search to Apple Silicon hardware:
- Maps `Config(..., num_warps=k)` to `k` Apple **32-thread SIMD-groups** (`k * 32` threads/threadgroup).
- Automatically prunes configs exceeding M1 Max's **32 KB threadgroup SRAM** or the compiled Metal Pipeline State Object's live `[pso maxTotalThreadsPerThreadgroup]` limit.
- Pre-compiles all feasible configs *before* timing so Apple Silicon GPU DVFS clock ramp-up never biases early candidates, measures native `MTLCommandBuffer` GPU timestamps, and saves winners atomically under `~/.cache/m1triton/autotune/`.

```python
import torch
import m1triton
import m1triton.language as tl

@m1triton.autotune(
    configs=[
        m1triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        m1triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        m1triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
    reset_to_zero=["out_ptr"],  # supports both PyTorch tensors and NumPy arrays
    warmup=3,
    rep=10,
)
@m1triton.heuristics(
    {
        "EVEN_N": lambda args: int(args["n_elements"] % args["BLOCK_SIZE"] == 0),
    }
)
@m1triton.jit
def tuned_scale_kernel(
    x_ptr, out_ptr, scale, n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN_N: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_N == 1:
        x = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, x * scale)
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, x * scale, mask=mask)

n = 1_048_576
x = torch.randn(n, device="mps", dtype=torch.float32)
out = torch.empty_like(x)
grid = lambda meta: (m1triton.cdiv(n, meta["BLOCK_SIZE"]),)
tuned_scale_kernel[grid](x, out, 3.0, n, wait=True)

torch.testing.assert_close(out.cpu(), (x * 3.0).cpu(), atol=1e-5, rtol=1e-5)
print("Best config:", tuned_scale_kernel.best_config)
```

For in-place accumulation kernels (such as `tl.atomic_add`), pass `restore_value=["acc_ptr"]` to snapshot and restore buffers before each candidate benchmark trial so tuning never corrupts inputs.

See [examples/02_autotuned_matmul.py](../../examples/02_autotuned_matmul.py) for a full autotuned SIMD-group matrix multiplication example.

**Next**: Continue to [Tutorial 3: Advanced Guide (`tlx` Fusion, Metal Graphs, Custom Autograd & `m1triton-opt`)](03_advanced.md).
