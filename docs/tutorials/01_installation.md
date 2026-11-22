# Tutorial 1: Installation, Verification & Troubleshooting

This guide walks through installing M1Triton (`0.2.0`) on an Apple Silicon Mac, verifying GPU and Metal 3 readiness with `m1triton doctor`, configuring disk caches, and resolving common setup issues.

## 1. Prerequisites

| Requirement | Supported Version / Details |
| --- | --- |
| **Hardware** | Apple Silicon Mac (M1 / M2 / M3 / M4 family; tuned and benchmarked on Apple M1 Max) |
| **OS** | macOS with Apple Command Line Tools (`/usr/bin/clang++` + `Metal.framework`) |
| **Xcode** | **Not required** (M1Triton compiles MSL at runtime via `MTLDevice::newLibraryWithSource` + `MTLBinaryArchive`, not `xcrun metal`) |
| **Build Tools** | `cmake` (`>= 3.20`), `ninja` (optional but strongly recommended for fast parallel builds) |
| **Python** | Python `>= 3.11` with `numpy >= 1.26`, `pybind11 >= 2.11`, and PyTorch with MPS enabled |

Install Command Line Tools and CMake/Ninja if not already present:

```bash
xcode-select --install          # provides /usr/bin/clang++ and Metal.framework
brew install cmake ninja        # C++20 / Objective-C++ build tools
```

## 2. Clean Virtual Environment Install

```bash
git clone https://github.com/zhangshuoming990105/m1triton.git
cd m1triton

# Create and activate an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (Apple Silicon MPS)
pip install --upgrade pip torch

# Build and install M1Triton (_C native Metal extension + m1triton-opt CLI)
# Standard PEP 517 editable install (automatically fetches setuptools/wheel/pybind11):
pip install -e ".[dev]"

# Or standard non-editable install:
# pip install .

# Offline / preinstalled build environment (requires setuptools, wheel, pybind11 installed in the venv):
# pip install --no-build-isolation -e ".[dev]"
```

> [!NOTE]
> `setup.py` automatically detects `/usr/bin/clang++`, Ninja, and `pybind11.get_cmake_dir()`, builds both the native `m1triton._C` extension and the standalone `m1triton-opt` compiler CLI, and bundles both inside the package. No `PYTHONPATH` exports are required after installation.

## 3. Verify Your Environment with `m1triton doctor`

Run the built-in environment diagnostic and GPU smoke test from any directory:

```bash
m1triton doctor
```

Expected healthy checks:
- **Apple Clang**: `/usr/bin/clang++` detected
- **Metal GPU**: Apple Silicon device detected (`has_unified_memory: true`, `32 KB` threadgroup SRAM, `supports_metal3: true`, `supports_indirect_command_buffers: true`)
- **PyTorch MPS**: `mps_available: true`
- **Smoke Tests**: Compiled Metal VectorAdd (`~300–370 GB/s` on M1 Max, `max_abs_err = 0.0`) and FP32 SGMM (`128x128x128`, `max_abs_err = 0.0`)

Machine-readable JSON output and fast non-GPU checks are also available:

```bash
m1triton doctor --json          # structured JSON report for CI / automation
m1triton doctor --no-smoke      # check toolchain & paths without running GPU kernels
m1triton-opt --help             # verify standalone IR pass & MSL CLI is on PATH
```

## 4. Run the Bundled Examples

Once `m1triton doctor` reports healthy, run the five progressive examples:

```bash
python examples/01_vector_add.py               # @m1triton.jit bandwidth benchmark
python examples/02_autotuned_matmul.py         # @m1triton.autotune + @m1triton.heuristics
python examples/03_torch_compile_fusion.py     # torch.compile(backend="m1triton") vs Eager & Inductor
python examples/04_flash_attention_fwd_bwd.py  # FlashAttention-2 forward + backward (torch.autograd)
python examples/05_w4a16_quant_gemm.py         # W4A16 4-bit quantized inference GEMM
```

## 5. Cache Directories & Environment Variables

M1Triton maintains two persistent disk caches so cold starts skip MSL compilation and autotuning:

| Cache | Default Location | Purpose |
| --- | --- | --- |
| Metal Binary Archive | `./.m1triton_cache/pipelines_metal3.metallib` | Compiled Apple GPU machine-code pipelines (`MTLBinaryArchive`) |
| Autotuner Winners | `~/.cache/m1triton/autotune/*.json` | Winning `Config` per `(kernel_sha256, gpu_name, key, dtypes, device)` |

All runtime and debugging environment variables:

| Variable | Values | Effect |
| --- | --- | --- |
| `M1TRITON_CACHE_DIR` | Directory path | Relocates both Metal binary archive (`<dir>/pipelines_metal3.metallib`) and autotune cache (`<dir>/autotune/`) |
| `M1TRITON_DISABLE_AUTOTUNE_CACHE` | `1` / `true` | Ignores autotune disk cache and re-benchmarks candidate configs |
| `M1TRITON_STRICT_COMPILE` | `1` / `true` | Forces `torch.compile(..., backend="m1triton")` to error instead of partitioning around unsupported FX ops |
| `M1TRITON_DUMP_IR` | `1` / `true` | Prints pre-pass and post-pass C++20 SSA IR to `stderr` on kernel compilation |
| `M1TRITON_DUMP_MSL` | `1` / `true` | Prints generated Metal Shading Language source to `stderr` |
| `M1TRITON_PROFILE` | `1` / `true` | Forces synchronous launch timing and logs `[M1TRITON_PROFILE]` GPU latency per dispatch |
| `MTL_DEBUG_LAYER` | `1` | Enables Apple's Metal API validation layer (useful when debugging custom runtime/ICB flows) |

## 6. Troubleshooting

- **`Cannot import 'setuptools.build_meta'`**: Python 3.12+ virtual environments no longer pre-install `setuptools`. Run standard `pip install -e ".[dev]"` (without `--no-build-isolation`), or install `pip install setuptools wheel pybind11` first when using `--no-build-isolation`.
- **`pybind11` or `cmake` not found during `pip install`**: Ensure `cmake` is on `PATH` (`brew install cmake ninja`). If using `--no-build-isolation`, install `pip install setuptools wheel pybind11` in the active virtualenv first.
- **`Apple Clang compiler (/usr/bin/clang++) is required`**: Run `xcode-select --install`. Avoid pointing `CXX` to Homebrew GCC/LLVM when building Objective-C++ `Metal.framework` sources (`setup.py` prefers `/usr/bin/clang++` automatically).
- **`torch.backends.mps.is_available()` is `False`**: Verify you installed an `arm64` native Python and PyTorch build (not Rosetta x86_64): `python -c "import platform, torch; print(platform.machine(), torch.backends.mps.is_available())"` should print `arm64 True`.
- **Non-contiguous CPU tensor error**: PyTorch MPS tensors support arbitrary strided views directly (zero-copy `MTLBuffer` + byte offset), while CPU NumPy/PyTorch buffers must be contiguous (`x.contiguous()` or `np.ascontiguousarray(x)`).
- **`OSError: could not get source code` in `python -c`**: `@m1triton.jit` parses kernel Python AST via `inspect.getsource`, so define kernels in a `.py` file or Jupyter cell rather than an inline `python -c "..."` string.
- **Resetting stale caches**: Delete `~/.cache/m1triton/autotune/` and `./.m1triton_cache/`, or set `M1TRITON_CACHE_DIR=/path/to/fresh_dir`.

**Next**: Continue to [Tutorial 2: Getting Started (`@jit`, `torch.compile`, and `@autotune`)](02_getting_started.md).
