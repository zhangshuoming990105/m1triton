# Core Compiler and Runtime Flow

## EntryPoints

- `scripts/ci_local.sh`
- `python/m1triton/runtime.py`
- `cpp/ir/target_spec.hpp`
- `cpp/runtime/metal_runtime.mm`
- `evals/roofline_probe.py`
- `evals/eval_kernelbench.py`
- `.agents/skills/verify-m1triton/scripts/driver.mjs`

## Observable Behaviors

- `bash scripts/ci_local.sh` runs guardrails, feature map audit, C++20/ObjC++ build via Apple Clang + Ninja, CTest, and Python pytest and exits 0.
- `m1triton.runtime.device_info()` reports the active Apple M1 Max Metal device with unified memory enabled and `>= 32768` bytes threadgroup SRAM.
- `evals/roofline_probe.py` compiles and runs native Metal STREAM and SGMM kernels and reports measured GB/s and TFLOPS.

## Invariants

- `cpp/ir/` has zero includes of runtime, codegen, bindings, or Metal headers (`ARCH-IR-PURE`).
- `cpp/passes/` has zero includes of runtime, bindings, or Metal headers (`ARCH-PASSES-NO-RUNTIME`).
- Parallel verification runs in different checkouts or worktrees never share the same port or `.audit/` state path.
