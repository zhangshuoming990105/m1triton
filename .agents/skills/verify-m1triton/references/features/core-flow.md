# Core Compiler and Runtime Flow

## EntryPoints

- `scripts/ci_local.sh`
- `python/m1triton/__init__.py`
- `python/m1triton/jit.py`
- `python/m1triton/ast_lowering.py`
- `python/m1triton/runtime.py`
- `cpp/ir/ir.hpp`
- `cpp/passes/pass_manager.hpp`
- `cpp/codegen/metal/metal_codegen.hpp`
- `cpp/runtime/metal_runtime.mm`
- `evals/roofline_probe.py`
- `evals/eval_kernelbench.py`
- `.agents/skills/verify-m1triton/scripts/driver.mjs`

## Observable Behaviors

- `@m1triton.jit` / `@triton.jit` lowers Python AST to C++20 SSA IR (`IRBuilder`), runs `PassManager::build_m1_pipeline()`, verifies IR invariants, emits readable MSL (`MetalCodegen`), compiles via native Metal `newLibraryWithSource` + `MTLBinaryArchive`, and executes zero-copy over PyTorch MPS `MTLBuffer`s.
- `bash scripts/ci_local.sh` runs guardrails, feature map audit, C++20/ObjC++ build (`-Wall -Wextra -Werror`), CTest (with `-UNDEBUG`), and Python pytest and exits 0.
- `evals/eval_kernelbench.py` enforces static AST, runtime `TorchDispatchMode`, and native Metal dispatch count anti-cheat gates.

## Invariants

- `cpp/ir/` has zero includes of runtime, codegen, bindings, Metal, or pybind11 (`ARCH-IR-PURE`).
- `cpp/passes/` has zero includes of runtime, bindings, Metal, or pybind11 (`ARCH-PASSES-NO-RUNTIME`).
- `evals/eval_kernelbench.py` and `evals/roofline_probe.py` remain SHA-256 frozen in `guardrails.json` (`FROZEN-EVAL`).
- Parallel verification runs in different checkouts or worktrees never share the same port or `.audit/` state path.
