# M1Triton Engineering Contract

Apple M1 Max specialized Triton-compatible + Co-designed compiler (`python/m1triton` -> `cpp/ir` -> `cpp/passes` -> `cpp/codegen/metal` -> `cpp/runtime` -> `cpp/bindings`).

## Build and Verification

- Local CI parity: `bash scripts/ci_local.sh`
- Guardrails and frozen-eval check: `node tools/check-guardrails.mjs`
- Native compiler toolchain on macOS: Apple Clang `/usr/bin/clang++` (Objective-C++ + `Metal.framework`, runtime `newLibraryWithSource` + `MTLBinaryArchive`; `xcrun metal` is not used).

## Rule Ledger

| Rule ID | Layer | Enforcer | Rationale |
| --- | --- | --- | --- |
| `ARCH-IR-PURE` | `cpp/ir` | `guardrail:ARCH-IR-PURE` | Core SSA IR stays pure C++20 without Metal, codegen, bindings, or runtime coupling. |
| `ARCH-PASSES-NO-RUNTIME` | `cpp/passes` | `guardrail:ARCH-PASSES-NO-RUNTIME` | Optimization passes depend only on IR and `TargetSpec`, enabling host unit testing and target portability. |
| `ARCH-FRONTEND-NO-TORCH-CHEAT` | `benchmarks/kernels` | `guardrail:ARCH-FRONTEND-NO-TORCH-CHEAT` | Candidate KernelBench operators must compile and dispatch M1Triton Metal kernels with zero PyTorch compute fallbacks. |
| `FROZEN-EVAL` | `evals` | `guardrail:FROZEN-EVAL` | Evaluation and roofline scripts are SHA-256 locked once vetted and must never be edited to pass benchmarks. |
| `COMMENT-HYGIENE` | `repo` | `guardrail:COMMENT-HYGIENE` | Rejects narration comments, unstructured hack comments, and unapproved lint/type suppressions. |
| `LOCAL-CI-PARITY` | `ci` | `script:scripts/ci_local.sh` | Every branch runs guardrails, C++ CTest, and Python pytest before merge. |
