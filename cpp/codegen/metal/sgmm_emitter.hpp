#pragma once

#include "codegen/metal/launch_spec.hpp"
#include "ir/ir.hpp"
#include "ir/target_spec.hpp"

namespace m1triton::codegen::metal {

// Emits an Apple M1 Max simdgroup_matrix_multiply_accumulate (8x8) MSL kernel
// when the IR kernel represents a standard/batched/transposed/structured/split-K
// 2D tiled GEMM (ForLoop + Dot + Store/AtomicAdd).
// Returns std::nullopt-equivalent (ok == false, error empty) when the kernel does
// not match the SGMM fast path so MetalCodegen falls back cleanly to general 2D emission.
CompiledMSL try_emit_sgmm_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out);

}  // namespace m1triton::codegen::metal
