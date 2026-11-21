#pragma once

#include "codegen/metal/launch_spec.hpp"
#include "ir/ir.hpp"
#include "ir/target_spec.hpp"

namespace m1triton::codegen::metal {

// Emits an Apple M1 Max hardware SIMD-group reduction/scan MSL kernel
// (simd_sum, simd_max, simd_min, simd_shuffle_xor, simd_prefix_inclusive_sum,
// cross-SIMD threadgroup reduction) for kernels containing Reduce / Scan / GEMV patterns.
// Sets *matched_out = false if the kernel is not a reduction/scan kernel.
CompiledMSL try_emit_reduce_scan_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out);

}  // namespace m1triton::codegen::metal
