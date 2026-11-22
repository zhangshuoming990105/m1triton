#pragma once

#include "codegen/metal/metal_codegen.hpp"

namespace m1triton::codegen::metal {

CompiledMSL try_emit_l3_gemm_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched);

}  // namespace m1triton::codegen::metal
