#pragma once

#include "codegen/metal/launch_spec.hpp"
#include "ir/ir.hpp"
#include "ir/target_spec.hpp"

namespace m1triton::codegen::metal {

CompiledMSL try_emit_conv_transpose_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out);

}  // namespace m1triton::codegen::metal
