#pragma once

#include "codegen/metal/launch_spec.hpp"
#include "ir/ir.hpp"
#include "ir/target_spec.hpp"

namespace m1triton::codegen::metal {

class MetalCodegen {
 public:
  static CompiledMSL emit(const ir::Kernel& kernel, const ir::TargetSpec& target);
};

}  // namespace m1triton::codegen::metal
