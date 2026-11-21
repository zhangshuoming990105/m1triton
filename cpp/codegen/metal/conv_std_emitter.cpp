#include "codegen/metal/conv_std_emitter.hpp"

namespace m1triton::codegen::metal {

CompiledMSL try_emit_conv_std_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  (void)kernel;
  (void)target;
  if (matched_out != nullptr) *matched_out = false;
  return CompiledMSL{false, "", {}, ""};
}

}  // namespace m1triton::codegen::metal
