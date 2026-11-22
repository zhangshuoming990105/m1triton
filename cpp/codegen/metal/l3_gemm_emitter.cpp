#include "codegen/metal/l3_gemm_emitter.hpp"

namespace m1triton::codegen::metal {

CompiledMSL try_emit_l3_gemm_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched) {
  (void)kernel;
  (void)target;
  if (matched != nullptr) *matched = false;
  return CompiledMSL{false, "", {}, "l3_gemm_emitter not yet implemented"};
}

}  // namespace m1triton::codegen::metal
