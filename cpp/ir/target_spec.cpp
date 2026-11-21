#include "ir/target_spec.hpp"

namespace m1triton::ir {

TargetSpec TargetSpec::apple_m1_max() {
  return TargetSpec{};
}

bool TargetSpec::fits_threadgroup_memory(uint32_t bytes) const {
  return bytes <= max_threadgroup_memory_bytes;
}

}  // namespace m1triton::ir
