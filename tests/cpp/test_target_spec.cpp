#include <cassert>
#include <iostream>

#include "ir/target_spec.hpp"

int main() {
  const auto spec = m1triton::ir::TargetSpec::apple_m1_max();
  assert(spec.gpu_cores == 32);
  assert(spec.simd_width == 32);
  assert(spec.max_threads_per_threadgroup == 1024);
  assert(spec.max_threadgroup_memory_bytes == 32768);
  assert(spec.fits_threadgroup_memory(32768));
  assert(!spec.fits_threadgroup_memory(32769));
  std::cout << "test_target_spec OK\n";
  return 0;
}
