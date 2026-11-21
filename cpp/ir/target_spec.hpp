#pragma once

#include <cstdint>
#include <string>

namespace m1triton::ir {

struct TargetSpec {
  std::string name = "Apple M1 Max";
  uint32_t gpu_cores = 32;
  uint32_t simd_width = 32;
  uint32_t max_threads_per_threadgroup = 1024;
  uint32_t max_threadgroup_memory_bytes = 32768;
  uint32_t sgmm_m = 8;
  uint32_t sgmm_n = 8;
  uint32_t sgmm_k = 8;
  uint32_t vector_bytes = 16;
  double peak_bandwidth_gbs = 400.0;
  double peak_fp32_tflops = 10.4;
  double peak_fp16_tflops = 20.8;

  static TargetSpec apple_m1_max();
  bool fits_threadgroup_memory(uint32_t bytes) const;
};

}  // namespace m1triton::ir
