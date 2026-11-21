#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace m1triton::runtime {

struct DeviceInfo {
  std::string name;
  uint64_t max_threads_per_threadgroup = 0;
  uint64_t max_threadgroup_memory_bytes = 0;
  uint64_t recommended_max_working_set_bytes = 0;
  bool has_unified_memory = false;
  bool available = false;
};

struct BufferArg {
  uintptr_t handle = 0;
  uint64_t byte_offset = 0;
  uint64_t nbytes = 0;
  uint32_t index = 0;
  bool is_metal_buffer = false;
  bool writable = false;
};

struct ScalarArg {
  uint32_t index = 0;
  std::vector<uint8_t> bytes;
};

struct LaunchConfig {
  uint32_t grid_x = 1;
  uint32_t grid_y = 1;
  uint32_t grid_z = 1;
  uint32_t tg_x = 1;
  uint32_t tg_y = 1;
  uint32_t tg_z = 1;
  uint32_t threadgroup_mem_bytes = 0;
  bool wait = false;
};

class MetalRuntime {
 public:
  static MetalRuntime& instance();

  DeviceInfo device_info() const;
  uint64_t dispatch_count() const;
  void reset_dispatch_count();
  std::string last_kernel_name() const;
  void synchronize();

  uint64_t compile_kernel(
      const std::string& cache_key,
      const std::string& msl_source,
      const std::string& fn_name,
      bool fast_math,
      std::string* error_out);

  double launch_kernel(
      uint64_t kernel_id,
      const std::vector<BufferArg>& buffers,
      const std::vector<ScalarArg>& scalars,
      const LaunchConfig& cfg,
      std::string* error_out);

  bool compile_msl_smoke(const std::string& msl_source, const std::string& fn_name, std::string* error_out);

 private:
  MetalRuntime();
  ~MetalRuntime();
  MetalRuntime(const MetalRuntime&) = delete;
  MetalRuntime& operator=(const MetalRuntime&) = delete;

  struct Impl;
  Impl* impl_;
};

}  // namespace m1triton::runtime
