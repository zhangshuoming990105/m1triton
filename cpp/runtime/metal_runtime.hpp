#pragma once

#include <cstddef>
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
  bool supports_metal3 = false;
  bool supports_indirect_command_buffers = false;
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

  uint64_t kernel_max_threads_per_threadgroup(uint64_t kernel_id) const;
  bool kernel_supports_icb(uint64_t kernel_id) const;

  double launch_kernel(
      uint64_t kernel_id,
      const std::vector<BufferArg>& buffers,
      const std::vector<ScalarArg>& scalars,
      const LaunchConfig& cfg,
      std::string* error_out);

  bool begin_batch(std::string* error_out);
  double end_batch(bool wait, std::string* error_out);
  void abort_batch();
  bool in_batch() const;

  bool begin_graph_capture(std::string* error_out);
  uint64_t end_graph_capture(std::string* error_out);
  void abort_graph_capture();
  bool in_graph_capture() const;
  double replay_graph(uint64_t graph_id, bool wait, std::string* error_out);
  void destroy_graph(uint64_t graph_id);
  size_t graph_command_count(uint64_t graph_id) const;

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
