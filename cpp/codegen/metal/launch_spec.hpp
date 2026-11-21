#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ir/types.hpp"

namespace m1triton::codegen::metal {

struct ParamSlot {
  uint32_t param_index = 0;
  uint32_t buffer_index = 0;
  std::string name;
  bool is_pointer = false;
  bool writable = false;
  ir::ScalarDType dtype = ir::ScalarDType::F32;
};

struct LaunchSpec {
  std::string entry_name;
  uint32_t tg_x = 32;
  uint32_t tg_y = 1;
  uint32_t tg_z = 1;
  uint32_t threadgroup_mem_bytes = 0;
  uint32_t vec_width = 1;
  bool fast_math = false;
  std::vector<ParamSlot> param_slots;
};

struct CompiledMSL {
  bool ok = false;
  std::string msl_source;
  LaunchSpec launch_spec;
  std::string error;
};

}  // namespace m1triton::codegen::metal
