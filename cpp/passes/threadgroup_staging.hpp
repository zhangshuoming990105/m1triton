#pragma once

#include <cstdint>

#include "passes/pass.hpp"

namespace m1triton::passes {

class ThreadgroupStagingAndSwizzlePass : public Pass {
 public:
  std::string_view name() const override { return "threadgroup-staging"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::PostTile; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;

  static uint32_t recommended_pad_elems(ir::ScalarDType dt);
  static ir::SwizzleMode recommended_swizzle_mode(ir::ScalarDType dt);
  static uint64_t compute_dot_sram_bytes(int64_t bm, int64_t bn, int64_t bk, ir::ScalarDType dt_a,
                                         ir::ScalarDType dt_b, uint32_t pad_a, uint32_t pad_b, uint32_t stages);
};

}  // namespace m1triton::passes
