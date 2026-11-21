#pragma once

#include <cstdint>

#include "passes/pass.hpp"

namespace m1triton::passes {

enum class SgmmSchedulePolicy : uint8_t {
  Auto,
  HintGuided,
};

class TileAndLayoutAssignmentPass : public Pass {
 public:
  explicit TileAndLayoutAssignmentPass(SgmmSchedulePolicy sgmm_policy = SgmmSchedulePolicy::Auto)
      : sgmm_policy_(sgmm_policy) {}

  std::string_view name() const override { return "tile-layout"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::PostTile; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;

 private:
  SgmmSchedulePolicy sgmm_policy_;
};

}  // namespace m1triton::passes
