#pragma once

#include "passes/pass.hpp"

namespace m1triton::passes {

class TileAndLayoutAssignmentPass : public Pass {
 public:
  std::string_view name() const override { return "tile-layout"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::PostTile; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;
};

}  // namespace m1triton::passes
