#pragma once

#include "passes/pass.hpp"

namespace m1triton::passes {

class DCEPass : public Pass {
 public:
  std::string_view name() const override { return "dce"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::HighLevel; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;
};

}  // namespace m1triton::passes
