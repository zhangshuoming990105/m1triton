#pragma once

#include "passes/pass.hpp"

namespace m1triton::passes {

class ConvLoweringPass final : public Pass {
 public:
  std::string_view name() const override { return "conv-lowering"; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;
};

}  // namespace m1triton::passes
