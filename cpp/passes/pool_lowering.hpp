#pragma once

#include "passes/pass.hpp"

namespace m1triton::passes {

class PoolLoweringPass final : public Pass {
 public:
  std::string_view name() const override { return "pool-lowering"; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;
};

}  // namespace m1triton::passes
