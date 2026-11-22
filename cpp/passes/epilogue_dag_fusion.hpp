#pragma once

#include "passes/pass.hpp"

namespace m1triton::passes {

class EpilogueDAGFusionPass final : public Pass {
 public:
  std::string_view name() const override { return "EpilogueDAGFusionPass"; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;
};

}  // namespace m1triton::passes
