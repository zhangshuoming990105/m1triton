#pragma once

#include <optional>

#include "passes/pass.hpp"

namespace m1triton::passes {

class SimdgroupMatrixLoweringPass : public Pass {
 public:
  std::string_view name() const override { return "simdgroup-matmul"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::PreCodegen; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;

  static std::optional<ir::SimdgroupMatmulAttr> compute_sgmm_attr(const ir::Kernel& kernel, const ir::Op& dot_op,
                                                                   const ir::TargetSpec& target);
};

}  // namespace m1triton::passes
