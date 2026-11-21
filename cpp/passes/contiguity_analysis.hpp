#pragma once

#include <cstdint>
#include <unordered_map>

#include "passes/pass.hpp"

namespace m1triton::passes {

struct IndexFact {
  int64_t step = 0;
  uint32_t contiguity = 1;
  uint32_t alignment = 1;
  bool uniform = false;
  uint32_t base_ptr_divisibility = 1;
  bool operator==(const IndexFact&) const = default;
};

class ContiguityAndAlignmentAnalysis : public Pass {
 public:
  std::string_view name() const override { return "contiguity-alignment"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::HighLevel; }
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;

  static std::unordered_map<ir::ValueId, IndexFact> analyze(const ir::Kernel& kernel);
};

}  // namespace m1triton::passes
