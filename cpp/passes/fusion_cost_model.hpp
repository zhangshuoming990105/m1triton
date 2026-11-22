#pragma once

#include <cstdint>
#include <string_view>

#include "ir/ir.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/pass.hpp"

namespace m1triton::passes {

struct FusionCostReport {
  uint64_t estimated_sram_bytes = 0;
  uint32_t estimated_registers_per_thread = 0;
  uint32_t threads_per_threadgroup = 0;
  uint32_t max_resident_tgs_per_core = 0;
  double estimated_occupancy = 0.0;
  uint32_t fused_epilogue_ops = 0;
  uint32_t epilogue_saved_regs = 0;
  bool has_fused_tail = false;
  double arithmetic_intensity = 0.0;
  bool clamped_tile = false;
  bool disabled_double_buffer = false;
  bool clamped_simdgroups = false;
};

class FusionOccupancyCostModelPass : public Pass {
 public:
  std::string_view name() const override { return "fusion-cost-model"; }
  ir::VerifyStage output_stage() const override { return ir::VerifyStage::HighLevel; }

  static FusionCostReport analyze(const ir::Kernel& kernel, const ir::TargetSpec& target);
  PassResult run(ir::Kernel& kernel, const PassContext& ctx) override;

  const FusionCostReport& last_report() const { return last_report_; }

 private:
  FusionCostReport last_report_{};
};

}  // namespace m1triton::passes
