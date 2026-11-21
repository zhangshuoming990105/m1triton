#pragma once

#include <string>
#include <vector>

#include "ir/ir.hpp"
#include "ir/target_spec.hpp"

namespace m1triton::ir {

enum class VerifyStage : uint8_t {
  HighLevel,
  PostTile,
  PreCodegen,
};

struct VerifyResult {
  bool ok = true;
  std::vector<std::string> errors;

  std::string summary() const;
};

class IRVerifier {
 public:
  static VerifyResult verify(const Kernel& kernel, const TargetSpec& target,
                             VerifyStage stage = VerifyStage::HighLevel);
  static VerifyResult verify(const Module& mod, const TargetSpec& target,
                             VerifyStage stage = VerifyStage::HighLevel);
};

}  // namespace m1triton::ir
