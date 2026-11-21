#pragma once

#include <ostream>
#include <string>
#include <string_view>

#include "ir/ir.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"

namespace m1triton::passes {

struct PassResult {
  bool changed = false;
  bool ok = true;
  std::string diagnostic;
};

struct PassContext {
  const ir::TargetSpec& target;
  bool verify_each_pass = true;
  bool dump_ir = false;
  std::ostream* dump_stream = nullptr;
};

class Pass {
 public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  virtual ir::VerifyStage output_stage() const { return ir::VerifyStage::HighLevel; }
  virtual PassResult run(ir::Kernel& kernel, const PassContext& ctx) = 0;
};

}  // namespace m1triton::passes
