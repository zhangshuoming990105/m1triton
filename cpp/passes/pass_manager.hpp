#pragma once

#include <memory>
#include <vector>

#include "passes/pass.hpp"

namespace m1triton::passes {

class PassManager {
 public:
  template <typename T, typename... Args>
  void add_pass(Args&&... args) {
    passes_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  PassResult run(ir::Kernel& kernel, const PassContext& ctx) const;
  PassResult run(ir::Module& mod, const PassContext& ctx) const;

  static PassManager build_m1_pipeline();

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace m1triton::passes
