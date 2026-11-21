#include "passes/epilogue_fusion.hpp"

namespace m1triton::passes {

PassResult EpilogueFusionPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  (void)kernel;
  (void)ctx;
  PassResult res;
  res.ok = true;
  res.changed = false;
  return res;
}

}  // namespace m1triton::passes
