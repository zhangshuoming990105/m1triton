#include "passes/pass_manager.hpp"

#include <string>

#include "ir/printer.hpp"
#include "passes/canonicalize.hpp"
#include "passes/contiguity_analysis.hpp"
#include "passes/cse.hpp"
#include "passes/dce.hpp"
#include "passes/epilogue_fusion.hpp"
#include "passes/licm.hpp"
#include "passes/online_softmax_welford.hpp"
#include "passes/pool_lowering.hpp"
#include "passes/simdgroup_matmul.hpp"
#include "passes/structure_pruning.hpp"
#include "passes/threadgroup_staging.hpp"
#include "passes/tile_layout.hpp"

namespace m1triton::passes {

namespace {

PassManager build_pipeline(SgmmSchedulePolicy sgmm_policy) {
  PassManager pm;
  pm.add_pass<CanonicalizePass>();
  pm.add_pass<CSEPass>();
  pm.add_pass<LICMPass>();
  pm.add_pass<EpilogueFusionPass>();
  pm.add_pass<StructureAwareDeadTilePruningPass>();
  pm.add_pass<OnlineSoftmaxWelfordPass>();
  pm.add_pass<PoolLoweringPass>();
  pm.add_pass<ContiguityAndAlignmentAnalysis>();
  pm.add_pass<TileAndLayoutAssignmentPass>(sgmm_policy);
  pm.add_pass<ThreadgroupStagingAndSwizzlePass>();
  pm.add_pass<SimdgroupMatrixLoweringPass>();
  pm.add_pass<DCEPass>();
  return pm;
}

}  // namespace

PassResult PassManager::run(ir::Kernel& kernel, const PassContext& ctx) const {
  PassResult overall;
  if (ctx.verify_each_pass) {
    const auto vr = ir::IRVerifier::verify(kernel, ctx.target, ir::VerifyStage::HighLevel);
    if (!vr.ok) {
      overall.ok = false;
      overall.diagnostic = "Pre-pass IR verification failed:\n" + vr.summary();
      return overall;
    }
  }

  for (const auto& pass : passes_) {
    PassResult pr = pass->run(kernel, ctx);
    overall.changed = overall.changed || pr.changed;
    if (!pr.ok) {
      overall.ok = false;
      overall.diagnostic = std::string(pass->name()) + ": " + pr.diagnostic;
      return overall;
    }

    if (ctx.dump_ir && ctx.dump_stream != nullptr) {
      *ctx.dump_stream << "// --- After pass: " << pass->name() << " ---\n"
                       << ir::IRPrinter::print(kernel) << "\n";
    }

    if (ctx.verify_each_pass) {
      const auto vr = ir::IRVerifier::verify(kernel, ctx.target, pass->output_stage());
      if (!vr.ok) {
        overall.ok = false;
        overall.diagnostic =
            "IR verification failed after pass '" + std::string(pass->name()) + "':\n" + vr.summary();
        return overall;
      }
    }
  }

  return overall;
}

PassResult PassManager::run(ir::Module& mod, const PassContext& ctx) const {
  PassResult overall;
  for (auto& kernel : mod.kernels) {
    PassResult kr = run(*kernel, ctx);
    overall.changed = overall.changed || kr.changed;
    if (!kr.ok) {
      overall.ok = false;
      overall.diagnostic = "kernel @" + kernel->name + ": " + kr.diagnostic;
      return overall;
    }
  }
  return overall;
}

PassManager PassManager::build_m1_pipeline() {
  return build_pipeline(SgmmSchedulePolicy::HintGuided);
}

PassManager PassManager::build_m2_pipeline() {
  return build_pipeline(SgmmSchedulePolicy::Auto);
}

}  // namespace m1triton::passes
