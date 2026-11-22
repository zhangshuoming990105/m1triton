#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "codegen/metal/metal_codegen.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/canonicalize.hpp"
#include "passes/contiguity_analysis.hpp"
#include "passes/conv_lowering.hpp"
#include "passes/cse.hpp"
#include "passes/dce.hpp"
#include "passes/epilogue_fusion.hpp"
#include "passes/flash_attn_lowering.hpp"
#include "passes/licm.hpp"
#include "passes/loss_lowering.hpp"
#include "passes/online_softmax_welford.hpp"
#include "passes/pass.hpp"
#include "passes/pass_manager.hpp"
#include "passes/pool_lowering.hpp"
#include "passes/simdgroup_matmul.hpp"
#include "passes/structure_pruning.hpp"
#include "passes/threadgroup_staging.hpp"
#include "passes/tile_layout.hpp"

namespace {

using PassList = std::vector<std::unique_ptr<m1triton::passes::Pass>>;

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

bool append_named_pass(PassList& passes, const std::string& name) {
  using namespace m1triton::passes;
  if (name == "canonicalize") {
    passes.push_back(std::make_unique<CanonicalizePass>());
  } else if (name == "cse") {
    passes.push_back(std::make_unique<CSEPass>());
  } else if (name == "licm") {
    passes.push_back(std::make_unique<LICMPass>());
  } else if (name == "epilogue-fusion") {
    passes.push_back(std::make_unique<EpilogueFusionPass>());
  } else if (name == "structure-pruning") {
    passes.push_back(std::make_unique<StructureAwareDeadTilePruningPass>());
  } else if (name == "online-softmax-welford") {
    passes.push_back(std::make_unique<OnlineSoftmaxWelfordPass>());
  } else if (name == "pool-lowering") {
    passes.push_back(std::make_unique<PoolLoweringPass>());
  } else if (name == "conv-lowering") {
    passes.push_back(std::make_unique<ConvLoweringPass>());
  } else if (name == "loss-lowering") {
    passes.push_back(std::make_unique<LossLoweringPass>());
  } else if (name == "flash-attn-lowering") {
    passes.push_back(std::make_unique<FlashAttentionLoweringPass>());
  } else if (name == "contiguity") {
    passes.push_back(std::make_unique<ContiguityAndAlignmentAnalysis>());
  } else if (name == "tile-layout") {
    passes.push_back(std::make_unique<TileAndLayoutAssignmentPass>());
  } else if (name == "threadgroup-staging") {
    passes.push_back(std::make_unique<ThreadgroupStagingAndSwizzlePass>());
  } else if (name == "simdgroup-matmul") {
    passes.push_back(std::make_unique<SimdgroupMatrixLoweringPass>());
  } else if (name == "dce") {
    passes.push_back(std::make_unique<DCEPass>());
  } else {
    return false;
  }
  return true;
}

PassList build_standard_pass_list(m1triton::passes::SgmmSchedulePolicy sgmm_policy) {
  using namespace m1triton::passes;
  PassList passes;
  passes.push_back(std::make_unique<CanonicalizePass>());
  passes.push_back(std::make_unique<CSEPass>());
  passes.push_back(std::make_unique<LICMPass>());
  passes.push_back(std::make_unique<EpilogueFusionPass>());
  passes.push_back(std::make_unique<StructureAwareDeadTilePruningPass>());
  passes.push_back(std::make_unique<OnlineSoftmaxWelfordPass>());
  passes.push_back(std::make_unique<PoolLoweringPass>());
  passes.push_back(std::make_unique<ConvLoweringPass>());
  passes.push_back(std::make_unique<LossLoweringPass>());
  passes.push_back(std::make_unique<FlashAttentionLoweringPass>());
  passes.push_back(std::make_unique<ContiguityAndAlignmentAnalysis>());
  passes.push_back(std::make_unique<TileAndLayoutAssignmentPass>(sgmm_policy));
  passes.push_back(std::make_unique<ThreadgroupStagingAndSwizzlePass>());
  passes.push_back(std::make_unique<SimdgroupMatrixLoweringPass>());
  passes.push_back(std::make_unique<DCEPass>());
  return passes;
}

size_t count_active_ops(const m1triton::ir::Kernel& k) {
  size_t count = 0;
  for (const auto& op : k.ops) {
    if (op && !op->erased) {
      ++count;
    }
  }
  return count;
}

void print_kernel_stats(
    const m1triton::ir::Kernel& k,
    const m1triton::codegen::metal::LaunchSpec* launch_spec) {
  std::cerr << "[stats] kernel=@" << k.name
            << " ops=" << count_active_ops(k)
            << " blocks=" << k.blocks.size()
            << " params=" << k.params.size();
  if (launch_spec != nullptr) {
    std::cerr << " tg=(" << launch_spec->tg_x << "," << launch_spec->tg_y << "," << launch_spec->tg_z << ")"
              << " smem=" << launch_spec->threadgroup_mem_bytes
              << " vec=" << launch_spec->vec_width;
  }
  std::cerr << "\n";
}

m1triton::passes::PassResult run_passes_with_hooks(
    m1triton::ir::Module& mod,
    const PassList& passes,
    const m1triton::passes::PassContext& ctx,
    bool dump_before_all,
    bool dump_after_all) {
  m1triton::passes::PassResult overall;
  for (auto& kernel : mod.kernels) {
    if (ctx.verify_each_pass) {
      const auto vr =
          m1triton::ir::IRVerifier::verify(*kernel, ctx.target, m1triton::ir::VerifyStage::HighLevel);
      if (!vr.ok) {
        overall.ok = false;
        overall.diagnostic = "kernel @" + kernel->name + ": Pre-pass IR verification failed:\n" + vr.summary();
        return overall;
      }
    }
    for (const auto& pass : passes) {
      if (dump_before_all) {
        std::cerr << "// --- Before pass: " << pass->name() << " ---\n"
                  << m1triton::ir::IRPrinter::print(*kernel) << "\n";
      }
      auto pr = pass->run(*kernel, ctx);
      overall.changed = overall.changed || pr.changed;
      if (!pr.ok) {
        overall.ok = false;
        overall.diagnostic =
            "kernel @" + kernel->name + ": " + std::string(pass->name()) + ": " + pr.diagnostic;
        return overall;
      }
      if (dump_after_all) {
        std::cerr << "// --- After pass: " << pass->name() << " ---\n"
                  << m1triton::ir::IRPrinter::print(*kernel) << "\n";
      }
      if (ctx.verify_each_pass) {
        const auto vr = m1triton::ir::IRVerifier::verify(*kernel, ctx.target, pass->output_stage());
        if (!vr.ok) {
          overall.ok = false;
          overall.diagnostic = "kernel @" + kernel->name + ": IR verification failed after pass '" +
                               std::string(pass->name()) + "':\n" + vr.summary();
          return overall;
        }
      }
    }
  }
  return overall;
}

void print_usage() {
  std::cerr << "Usage: m1triton-opt [input.ir|-] "
               "[--passes=m1|m2|canonicalize,cse,licm,epilogue-fusion,structure-pruning,"
               "online-softmax-welford,pool-lowering,conv-lowering,loss-lowering,flash-attn-lowering,"
               "contiguity,tile-layout,threadgroup-staging,simdgroup-matmul,dce] "
               "[--dump-ir-before-all] [--dump-ir-after-all|--dump-after-each] [--stats] "
               "[--emit-msl] [--no-verify]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path = "-";
  std::string passes_arg = "";
  bool dump_before_all = false;
  bool dump_after_all = false;
  bool show_stats = false;
  bool emit_msl = false;
  bool verify = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      return 0;
    } else if (arg.rfind("--passes=", 0) == 0) {
      passes_arg = arg.substr(9);
    } else if (arg == "--dump-ir-before-all") {
      dump_before_all = true;
    } else if (arg == "--dump-after-each" || arg == "--dump-ir-after-all") {
      dump_after_all = true;
    } else if (arg == "--stats") {
      show_stats = true;
    } else if (arg == "--emit-msl") {
      emit_msl = true;
    } else if (arg == "--no-verify") {
      verify = false;
    } else if (arg.rfind("--", 0) == 0) {
      std::cerr << "Unknown option: " << arg << "\n";
      return 2;
    } else {
      input_path = arg;
    }
  }

  std::string text;
  if (input_path == "-") {
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    text = buf.str();
  } else {
    std::ifstream ifs(input_path);
    if (!ifs) {
      std::cerr << "Failed to open input file: " << input_path << "\n";
      return 1;
    }
    std::ostringstream buf;
    buf << ifs.rdbuf();
    text = buf.str();
  }

  auto parsed = m1triton::ir::IRParser::parse(text);
  if (!parsed.ok || !parsed.module) {
    std::cerr << "IR parse error: " << parsed.error << "\n";
    return 1;
  }

  const auto target = m1triton::ir::TargetSpec::apple_m1_max();
  if (verify) {
    auto vr = m1triton::ir::IRVerifier::verify(*parsed.module, target, m1triton::ir::VerifyStage::HighLevel);
    if (!vr.ok) {
      std::cerr << "Input IR verification failed:\n" << vr.summary() << "\n";
      return 1;
    }
  }

  if (!passes_arg.empty()) {
    PassList passes;
    if (passes_arg == "m1") {
      passes = build_standard_pass_list(m1triton::passes::SgmmSchedulePolicy::HintGuided);
    } else if (passes_arg == "m2") {
      passes = build_standard_pass_list(m1triton::passes::SgmmSchedulePolicy::Auto);
    } else {
      for (const auto& name : split_csv(passes_arg)) {
        if (!append_named_pass(passes, name)) {
          std::cerr << "Unknown pass: " << name << "\n";
          return 2;
        }
      }
    }
    m1triton::passes::PassContext ctx{target, verify, false, &std::cerr};
    auto pr = run_passes_with_hooks(*parsed.module, passes, ctx, dump_before_all, dump_after_all);
    if (!pr.ok) {
      std::cerr << "Pass pipeline failed: " << pr.diagnostic << "\n";
      return 1;
    }
  } else if (dump_before_all) {
    for (const auto& k : parsed.module->kernels) {
      std::cerr << "// --- Before all passes ---\n" << m1triton::ir::IRPrinter::print(*k) << "\n";
    }
  }

  if (emit_msl) {
    for (const auto& k : parsed.module->kernels) {
      auto compiled = m1triton::codegen::metal::MetalCodegen::emit(*k, target);
      if (!compiled.ok) {
        std::cerr << "MetalCodegen failed for @" << k->name << ": " << compiled.error << "\n";
        return 1;
      }
      if (show_stats) {
        print_kernel_stats(*k, &compiled.launch_spec);
      }
      std::cout << "// LaunchSpec: tg=(" << compiled.launch_spec.tg_x << "," << compiled.launch_spec.tg_y << ","
                << compiled.launch_spec.tg_z << ") vec=" << compiled.launch_spec.vec_width
                << " smem=" << compiled.launch_spec.threadgroup_mem_bytes << "\n";
      std::cout << compiled.msl_source << "\n";
    }
  } else {
    if (show_stats) {
      for (const auto& k : parsed.module->kernels) {
        print_kernel_stats(*k, nullptr);
      }
    }
    std::cout << m1triton::ir::IRPrinter::print(*parsed.module);
  }
  return 0;
}
