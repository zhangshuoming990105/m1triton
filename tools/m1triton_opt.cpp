#include <fstream>
#include <iostream>
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
#include "passes/cse.hpp"
#include "passes/dce.hpp"
#include "passes/licm.hpp"
#include "passes/pass_manager.hpp"
#include "passes/tile_layout.hpp"

namespace {

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

bool add_named_pass(m1triton::passes::PassManager& pm, const std::string& name) {
  using namespace m1triton::passes;
  if (name == "canonicalize") {
    pm.add_pass<CanonicalizePass>();
  } else if (name == "cse") {
    pm.add_pass<CSEPass>();
  } else if (name == "licm") {
    pm.add_pass<LICMPass>();
  } else if (name == "contiguity") {
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
  } else if (name == "tile-layout") {
    pm.add_pass<TileAndLayoutAssignmentPass>();
  } else if (name == "dce") {
    pm.add_pass<DCEPass>();
  } else {
    return false;
  }
  return true;
}

void print_usage() {
  std::cerr << "Usage: m1triton-opt [input.ir|-] [--passes=m1|canonicalize,cse,...] "
               "[--dump-after-each] [--emit-msl] [--no-verify]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path = "-";
  std::string passes_arg = "";
  bool dump_after_each = false;
  bool emit_msl = false;
  bool verify = true;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      return 0;
    } else if (arg.rfind("--passes=", 0) == 0) {
      passes_arg = arg.substr(9);
    } else if (arg == "--dump-after-each") {
      dump_after_each = true;
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
    m1triton::passes::PassManager pm;
    if (passes_arg == "m1") {
      pm = m1triton::passes::PassManager::build_m1_pipeline();
    } else {
      for (const auto& name : split_csv(passes_arg)) {
        if (!add_named_pass(pm, name)) {
          std::cerr << "Unknown pass: " << name << "\n";
          return 2;
        }
      }
    }
    m1triton::passes::PassContext ctx{target, verify, dump_after_each, &std::cerr};
    auto pr = pm.run(*parsed.module, ctx);
    if (!pr.ok) {
      std::cerr << "Pass pipeline failed: " << pr.diagnostic << "\n";
      return 1;
    }
  }

  if (emit_msl) {
    for (const auto& k : parsed.module->kernels) {
      auto compiled = m1triton::codegen::metal::MetalCodegen::emit(*k, target);
      if (!compiled.ok) {
        std::cerr << "MetalCodegen failed for @" << k->name << ": " << compiled.error << "\n";
        return 1;
      }
      std::cout << "// LaunchSpec: tg=(" << compiled.launch_spec.tg_x << "," << compiled.launch_spec.tg_y << ","
                << compiled.launch_spec.tg_z << ") vec=" << compiled.launch_spec.vec_width
                << " smem=" << compiled.launch_spec.threadgroup_mem_bytes << "\n";
      std::cout << compiled.msl_source << "\n";
    }
  } else {
    std::cout << m1triton::ir::IRPrinter::print(*parsed.module);
  }
  return 0;
}
