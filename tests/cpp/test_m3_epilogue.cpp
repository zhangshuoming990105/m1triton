#include <cassert>
#include <iostream>
#include <memory>
#include <string>

#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/epilogue_fusion.hpp"
#include "passes/pass_manager.hpp"

using namespace m1triton::ir;
using namespace m1triton::passes;

namespace {

constexpr const char* kHeader =
    "kernel @act(%x_ptr: ptr<f32, device> {divisibility=16}, %o_ptr: ptr<f32, device> {divisibility=16, "
    "writable=true}) {\n";

std::unique_ptr<Module> parse_or_die(const std::string& text) {
  auto parsed = IRParser::parse(text);
  if (!parsed.ok || !parsed.module) {
    std::cerr << "IR parse failed: " << parsed.error << "\n--- IR ---\n" << text << "\n";
    assert(false);
  }
  return std::move(parsed.module);
}

void assert_ir_diff(const std::string& src_ir, const std::string& expected_dst_ir, bool expect_changed = true) {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<EpilogueFusionPass>();
  auto mod = parse_or_die(src_ir);
  PassContext ctx{target, true, false, nullptr};

  const auto pr = pm.run(*mod, ctx);
  if (!pr.ok) {
    std::cerr << "PassManager failed: " << pr.diagnostic << "\n--- SRC IR ---\n" << src_ir << "\n";
    assert(false);
  }
  assert(pr.changed == expect_changed);
  const std::string actual = IRPrinter::print(*mod);
  if (actual != expected_dst_ir) {
    std::cerr << "IR diff mismatch!\n--- EXPECTED ---\n" << expected_dst_ir << "--- ACTUAL ---\n" << actual << "\n";
    assert(false);
  }

  const auto vr = IRVerifier::verify(*mod, target, VerifyStage::HighLevel);
  if (!vr.ok) {
    std::cerr << "Verifier failed:\n" << vr.summary() << "\n--- IR ---\n" << actual << "\n";
    assert(false);
  }

  const auto reparsed = parse_or_die(actual);
  assert(IRPrinter::print(*reparsed) == actual);

  const auto pr2 = pm.run(*mod, ctx);
  assert(pr2.ok && !pr2.changed);
  assert(IRPrinter::print(*mod) == expected_dst_ir);
}

const std::string kPrologue = std::string(kHeader) + R"(  %0 = arange {start=0, end=8} : tile<8, i32>
  %1 = splat %x_ptr : tile<8, ptr<f32, device>>
  %2 = ptr_add %1, %0 : tile<8, ptr<f32, device>>
  %3 = load %2 {align=4, vec=1} : tile<8, f32>
)";

const std::string kPrologueVec = std::string(kHeader) + R"(  %0 = arange {start=0, end=8} : tile<8, i32>
  %1 = splat %x_ptr : tile<8, ptr<f32, device>>
  %2 = ptr_add %1, %0 : tile<8, ptr<f32, device>>
  %3 = load %2 {align=4, vec=1} hints {vec_width=4} : tile<8, f32>
)";

std::string add_vec_hints(const std::string& ir) {
  std::string out;
  size_t pos = 0;
  while (pos < ir.size()) {
    size_t eol = ir.find('\n', pos);
    if (eol == std::string::npos) eol = ir.size();
    std::string line = ir.substr(pos, eol - pos);
    const bool is_store = line.rfind("  store ", 0) == 0;
    const bool is_load = line.find(" = load ") != std::string::npos;
    if (is_store) {
      line += " hints {vec_width=4}";
    } else if (is_load) {
      line.insert(line.find(" : tile<"), " hints {vec_width=4}");
    }
    out += line;
    if (eol < ir.size()) out += '\n';
    pos = eol + 1;
  }
  return out;
}

void test_pow_to_mul() {
  const std::string src_cube = kPrologue + R"(  %4 = const {val=3.0} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.pow %3, %5 : tile<8, f32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %0 : tile<8, ptr<f32, device>>
  store %8, %6 {align=4, vec=1}
  return
}
)";
  const std::string dst_cube = kPrologueVec + R"(  %4 = binary.mul %3, %3 : tile<8, f32>
  %5 = binary.mul %4, %3 : tile<8, f32>
  %6 = splat %o_ptr : tile<8, ptr<f32, device>>
  %7 = ptr_add %6, %0 : tile<8, ptr<f32, device>>
  store %7, %5 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
  assert_ir_diff(src_cube, dst_cube);

  const std::string src_square = kPrologue + R"(  %4 = const {val=2.0} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.pow %3, %5 : tile<8, f32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %0 : tile<8, ptr<f32, device>>
  store %8, %6 {align=4, vec=1}
  return
}
)";
  const std::string dst_square = kPrologueVec + R"(  %4 = binary.mul %3, %3 : tile<8, f32>
  %5 = splat %o_ptr : tile<8, ptr<f32, device>>
  %6 = ptr_add %5, %0 : tile<8, ptr<f32, device>>
  store %6, %4 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
  assert_ir_diff(src_square, dst_square);

  const std::string src_frac = kPrologue + R"(  %4 = const {val=0.5} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.pow %3, %5 : tile<8, f32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %0 : tile<8, ptr<f32, device>>
  store %8, %6 {align=4, vec=1}
  return
}
)";
  const std::string dst_frac = kPrologueVec + R"(  %4 = const {val=0.5} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.pow %3, %5 : tile<8, f32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %0 : tile<8, ptr<f32, device>>
  store %8, %6 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
  assert_ir_diff(src_frac, dst_frac);
  std::cout << "[PASS] test_pow_to_mul\n";
}

void test_sigmoid_fusion() {
  const std::string src = kPrologue + R"(  %4 = unary.neg %3 : tile<8, f32>
  %5 = unary.exp %4 : tile<8, f32>
  %6 = const {val=1.0} : f32
  %7 = splat %6 : tile<8, f32>
  %8 = binary.add %7, %5 : tile<8, f32>
  %9 = binary.div %7, %8 : tile<8, f32>
  %10 = splat %o_ptr : tile<8, ptr<f32, device>>
  %11 = ptr_add %10, %0 : tile<8, ptr<f32, device>>
  store %11, %9 {align=4, vec=1}
  return
}
)";
  const std::string dst = kPrologueVec + R"(  %4 = unary.sigmoid %3 : tile<8, f32>
  %5 = splat %o_ptr : tile<8, ptr<f32, device>>
  %6 = ptr_add %5, %0 : tile<8, ptr<f32, device>>
  store %6, %4 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
  assert_ir_diff(src, dst);

  const std::string src_not_sigmoid = kPrologue + R"(  %4 = unary.exp %3 : tile<8, f32>
  %5 = const {val=1.0} : f32
  %6 = splat %5 : tile<8, f32>
  %7 = binary.add %6, %4 : tile<8, f32>
  %8 = binary.div %6, %7 : tile<8, f32>
  %9 = splat %o_ptr : tile<8, ptr<f32, device>>
  %10 = ptr_add %9, %0 : tile<8, ptr<f32, device>>
  store %10, %8 {align=4, vec=1}
  return
}
)";
  const std::string dst_not_sigmoid = kPrologueVec + R"(  %4 = unary.exp %3 : tile<8, f32>
  %5 = const {val=1.0} : f32
  %6 = splat %5 : tile<8, f32>
  %7 = binary.add %6, %4 : tile<8, f32>
  %8 = binary.div %6, %7 : tile<8, f32>
  %9 = splat %o_ptr : tile<8, ptr<f32, device>>
  %10 = ptr_add %9, %0 : tile<8, ptr<f32, device>>
  store %10, %8 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
  assert_ir_diff(src_not_sigmoid, dst_not_sigmoid);
  std::cout << "[PASS] test_sigmoid_fusion\n";
}

void test_relu_fusion() {
  for (const char* cmp : {"cmp_gt", "cmp_ge"}) {
    const std::string c = cmp;
    const std::string src = kPrologue + R"(  %4 = const {val=0.0} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.)" + c + R"( %3, %5 : tile<8, bool>
  %7 = select %6, %3, %5 : tile<8, f32>
  %8 = splat %o_ptr : tile<8, ptr<f32, device>>
  %9 = ptr_add %8, %0 : tile<8, ptr<f32, device>>
  store %9, %7 {align=4, vec=1}
  return
}
)";
    const std::string dst = kPrologueVec + R"(  %4 = const {val=0.0} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = binary.max %3, %5 : tile<8, f32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %0 : tile<8, ptr<f32, device>>
  store %8, %6 {align=4, vec=1} hints {vec_width=4}
  return
}
)";
    assert_ir_diff(src, dst);
  }

  const std::string src_leaky = kPrologue + R"(  %4 = const {val=0.0} : f32
  %5 = splat %4 : tile<8, f32>
  %6 = const {val=0.01} : f32
  %7 = splat %6 : tile<8, f32>
  %8 = binary.mul %3, %7 : tile<8, f32>
  %9 = binary.cmp_gt %3, %5 : tile<8, bool>
  %10 = select %9, %3, %8 : tile<8, f32>
  %11 = splat %o_ptr : tile<8, ptr<f32, device>>
  %12 = ptr_add %11, %0 : tile<8, ptr<f32, device>>
  store %12, %10 {align=4, vec=1}
  return
}
)";
  const std::string dst_leaky = add_vec_hints(src_leaky);
  assert_ir_diff(src_leaky, dst_leaky);
  std::cout << "[PASS] test_relu_fusion\n";
}

void test_vec_width_promotion() {
  const std::string src = std::string(kHeader).replace(std::string(kHeader).find(") {\n"), 4, ", %n: i32) {\n") + R"(  %0 = program_id {axis=0} : i32
  %1 = const {val=1024} : i32
  %2 = binary.mul %0, %1 : i32
  %3 = splat %2 : tile<1024, i32>
  %4 = arange {start=0, end=1024} : tile<1024, i32>
  %5 = binary.add %3, %4 : tile<1024, i32>
  %6 = splat %n : tile<1024, i32>
  %7 = binary.cmp_lt %5, %6 : tile<1024, bool>
  %8 = splat %x_ptr : tile<1024, ptr<f32, device>>
  %9 = ptr_add %8, %5 : tile<1024, ptr<f32, device>>
  %10 = const {val=0.0} : f32
  %11 = splat %10 : tile<1024, f32>
  %12 = load %9, %7, %11 {align=4, vec=1, has_mask=true, has_other=true} : tile<1024, f32>
  %13 = splat %o_ptr : tile<1024, ptr<f32, device>>
  %14 = ptr_add %13, %5 : tile<1024, ptr<f32, device>>
  store %14, %12, %7 {align=4, vec=1, has_mask=true}
  return
}
)";
  const std::string dst = add_vec_hints(src);
  assert_ir_diff(src, dst);

  const std::string src_strided = kPrologue + R"(  %4 = const {val=2} : i32
  %5 = splat %4 : tile<8, i32>
  %6 = binary.mul %0, %5 : tile<8, i32>
  %7 = splat %o_ptr : tile<8, ptr<f32, device>>
  %8 = ptr_add %7, %6 : tile<8, ptr<f32, device>>
  store %8, %3 {align=4, vec=1}
  return
}
)";
  assert_ir_diff(src_strided, src_strided, false);

  const std::string src_odd = std::string(kHeader) + R"(  %0 = arange {start=0, end=6} : tile<6, i32>
  %1 = splat %x_ptr : tile<6, ptr<f32, device>>
  %2 = ptr_add %1, %0 : tile<6, ptr<f32, device>>
  %3 = load %2 {align=4, vec=1} : tile<6, f32>
  %4 = splat %o_ptr : tile<6, ptr<f32, device>>
  %5 = ptr_add %4, %0 : tile<6, ptr<f32, device>>
  store %5, %3 {align=4, vec=1}
  return
}
)";
  assert_ir_diff(src_odd, src_odd, false);

  // Constexpr parameter preceding an unaligned pointer must NOT mis-index divisibility.
  const std::string src_ce_unaligned =
      "kernel @act_ce(%B: i32 {constexpr=8}, %x_ptr: ptr<f32, device>, "
      "%o_ptr: ptr<f32, device> {divisibility=16, writable=true}) {\n"
      "  %0 = arange {start=0, end=8} : tile<8, i32>\n"
      "  %1 = splat %x_ptr : tile<8, ptr<f32, device>>\n"
      "  %2 = ptr_add %1, %0 : tile<8, ptr<f32, device>>\n"
      "  %3 = load %2 {align=4, vec=1} : tile<8, f32>\n"
      "  %4 = splat %o_ptr : tile<8, ptr<f32, device>>\n"
      "  %5 = ptr_add %4, %0 : tile<8, ptr<f32, device>>\n"
      "  store %5, %3 {align=4, vec=1}\n"
      "  return\n"
      "}\n";
  assert_ir_diff(src_ce_unaligned, src_ce_unaligned, false);
  std::cout << "[PASS] test_vec_width_promotion\n";
}

}  // namespace

int main() {
  test_pow_to_mul();
  test_sigmoid_fusion();
  test_relu_fusion();
  test_vec_width_promotion();
  std::cout << "PASS\n";
  return 0;
}
