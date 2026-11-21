#include <cassert>
#include <iostream>
#include <string>

#include "codegen/metal/row_scan_emitter.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/pass_manager.hpp"

namespace {

using namespace m1triton;

void assert_contains(const std::string& haystack, const std::string& needle) {
  if (haystack.find(needle) == std::string::npos) {
    std::cerr << "Expected MSL to contain:\n  " << needle << "\nActual MSL:\n" << haystack << "\n";
    assert(false);
  }
}

void assert_absent(const std::string& haystack, const std::string& needle) {
  if (haystack.find(needle) != std::string::npos) {
    std::cerr << "Expected MSL to omit:\n  " << needle << "\n";
    assert(false);
  }
}

codegen::metal::CompiledMSL emit_scan(ir::ScanAttr attr, bool with_mask, bool run_pipeline = true) {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("row_scan_test");
  ir::IRBuilder b(k);
  auto x = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 1, true);
  ir::ValueId mask = ir::kInvalidId;
  if (with_mask) mask = b.add_param("mask_ptr", ir::Type::ptr(ir::ScalarDType::Bool));
  auto rows = b.add_param("num_rows", ir::Type::scalar(ir::ScalarDType::I32));
  auto len = b.add_param("row_len", ir::Type::scalar(ir::ScalarDType::I32));
  b.emit_row_scan(x, out, rows, len, attr, mask);
  b.emit_return();

  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  if (run_pipeline) {
    auto pm = passes::PassManager::build_m1_pipeline();
    passes::PassContext ctx{target};
    pm.run(k, ctx);
    auto vr2 = ir::IRVerifier::verify(k, target);
    assert(vr2.ok);
  }

  bool matched = false;
  auto res = codegen::metal::try_emit_row_scan_kernel(k, target, &matched);
  assert(matched);
  if (!res.ok) std::cerr << res.error << "\n";
  assert(res.ok);
  return res;
}

void test_non_scan_returns_unmatched() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("vector_add");
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();
  bool matched = true;
  auto res = codegen::metal::try_emit_row_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched);
  assert(!res.ok);
  assert(res.msl_source.empty());
  assert(res.error.empty());
}

void test_cumsum_inclusive() {
  ir::ScanAttr attr;
  auto res = emit_scan(attr, false);
  const auto& msl = res.msl_source;
  assert_contains(msl, "#define M1_OP(a, b) ((a) + (b))");
  assert_contains(msl, "#define M1_ID 0.0f");
  assert_contains(msl, "simd_prefix_exclusive_sum(s3)");
  assert_absent(msl, "threadgroup_barrier");
  assert_contains(msl, "#define M1_ROWS_PER_TG 8");
  assert_contains(msl, "*(device const float4*)(src + p)");
  assert_contains(msl, "const long base = row * n;");
  assert_contains(msl, "device const float* x_ptr [[buffer(0)]]");
  assert_contains(msl, "device float* out_ptr [[buffer(1)]]");
  assert_contains(msl, "constant int& num_rows [[buffer(2)]]");
  assert_contains(msl, "#define M1_REV 0");
  assert_contains(msl, "#define M1_EXCL 0");
  assert_contains(msl, "#define M1_MASK 0");
  assert_absent(msl, "simd_shuffle_up(pp");
  assert(res.launch_spec.tg_x == 256);
  assert(res.launch_spec.tg_y == 1);
  assert(res.launch_spec.param_slots.size() == 4);
  assert(res.launch_spec.param_slots[1].writable);
  assert(!res.launch_spec.param_slots[0].writable);
}

void test_cumprod_uses_shuffle_butterfly() {
  ir::ScanAttr attr;
  attr.kind = ir::ScanKind::Cumprod;
  auto res = emit_scan(attr, false);
  assert_contains(res.msl_source, "#define M1_OP(a, b) ((a) * (b))");
  assert_contains(res.msl_source, "#define M1_ID 1.0f");
  assert_contains(res.msl_source, "simd_shuffle_up(pp, d)");
  assert_absent(res.msl_source, "simd_prefix_exclusive_sum");
}

void test_reverse_and_exclusive_flags() {
  ir::ScanAttr rev;
  rev.reverse = true;
  assert_contains(emit_scan(rev, false).msl_source, "#define M1_REV 1");
  ir::ScanAttr excl;
  excl.exclusive = true;
  assert_contains(emit_scan(excl, false).msl_source, "#define M1_EXCL 1");
}

void test_masked_cumsum_reads_uchar_mask() {
  ir::ScanAttr attr;
  auto res = emit_scan(attr, true);
  const auto& msl = res.msl_source;
  assert_contains(msl, "#define M1_MASK 1");
  assert_contains(msl, "device const bool* mask_ptr [[buffer(2)]]");
  assert_contains(msl, "(device const uchar*)mask_ptr + base");
  assert_contains(msl, "float4(*(device const uchar4*)(msk + p))");
  assert(res.launch_spec.param_slots.size() == 5);
  assert(res.launch_spec.param_slots[2].dtype == ir::ScalarDType::Bool);
}

}  // namespace

int main() {
  test_non_scan_returns_unmatched();
  test_cumsum_inclusive();
  test_cumprod_uses_shuffle_butterfly();
  test_reverse_and_exclusive_flags();
  test_masked_cumsum_reads_uchar_mask();
  std::cout << "PASS test_row_scan_emitter\n";
  return 0;
}
