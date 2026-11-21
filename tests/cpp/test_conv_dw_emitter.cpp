#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "codegen/metal/conv_dw_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
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

void assert_not_contains(const std::string& haystack, const std::string& needle) {
  if (haystack.find(needle) != std::string::npos) {
    std::cerr << "Expected MSL to NOT contain:\n  " << needle << "\nActual MSL:\n" << haystack << "\n";
    assert(false);
  }
}

struct DwSpec {
  std::vector<int32_t> k{3, 3};
  std::vector<int32_t> s{1, 1};
  std::vector<int32_t> p{0, 0};
  std::vector<int32_t> d{1, 1};
  bool has_bias = false;
  bool separate_cout = false;
  int64_t block_w = 0;
  int64_t block_h = 0;
  bool transposed = false;
  bool depthwise = true;
  ir::ScalarDType dtype = ir::ScalarDType::F32;
};

void build_dw_kernel(ir::Kernel& k, const DwSpec& spec) {
  ir::IRBuilder b(k);
  auto x = b.add_param("x_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto w = b.add_param("w_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto o = b.add_param("out_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, true);
  std::vector<ir::ValueId> operands{x, w, o};
  if (spec.has_bias) operands.push_back(b.add_param("bias_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false));
  operands.push_back(b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32)));
  operands.push_back(b.add_param("C_in", ir::Type::scalar(ir::ScalarDType::I32)));
  if (spec.separate_cout) operands.push_back(b.add_param("C_out", ir::Type::scalar(ir::ScalarDType::I32)));
  for (const char* name : {"H_in", "W_in", "H_out", "W_out"}) {
    operands.push_back(b.add_param(name, ir::Type::scalar(ir::ScalarDType::I32)));
  }
  if (spec.block_w > 0) b.add_param("BLOCK_W", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_w);
  if (spec.block_h > 0) b.add_param("BLOCK_H", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_h);
  ir::ConvAttr attr;
  attr.ndim = 2;
  attr.kernel_size = spec.k;
  attr.stride = spec.s;
  attr.padding = spec.p;
  attr.dilation = spec.d;
  attr.has_bias = spec.has_bias;
  attr.is_depthwise = spec.depthwise;
  attr.transposed = spec.transposed;
  b.emit_implicit_gemm_conv(operands, attr);
  b.emit_return();
}

codegen::metal::CompiledMSL emit(const ir::Kernel& k) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  bool matched = false;
  auto res = codegen::metal::try_emit_conv_dw_kernel(k, target, &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error: " << res.error << "\n";
    assert(false);
  }
  return res;
}

void test_sram_halo_3x3() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw3x3");
  DwSpec spec;
  spec.block_w = 64;
  spec.block_h = 32;
  build_dw_kernel(k, spec);
  const auto res = emit(k);
  const std::string& msl = res.msl_source;
  assert_contains(msl, "kernel void dw3x3(");
  assert_contains(msl, "constexpr int m1_HW = 66;");
  assert_contains(msl, "constexpr int m1_HH = 34;");
  assert_contains(msl, "threadgroup float m1_smem[m1_HH * m1_HW];");
  assert_contains(msl, "threadgroup_barrier(mem_flags::mem_threadgroup);");
  assert_contains(msl, "float m1_wr[m1_KH * m1_KW];");
  assert_not_contains(msl, "m1_row[m1_iw]");
  assert(res.launch_spec.tg_x == 32 && res.launch_spec.tg_y == 8);
  assert(res.launch_spec.param_slots.size() == 9);
  assert(res.launch_spec.param_slots[2].writable && !res.launch_spec.param_slots[0].writable);
  assert(res.launch_spec.param_slots[3].name == "N" && res.launch_spec.param_slots[3].buffer_index == 3);
}

void test_default_tile_without_constexprs() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_default");
  build_dw_kernel(k, DwSpec{});
  const auto res = emit(k);
  assert_contains(res.msl_source, "constexpr int m1_BW = 64;");
  assert_contains(res.msl_source, "constexpr int m1_BH = 32;");
}

void test_asymmetric_kernel_attrs() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw3x7");
  DwSpec spec;
  spec.k = {3, 7};
  spec.s = {2, 1};
  spec.p = {1, 3};
  spec.d = {1, 2};
  spec.block_w = 32;
  spec.block_h = 8;
  build_dw_kernel(k, spec);
  const std::string msl = emit(k).msl_source;
  assert_contains(msl, "constexpr int m1_KH = 3, m1_KW = 7;");
  assert_contains(msl, "constexpr int m1_SH = 2, m1_SWD = 1;");
  assert_contains(msl, "constexpr int m1_PH = 1, m1_PW = 3;");
  assert_contains(msl, "constexpr int m1_DH = 1, m1_DW = 2;");
  assert_contains(msl, "constexpr int m1_HW = 44;");
  assert_contains(msl, "constexpr int m1_HH = 17;");
}

void test_bias_and_depth_multiplier() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_bias_mult");
  DwSpec spec;
  spec.has_bias = true;
  spec.separate_cout = true;
  build_dw_kernel(k, spec);
  const auto res = emit(k);
  assert_contains(res.msl_source, "device const float* bias_ptr [[buffer(3)]]");
  assert_contains(res.msl_source, "const float m1_bv = (float)bias_ptr[m1_co];");
  assert_contains(res.msl_source, "m1_Cout / m1_Cin");
  assert(res.launch_spec.param_slots.size() == 11);
}

void test_direct_fallback_for_large_halo() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_dilated");
  DwSpec spec;
  spec.k = {7, 7};
  spec.s = {4, 4};
  spec.d = {8, 8};
  spec.block_w = 64;
  spec.block_h = 32;
  build_dw_kernel(k, spec);
  const std::string msl = emit(k).msl_source;
  assert_not_contains(msl, "m1_smem");
  assert_not_contains(msl, "threadgroup_barrier");
  assert_contains(msl, "m1_row[m1_iw]");
  assert_contains(msl, "const long m1_iw = m1_iw0 + m1_kw * m1_DW;");
}

void test_f16_halo_budget_and_types() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_half");
  DwSpec spec;
  spec.dtype = ir::ScalarDType::F16;
  build_dw_kernel(k, spec);
  const std::string msl = emit(k).msl_source;
  assert_contains(msl, "threadgroup half m1_smem[m1_HH * m1_HW];");
  assert_contains(msl, "device half* out_ptr");
}

void test_full_pipeline_routes_depthwise() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_pipeline");
  DwSpec spec;
  spec.has_bias = true;
  spec.block_w = 64;
  spec.block_h = 32;
  build_dw_kernel(k, spec);
  const auto target = ir::TargetSpec::apple_m1_max();
  auto pm = passes::PassManager::build_m1_pipeline();
  passes::PassContext ctx{target};
  auto pr = pm.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "Pipeline error: " << pr.diagnostic << "\n";
    assert(false);
  }
  codegen::metal::MetalCodegen cg;
  auto res = cg.emit(k, target);
  if (!res.ok) {
    std::cerr << "Codegen error: " << res.error << "\n";
    assert(false);
  }
  assert_contains(res.msl_source, "kernel void dw_pipeline(");
  assert_contains(res.msl_source, "threadgroup float m1_smem");
}

void test_unmatched_kernels() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("noop");
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();
  bool matched = true;
  auto res = codegen::metal::try_emit_conv_dw_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !res.ok && res.error.empty());

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("dense_conv");
  DwSpec dense;
  dense.depthwise = false;
  build_dw_kernel(k2, dense);
  matched = true;
  res = codegen::metal::try_emit_conv_dw_kernel(k2, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !res.ok && res.error.empty());

  ir::Module mod3;
  ir::Kernel& k3 = mod3.add_kernel("transposed_dw");
  DwSpec tr;
  tr.transposed = true;
  build_dw_kernel(k3, tr);
  matched = true;
  res = codegen::metal::try_emit_conv_dw_kernel(k3, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !res.ok);
}

void test_malformed_operands_report_error() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("dw_bad");
  ir::IRBuilder b(k);
  auto x = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto w = b.add_param("w_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto o = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  std::vector<ir::ValueId> operands{x, w, o};
  for (const char* name : {"N", "C", "H", "W", "HO"}) {
    operands.push_back(b.add_param(name, ir::Type::scalar(ir::ScalarDType::I32)));
  }
  ir::ConvAttr attr;
  attr.kernel_size = {3, 3};
  attr.is_depthwise = true;
  b.emit_implicit_gemm_conv(operands, attr);
  b.emit_return();
  bool matched = false;
  auto res = codegen::metal::try_emit_conv_dw_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && !res.ok && !res.error.empty());
}

}  // namespace

int main() {
  test_sram_halo_3x3();
  test_default_tile_without_constexprs();
  test_asymmetric_kernel_attrs();
  test_bias_and_depth_multiplier();
  test_direct_fallback_for_large_halo();
  test_f16_halo_budget_and_types();
  test_full_pipeline_routes_depthwise();
  test_unmatched_kernels();
  test_malformed_operands_report_error();
  std::cout << "test_conv_dw_emitter: PASS\n";
  return 0;
}
