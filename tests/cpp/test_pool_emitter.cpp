#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "codegen/metal/metal_codegen.hpp"
#include "codegen/metal/pool_emitter.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/pass_manager.hpp"
#include "passes/pool_lowering.hpp"

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

struct PoolSpec {
  ir::PoolKind kind = ir::PoolKind::Max;
  uint32_t ndim = 1;
  std::vector<int32_t> k;
  std::vector<int32_t> s;
  std::vector<int32_t> p;
  std::vector<int32_t> d;
  bool count_include_pad = true;
  bool ceil_mode = false;
  int64_t block_w = 0;
  int64_t block_h = 0;
};

ir::OpId build_pool_kernel(ir::Kernel& k, const PoolSpec& spec) {
  ir::IRBuilder b(k);
  auto x = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto o = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  std::vector<ir::ValueId> operands{x, o};
  operands.push_back(b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32)));
  operands.push_back(b.add_param("C", ir::Type::scalar(ir::ScalarDType::I32)));
  const char* in_names[3] = {"D_in", "H_in", "W_in"};
  const char* out_names[3] = {"D_out", "H_out", "W_out"};
  for (uint32_t i = 0; i < spec.ndim; ++i) {
    operands.push_back(b.add_param(in_names[3 - spec.ndim + i], ir::Type::scalar(ir::ScalarDType::I32)));
  }
  for (uint32_t i = 0; i < spec.ndim; ++i) {
    operands.push_back(b.add_param(out_names[3 - spec.ndim + i], ir::Type::scalar(ir::ScalarDType::I32)));
  }
  if (spec.block_w > 0) b.add_param("BLOCK_W", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_w);
  if (spec.block_h > 0) b.add_param("BLOCK_H", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_h);
  ir::PoolAttr attr;
  attr.kind = spec.kind;
  attr.ndim = spec.ndim;
  attr.kernel_size = spec.k;
  attr.stride = spec.s;
  attr.padding = spec.p;
  attr.dilation = spec.d;
  attr.count_include_pad = spec.count_include_pad;
  attr.ceil_mode = spec.ceil_mode;
  const ir::OpId pool = b.emit_pool(operands, attr);
  b.emit_return();
  return pool;
}

const ir::PoolAttr& lower(ir::Kernel& k, ir::OpId pool) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  passes::PoolLoweringPass pass;
  passes::PassContext ctx{target};
  auto pr = pass.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "PoolLoweringPass error (" << k.name << "): " << pr.diagnostic << "\n";
    assert(false);
  }
  return std::get<ir::PoolAttr>(k.op(pool).attr);
}

std::string emit(const ir::Kernel& k, codegen::metal::CompiledMSL* out = nullptr) {
  bool matched = false;
  auto res = codegen::metal::try_emit_pool_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error: " << res.error << "\n";
    assert(false);
  }
  if (out != nullptr) *out = res;
  return res.msl_source;
}

void test_pass_1d_halo_decision() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool1d");
  auto pool = build_pool_kernel(k, {ir::PoolKind::Max, 1, {8}, {1}, {4}, {3}});
  const auto& pa = lower(k, pool);
  assert(pa.tg_x == 256 && pa.tg_y == 1);
  assert(pa.use_sram_halo);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("maxpool1d_s2");
  auto pool2 = build_pool_kernel(k2, {ir::PoolKind::Max, 1, {8}, {2}, {4}, {3}});
  assert(!lower(k2, pool2).use_sram_halo);
}

void test_pass_2d_halo_decision_and_defaults() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool2d");
  auto pool = build_pool_kernel(k, {ir::PoolKind::Max, 2, {4, 4}, {1, 1}, {1, 1}, {1, 1}});
  const auto& pa = lower(k, pool);
  assert(pa.tg_x == 16 && pa.tg_y == 16);
  assert(pa.use_sram_halo);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("avgpool2d");
  auto pool2 = build_pool_kernel(k2, {ir::PoolKind::Avg, 2, {11}, {}, {0}, {}});
  const auto& pa2 = lower(k2, pool2);
  assert((pa2.kernel_size == std::vector<int32_t>{11, 11}));
  assert((pa2.stride == std::vector<int32_t>{11, 11}));
  assert((pa2.dilation == std::vector<int32_t>{1, 1}));
  assert(!pa2.use_sram_halo);

  ir::Module mod3;
  ir::Kernel& k3 = mod3.add_kernel("maxpool2d_huge");
  auto pool3 = build_pool_kernel(k3, {ir::PoolKind::Max, 2, {33, 33}, {1, 1}, {0, 0}, {4, 4}});
  assert(!lower(k3, pool3).use_sram_halo);
}

void test_pass_3d_direct() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool3d");
  auto pool = build_pool_kernel(k, {ir::PoolKind::Max, 3, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {3, 3, 3}});
  const auto& pa = lower(k, pool);
  assert(!pa.use_sram_halo);
  assert(pa.tg_x * pa.tg_y <= 1024);
}

void test_emit_1d_max_sram() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool1d_sram");
  PoolSpec spec{ir::PoolKind::Max, 1, {8}, {1}, {4}, {3}};
  spec.block_w = 1024;
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  codegen::metal::CompiledMSL res;
  const std::string msl = emit(k, &res);
  assert_contains(msl, "kernel void maxpool1d_sram(");
  assert_contains(msl, "threadgroup float m1_smem[m1_SH * m1_SW];");
  assert_contains(msl, "constexpr int m1_SW = 1045;");
  assert_contains(msl, "constexpr int m1_SH = 1;");
  assert_contains(msl, "threadgroup_barrier(mem_flags::mem_threadgroup);");
  assert_contains(msl, "-INFINITY");
  assert_contains(msl, "isnan(m1_v)");
  assert_contains(msl, "const long m1_iw = m1_iw0 + m1_i;");
  assert_contains(msl, "device float* out_ptr [[buffer(1)]]");
  assert_contains(msl, "device const float* x_ptr [[buffer(0)]]");
  assert(res.launch_spec.tg_x == 256 && res.launch_spec.tg_y == 1);
  assert(res.launch_spec.param_slots.size() == 6);
  assert(res.launch_spec.param_slots[1].writable);
  assert(!res.launch_spec.param_slots[0].writable);
}

void test_emit_1d_avg_sram() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("avgpool1d_sram");
  PoolSpec spec{ir::PoolKind::Avg, 1, {8}, {1}, {4}, {1}};
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  const std::string msl = emit(k);
  assert_contains(msl, "constexpr int m1_SW = 263;");
  assert_contains(msl, "m1_acc / 8.0f");
  assert_contains(msl, "float m1_v = 0.0f;");
  assert_not_contains(msl, "INFINITY");
}

void test_emit_2d_max_sram() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool2d_sram");
  PoolSpec spec{ir::PoolKind::Max, 2, {4, 4}, {1, 1}, {1, 1}, {1, 1}};
  spec.block_w = 32;
  spec.block_h = 32;
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  codegen::metal::CompiledMSL res;
  const std::string msl = emit(k, &res);
  assert_contains(msl, "constexpr int m1_SW = 35;");
  assert_contains(msl, "constexpr int m1_SH = 35;");
  assert_contains(msl, "m1_ih * m1_Win + m1_iw");
  assert_contains(msl, "for (int m1_kh = 0; m1_kh < 4; ++m1_kh)");
  assert(res.launch_spec.tg_x == 16 && res.launch_spec.tg_y == 16);
}

void test_emit_2d_avg_direct() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("avgpool2d_direct");
  PoolSpec spec{ir::PoolKind::Avg, 2, {11, 11}, {11, 11}, {0, 0}, {1, 1}};
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  const std::string msl = emit(k);
  assert_not_contains(msl, "threadgroup float");
  assert_contains(msl, "m1_acc / 121.0f");
  assert_contains(msl, "const long m1_iw0 = m1_ow * 11 - 0;");
}

void test_emit_3d_direct_64bit() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("avgpool3d_direct");
  PoolSpec spec{ir::PoolKind::Avg, 3, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}};
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  const std::string msl = emit(k);
  assert_contains(msl, "const long m1_nc = m1_z / m1_Dout;");
  assert_contains(msl, "device const float* m1_xb = x_ptr + m1_nc * (m1_Din * m1_in_plane);");
  assert_contains(msl, "for (int m1_kd = 0; m1_kd < 3; ++m1_kd)");
  assert_contains(msl, "const long m1_id = m1_id0 + m1_kd * 1;");
  assert_contains(msl, "m1_acc / 27.0f");
  assert_not_contains(msl, "threadgroup float");
}

void test_emit_avg_exclude_pad_divisor() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("avgpool2d_exclude");
  PoolSpec spec{ir::PoolKind::Avg, 2, {3, 3}, {2, 2}, {1, 1}, {1, 1}};
  spec.count_include_pad = false;
  spec.ceil_mode = true;
  auto pool = build_pool_kernel(k, spec);
  lower(k, pool);
  const std::string msl = emit(k);
  assert_contains(msl, "float m1_div = 1.0f;");
  assert_contains(msl, "max(m1_sh, 0L)");
  assert_contains(msl, "m1_acc / m1_div");
}

void test_full_pipeline_routes_pool() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("maxpool3d_pipeline");
  PoolSpec spec{ir::PoolKind::Max, 3, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {3, 3, 3}};
  build_pool_kernel(k, spec);
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
  assert_contains(res.msl_source, "kernel void maxpool3d_pipeline(");
  assert_contains(res.msl_source, "const long m1_id = m1_id0 + m1_kd * 3;");
}

void test_non_pool_kernel_unmatched() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("noop");
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();
  bool matched = true;
  auto res = codegen::metal::try_emit_pool_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched);
  assert(!res.ok);
  assert(res.error.empty());
}

}  // namespace

int main() {
  test_pass_1d_halo_decision();
  test_pass_2d_halo_decision_and_defaults();
  test_pass_3d_direct();
  test_emit_1d_max_sram();
  test_emit_1d_avg_sram();
  test_emit_2d_max_sram();
  test_emit_2d_avg_direct();
  test_emit_3d_direct_64bit();
  test_emit_avg_exclude_pad_divisor();
  test_full_pipeline_routes_pool();
  test_non_pool_kernel_unmatched();
  std::cout << "test_pool_emitter: PASS\n";
  return 0;
}
