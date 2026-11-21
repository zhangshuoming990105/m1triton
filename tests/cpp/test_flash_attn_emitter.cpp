#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "codegen/metal/flash_attn_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/flash_attn_lowering.hpp"
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

struct FaSpec {
  ir::ScalarDType dtype = ir::ScalarDType::F32;
  float scale = 0.03125f;
  bool causal = false;
  int32_t causal_offset = 0;
  int64_t block_m = 0;
  int64_t block_n = 0;
  int64_t block_d = 0;
  int64_t num_sg = 0;
  std::optional<ir::TileSpecAttr> tile_hint;
};

ir::OpId build_fa_kernel(ir::Kernel& k, const FaSpec& spec) {
  ir::IRBuilder b(k);
  auto q = b.add_param("q_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto kk = b.add_param("k_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto v = b.add_param("v_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto o = b.add_param("out_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, true);
  auto bh = b.add_param("BH", ir::Type::scalar(ir::ScalarDType::I32));
  auto lq = b.add_param("L_q", ir::Type::scalar(ir::ScalarDType::I32));
  auto lkv = b.add_param("L_kv", ir::Type::scalar(ir::ScalarDType::I32));
  auto d = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I32));
  if (spec.block_m > 0) b.add_param("BLOCK_M", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_m);
  if (spec.block_n > 0) b.add_param("BLOCK_N", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_n);
  if (spec.block_d > 0) b.add_param("BLOCK_D", ir::Type::scalar(ir::ScalarDType::I32), true, spec.block_d);
  if (spec.num_sg > 0) b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, spec.num_sg);

  ir::FlashAttnAttr attr;
  attr.scale = spec.scale;
  attr.causal = spec.causal;
  attr.causal_offset = spec.causal_offset;

  ir::OpHints hints;
  hints.tile_spec = spec.tile_hint;
  const ir::OpId fa = b.emit_flash_attention({q, kk, v, o, bh, lq, lkv, d}, attr, hints);
  b.emit_return();
  return fa;
}

const ir::FlashAttnAttr& run_pipeline(ir::Kernel& k, ir::OpId fa_op) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  auto pm = passes::PassManager::build_m1_pipeline();
  passes::PassContext ctx{target, true, false, nullptr};
  auto pr = pm.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "PassManager error: " << pr.diagnostic << "\n";
    assert(false);
  }
  return std::get<ir::FlashAttnAttr>(k.op(fa_op).attr);
}

void test_single_pass_sgmm_512() {
  ir::Kernel k("fa_single_512");
  FaSpec spec;
  spec.block_m = 16;
  spec.block_n = 512;
  spec.block_d = 16;
  spec.num_sg = 16;
  const ir::OpId fa = build_fa_kernel(k, spec);
  const auto& attr = run_pipeline(k, fa);
  assert(attr.bm == 16u);
  assert(attr.bn == 512u);
  assert(attr.bd == 16u);
  assert(attr.num_simdgroups == 16u);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 512u);
  assert(msl.launch_spec.tg_y == 1u);
  assert(msl.launch_spec.param_slots.size() == 8u);
  assert(msl.launch_spec.param_slots[3].writable);
  assert_contains(msl.msl_source, "threadgroup float m1_shm[8192];");
  assert_contains(msl.msl_source, "simdgroup_multiply_accumulate");
  assert_contains(msl.msl_source, "ulong2(0, 0), true");
  assert_contains(msl.msl_source, "simd_max");
  assert_contains(msl.msl_source, "simd_sum");
  assert_contains(msl.msl_source, "m1_bh * (m1_Lq * m1_D)");
  assert_not_contains(msl.msl_source, "m1_ms[");
}

void test_multitile_online_softmax_default_and_causal() {
  ir::Kernel k("fa_causal_256");
  FaSpec spec;
  spec.causal = true;
  spec.causal_offset = 4;
  spec.scale = 0.125f;
  const ir::OpId fa = build_fa_kernel(k, spec);
  const auto& attr = run_pipeline(k, fa);
  assert(attr.bm == 16u);
  assert(attr.bn == 256u);
  assert(attr.bd == 16u);
  assert(attr.num_simdgroups == 16u);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 512u);
  assert_contains(msl.msl_source, "m1_ms[16]");
  assert_contains(msl.msl_source, "m1_ls[16]");
  assert_contains(msl.msl_source, "m1_as[16]");
  assert_contains(msl.msl_source, "+ 4L");
  assert_contains(msl.msl_source, "m1_kv_lim");
  assert_contains(msl.msl_source, "m1_mma2");
}

void test_tile_hint_and_sram_budget_clamp() {
  ir::Kernel k("fa_hint_clamp");
  FaSpec spec;
  ir::TileSpecAttr ts;
  ts.bm = 16;
  ts.bn = 128;
  ts.bk = 32;
  ts.num_simdgroups = 16;
  spec.tile_hint = ts;
  const ir::OpId fa = build_fa_kernel(k, spec);
  const auto& attr = run_pipeline(k, fa);
  assert(attr.bm == 16u);
  assert(attr.bn == 128u);
  assert(attr.bd == 32u);
  assert(attr.num_simdgroups == 16u);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert_contains(msl.msl_source, "threadgroup float m1_shm[2112];");
}

void test_fp16_general_rowwise() {
  ir::Kernel k("fa_fp16");
  FaSpec spec;
  spec.dtype = ir::ScalarDType::F16;
  spec.block_m = 16;
  spec.block_n = 512;
  const ir::OpId fa = build_fa_kernel(k, spec);
  const auto& attr = run_pipeline(k, fa);
  assert(attr.bn == 256u);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert_contains(msl.msl_source, "device const half* q_ptr");
  assert_contains(msl.msl_source, "device half* out_ptr");
  assert_contains(msl.msl_source, "half(m1_acc)");
}

void test_invalid_nonfinite_scale_rejected() {
  ir::Kernel k("fa_bad_scale");
  FaSpec spec;
  spec.scale = std::numeric_limits<float>::infinity();
  (void)build_fa_kernel(k, spec);
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FlashAttentionLoweringPass pass;
  passes::PassContext ctx{target, false, false, nullptr};
  auto pr = pass.run(k, ctx);
  assert(!pr.ok);
}

void test_tight_sram_target_clamps_bn() {
  ir::Kernel k("fa_tight_sram");
  FaSpec spec;
  spec.block_m = 16;
  spec.block_n = 512;
  spec.num_sg = 16;
  const ir::OpId fa = build_fa_kernel(k, spec);

  auto target = ir::TargetSpec::apple_m1_max();
  target.max_threadgroup_memory_bytes = 8192;
  passes::FlashAttentionLoweringPass pass;
  passes::PassContext ctx{target, false, false, nullptr};
  auto pr = pass.run(k, ctx);
  assert(pr.ok);
  const auto& attr = std::get<ir::FlashAttnAttr>(k.op(fa).attr);
  assert(attr.bn <= 128u);
  assert((attr.bm * (attr.bn + 4u) + 3u * attr.bm) * 4u <= target.max_threadgroup_memory_bytes);
}

}  // namespace

int main() {
  test_single_pass_sgmm_512();
  test_multitile_online_softmax_default_and_causal();
  test_tile_hint_and_sram_budget_clamp();
  test_fp16_general_rowwise();
  test_invalid_nonfinite_scale_rejected();
  test_tight_sram_target_clamps_bn();
  std::cout << "test_flash_attn_emitter: PASS\n";
  return 0;
}
