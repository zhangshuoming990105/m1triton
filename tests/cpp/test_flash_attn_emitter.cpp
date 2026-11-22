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

void test_extended_fwd_gqa_lse_window_alibi_softcap() {
  ir::Kernel k("fa_ext_all");
  ir::IRBuilder b(k);
  auto q = b.add_param("q_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto kk = b.add_param("k_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto v = b.add_param("v_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto o = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  auto lse = b.add_param("lse_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  auto alibi = b.add_param("alibi_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto bh = b.add_param("BH", ir::Type::scalar(ir::ScalarDType::I32));
  auto lq = b.add_param("L_q", ir::Type::scalar(ir::ScalarDType::I32));
  auto lkv = b.add_param("L_kv", ir::Type::scalar(ir::ScalarDType::I32));
  auto d = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I32));
  auto hq = b.add_param("H_q", ir::Type::scalar(ir::ScalarDType::I32));
  b.add_param("BLOCK_M", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
  b.add_param("BLOCK_N", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
  b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, 4);

  ir::FlashAttnAttr attr;
  attr.mode = ir::FlashAttnMode::Fwd;
  attr.scale = 0.125f;
  attr.causal = true;
  attr.window_size = 128;
  attr.softcap = 50.0f;
  attr.has_alibi = true;
  attr.return_lse = true;
  attr.kv_group_size = 4;

  const ir::OpId fa = b.emit_flash_attention({q, kk, v, o, bh, lq, lkv, d, lse, alibi, hq}, attr, {});
  b.emit_return();
  const auto& post_attr = run_pipeline(k, fa);
  assert(post_attr.bm == 32u);
  assert(post_attr.bn == 32u);
  assert(post_attr.num_simdgroups == 4u);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 128u);
  assert(msl.launch_spec.param_slots.size() == 11u);
  assert(msl.launch_spec.param_slots[3].writable);
  assert(msl.launch_spec.param_slots[4].writable);
  assert(!msl.launch_spec.param_slots[5].writable);
  assert_contains(msl.msl_source, "device float* out_ptr");
  assert_contains(msl.msl_source, "device float* lse_ptr");
  assert_contains(msl.msl_source, "device const float* alibi_ptr");
  assert_contains(msl.msl_source, "m1_tanh");
  assert_contains(msl.msl_source, "m1_kv_bh");
  assert_contains(msl.msl_source, "m1_alibi_slope");
  assert_contains(msl.msl_source, "m1_Lb");
  assert_contains(msl.msl_source, "128L");
  assert_contains(msl.msl_source, "simd_shuffle_xor");
}

void test_bwd_dkdv_and_dq_emission() {
  const auto target = ir::TargetSpec::apple_m1_max();
  {
    ir::Kernel k("fa_bwd_dkdv");
    ir::IRBuilder b(k);
    auto q = b.add_param("q_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto kk = b.add_param("k_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto v = b.add_param("v_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto do_p = b.add_param("do_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto l_p = b.add_param("l_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto delta_p = b.add_param("delta_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto dk = b.add_param("dk_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
    auto dv = b.add_param("dv_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
    auto bh = b.add_param("BH", ir::Type::scalar(ir::ScalarDType::I32));
    auto lq = b.add_param("L_q", ir::Type::scalar(ir::ScalarDType::I32));
    auto lkv = b.add_param("L_kv", ir::Type::scalar(ir::ScalarDType::I32));
    auto d = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I32));
    b.add_param("BLOCK_M", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
    b.add_param("BLOCK_N", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
    b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, 4);

    ir::FlashAttnAttr attr;
    attr.mode = ir::FlashAttnMode::BwdDKDV;
    attr.scale = 0.125f;
    attr.causal = true;
    attr.softcap = 30.0f;
    const ir::OpId fa = b.emit_flash_attention({q, kk, v, do_p, l_p, delta_p, dk, dv, bh, lq, lkv, d}, attr, {});
    b.emit_return();
    (void)run_pipeline(k, fa);

    auto msl = codegen::metal::MetalCodegen::emit(k, target);
    assert(msl.ok);
    assert(msl.launch_spec.tg_x == 128u);
    assert(msl.launch_spec.param_slots.size() == 12u);
    assert(msl.launch_spec.param_slots[6].writable);
    assert(msl.launch_spec.param_slots[7].writable);
    assert_contains(msl.msl_source, "device float* dk_ptr");
    assert_contains(msl.msl_source, "device float* dv_ptr");
    assert_contains(msl.msl_source, "m1_tanh");
    assert_contains(msl.msl_source, "simdgroup_multiply_accumulate");
    assert_contains(msl.msl_source, "m1_q_start");
    assert_not_contains(msl.msl_source, "atomic_fetch_add");
  }
  {
    ir::Kernel k("fa_bwd_dq");
    ir::IRBuilder b(k);
    auto q = b.add_param("q_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto kk = b.add_param("k_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto v = b.add_param("v_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto do_p = b.add_param("do_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto l_p = b.add_param("l_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto delta_p = b.add_param("delta_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto dq = b.add_param("dq_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
    auto bh = b.add_param("BH", ir::Type::scalar(ir::ScalarDType::I32));
    auto lq = b.add_param("L_q", ir::Type::scalar(ir::ScalarDType::I32));
    auto lkv = b.add_param("L_kv", ir::Type::scalar(ir::ScalarDType::I32));
    auto d = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I32));
    b.add_param("BLOCK_M", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
    b.add_param("BLOCK_N", ir::Type::scalar(ir::ScalarDType::I32), true, 32);
    b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, 4);

    ir::FlashAttnAttr attr;
    attr.mode = ir::FlashAttnMode::BwdDQ;
    attr.scale = 0.125f;
    attr.causal = true;
    const ir::OpId fa = b.emit_flash_attention({q, kk, v, do_p, l_p, delta_p, dq, bh, lq, lkv, d}, attr, {});
    b.emit_return();
    (void)run_pipeline(k, fa);

    auto msl = codegen::metal::MetalCodegen::emit(k, target);
    assert(msl.ok);
    assert(msl.launch_spec.tg_x == 128u);
    assert(msl.launch_spec.param_slots.size() == 11u);
    assert(msl.launch_spec.param_slots[6].writable);
    assert_contains(msl.msl_source, "device float* dq_ptr");
    assert_contains(msl.msl_source, "simdgroup_multiply_accumulate");
    assert_contains(msl.msl_source, "m1_kv_end");
    assert_not_contains(msl.msl_source, "atomic_fetch_add");
  }
}

void test_paged_decode_emission() {
  ir::Kernel k("fa_paged_decode");
  ir::IRBuilder b(k);
  auto q = b.add_param("q_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto kc = b.add_param("k_cache", ir::Type::ptr(ir::ScalarDType::F32));
  auto vc = b.add_param("v_cache", ir::Type::ptr(ir::ScalarDType::F32));
  auto bt = b.add_param("block_tables", ir::Type::ptr(ir::ScalarDType::I32));
  auto cl = b.add_param("context_lens", ir::Type::ptr(ir::ScalarDType::I32));
  auto po = b.add_param("partial_o", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  auto plse = b.add_param("partial_lse", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  auto B = b.add_param("B", ir::Type::scalar(ir::ScalarDType::I32));
  auto hq = b.add_param("H_q", ir::Type::scalar(ir::ScalarDType::I32));
  auto hkv = b.add_param("H_kv", ir::Type::scalar(ir::ScalarDType::I32));
  auto mb = b.add_param("MaxBlocks", ir::Type::scalar(ir::ScalarDType::I32));
  auto d = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I32));
  b.add_param("BLOCK_SIZE", ir::Type::scalar(ir::ScalarDType::I32), true, 16);
  b.add_param("NUM_SPLITS", ir::Type::scalar(ir::ScalarDType::I32), true, 8);
  b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, 4);

  ir::FlashAttnAttr attr;
  attr.mode = ir::FlashAttnMode::PagedDecode;
  attr.scale = 0.125f;
  const ir::OpId fa = b.emit_flash_attention({q, kc, vc, bt, cl, po, plse, B, hq, hkv, mb, d}, attr, {});
  b.emit_return();
  (void)run_pipeline(k, fa);

  const auto target = ir::TargetSpec::apple_m1_max();
  auto msl = codegen::metal::MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 128u);
  assert(msl.launch_spec.param_slots.size() == 12u);
  assert(msl.launch_spec.param_slots[5].writable);
  assert(msl.launch_spec.param_slots[6].writable);
  assert_contains(msl.msl_source, "device float* partial_o");
  assert_contains(msl.msl_source, "device float* partial_lse");
  assert_contains(msl.msl_source, "block_tables");
  assert_contains(msl.msl_source, "context_lens");
  assert_contains(msl.msl_source, "m1_num_splits = 8L");
  assert_contains(msl.msl_source, "simd_sum");
}

}  // namespace

int main() {
  test_single_pass_sgmm_512();
  test_multitile_online_softmax_default_and_causal();
  test_tile_hint_and_sram_budget_clamp();
  test_fp16_general_rowwise();
  test_invalid_nonfinite_scale_rejected();
  test_tight_sram_target_clamps_bn();
  test_extended_fwd_gqa_lse_window_alibi_softcap();
  test_bwd_dkdv_and_dq_emission();
  test_paged_decode_emission();
  std::cout << "test_flash_attn_emitter: PASS\n";
  return 0;
}
