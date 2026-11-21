#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "codegen/metal/metal_codegen.hpp"
#include "codegen/metal/norm_softmax_emitter.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/online_softmax_welford.hpp"
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
    std::cerr << "Expected MSL NOT to contain:\n  " << needle << "\nActual MSL:\n" << haystack << "\n";
    assert(false);
  }
}

ir::ValueId ptr_param(ir::IRBuilder& b, const std::string& name, bool writable = false) {
  return b.add_param(name, ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, writable);
}

ir::ValueId dim(ir::IRBuilder& b, const std::string& name, int64_t v) {
  return b.add_param(name, ir::Type::scalar(ir::ScalarDType::I32), true, v);
}

const ir::Op& find_fused(const ir::Kernel& k) {
  for (const auto& op : k.ops) {
    if (!op->erased && (op->kind == ir::OpKind::OnlineSoftmax || op->kind == ir::OpKind::WelfordReduce)) return *op;
  }
  assert(false && "fused op not found");
  return *k.ops.front();
}

codegen::metal::CompiledMSL optimize_and_emit(ir::Kernel& k) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  auto pm = passes::PassManager::build_m2_pipeline();
  passes::PassContext ctx{target};
  auto pr = pm.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "Pass error: " << pr.diagnostic << "\n";
    assert(false);
  }
  bool matched = false;
  auto res = codegen::metal::try_emit_norm_softmax_kernel(k, target, &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error: " << res.error << "\n";
    assert(false);
  }
  auto via_codegen = codegen::metal::MetalCodegen::emit(k, target);
  assert(via_codegen.ok);
  assert(via_codegen.msl_source == res.msl_source);
  assert(res.launch_spec.entry_name == k.name);
  assert(res.launch_spec.tg_x == 256 && res.launch_spec.tg_y == 1 && res.launch_spec.tg_z == 1);
  assert(res.launch_spec.vec_width == 4);
  assert_contains(res.msl_source, "kernel void " + k.name + "(");
  return res;
}

ir::WelfordNormAttr norm_attr(ir::NormKind kind, int32_t stage = 0, bool affine = false) {
  ir::WelfordNormAttr a;
  a.kind = kind;
  a.stage = stage;
  a.has_weight = affine;
  a.has_bias = affine;
  a.tg_threads = 0;
  a.vec_width = 0;
  return a;
}

void test_pass_configures_attrs() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("rms_cfg");
  ir::IRBuilder b(k);
  auto x = ptr_param(b, "x_ptr");
  auto o = ptr_param(b, "out_ptr", true);
  b.emit_welford_norm({x, o, dim(b, "N", 2), dim(b, "C", 64), dim(b, "HW", 4096)}, norm_attr(ir::NormKind::RMSNorm));
  b.emit_return();

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("softmax_cfg");
  ir::IRBuilder b2(k2);
  auto x2 = ptr_param(b2, "x_ptr");
  auto o2 = ptr_param(b2, "out_ptr", true);
  ir::OnlineSoftmaxAttr sa;
  sa.tg_threads = 32;
  sa.vec_width = 1;
  b2.emit_online_softmax(x2, o2, dim(b2, "R", 4), dim(b2, "L", 1024), sa);
  b2.emit_return();

  const auto target = ir::TargetSpec::apple_m1_max();
  passes::OnlineSoftmaxWelfordPass pass;
  passes::PassContext ctx{target};
  auto r1 = pass.run(k, ctx);
  auto r2 = pass.run(k2, ctx);
  assert(r1.ok && r1.changed && r2.ok && r2.changed);
  const auto& wa = std::get<ir::WelfordNormAttr>(find_fused(k).attr);
  assert(wa.tg_threads == 256 && wa.vec_width == 4 && wa.rms_sram_cache);
  const auto& oa = std::get<ir::OnlineSoftmaxAttr>(find_fused(k2).attr);
  assert(oa.tg_threads == 256 && oa.vec_width == 4);
  assert(!pass.run(k, ctx).changed);

  ir::TargetSpec tiny = target;
  tiny.max_threadgroup_memory_bytes = 8192;
  passes::PassContext tiny_ctx{tiny};
  std::get<ir::WelfordNormAttr>(k.op(find_fused(k).id).attr).rms_sram_cache = false;
  pass.run(k, tiny_ctx);
  assert(!std::get<ir::WelfordNormAttr>(find_fused(k).attr).rms_sram_cache);
}

void test_softmax(bool log_softmax) {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel(log_softmax ? "log_softmax_rows" : "softmax_rows");
  ir::IRBuilder b(k);
  auto x = ptr_param(b, "x_ptr");
  auto o = ptr_param(b, "out_ptr", true);
  ir::OnlineSoftmaxAttr sa;
  sa.axis = 1;
  sa.log_softmax = log_softmax;
  b.emit_online_softmax(x, o, dim(b, "NUM_ROWS", 4096), dim(b, "ROW_LEN", 393216), sa);
  b.emit_return();
  auto res = optimize_and_emit(k);
  const std::string& msl = res.msl_source;
  assert_contains(msl, "device const float* m1p_x_ptr [[buffer(0)]]");
  assert_contains(msl, "device float* m1p_out_ptr [[buffer(1)]]");
  assert_contains(msl, "const long m1_base = long(m1_tgp.x) * long(m1_L);");
  assert_contains(msl, "device const float4* m1_xv");
  assert_contains(msl, "simd_shuffle_xor(m1_d, m1_o)");
  assert_contains(msl, "threadgroup float m1_sm[8];");
  assert_contains(msl, "for (int m1_i = int(m1_nv) - 1 - int(m1_tid); m1_i >= 0; m1_i -= 256)");
  assert_contains(msl, log_softmax ? "float4(v - m1_lse)" : "float4(exp(v - m1_rm) * m1_inv)");
  assert(res.launch_spec.param_slots.size() == 2);
  assert(res.launch_spec.param_slots[1].writable && !res.launch_spec.param_slots[0].writable);
}

void test_row_norms() {
  struct Case {
    ir::NormKind kind;
    int64_t rows;
    int64_t len;
    std::string must;
    std::string vec;
  };
  const std::vector<Case> cases = {
      {ir::NormKind::InstanceNorm, 7168, 262144, "m1_wf_tg(m1_wf_shifted(m1_cnt, m1_s, m1_ss, m1_k)", "float4"},
      {ir::NormKind::L1Norm, 32768, 65535, "float(m1_L) / m1_tot", "float4"},
      {ir::NormKind::L2Norm, 32768, 65535, "1.0f / sqrt(m1_tot)", "float4"},
  };
  for (const auto& c : cases) {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("row_norm");
    ir::IRBuilder b(k);
    auto x = ptr_param(b, "x_ptr");
    auto o = ptr_param(b, "out_ptr", true);
    auto attr = norm_attr(c.kind);
    b.emit_welford_norm({x, o, dim(b, "ROWS", c.rows), dim(b, "LEN", c.len)}, attr);
    b.emit_return();
    auto res = optimize_and_emit(k);
    assert_contains(res.msl_source, c.must);
    assert_contains(res.msl_source, "device const " + c.vec + "* m1_xv");
    assert_contains(res.msl_source, "long(m1_tgp.x) * long(m1_L)");
  }
}

void test_l2_unaligned_ptr_peels_head() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("l2_peel");
  ir::IRBuilder b(k);
  auto x = ptr_param(b, "x_ptr");
  auto o = ptr_param(b, "out_ptr", true);
  auto rows = b.add_param("ROWS", ir::Type::scalar(ir::ScalarDType::I32));
  auto len = b.add_param("LEN", ir::Type::scalar(ir::ScalarDType::I32));
  b.emit_welford_norm({x, o, rows, len}, norm_attr(ir::NormKind::L2Norm));
  b.emit_return();
  auto res = optimize_and_emit(k);
  assert_contains(res.msl_source, "constant int& m1p_ROWS [[buffer(2)]]");
  assert_contains(res.msl_source, "min(uint((4L - (m1_base & 3L)) & 3L), m1_len)");
  assert(res.launch_spec.param_slots.size() == 4);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("l2_packed");
  ir::IRBuilder b2(k2);
  auto x2 = b2.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 4, false);
  auto o2 = b2.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 4, true);
  b2.emit_welford_norm({x2, o2, dim(b2, "ROWS", 32768), dim(b2, "LEN", 65535)}, norm_attr(ir::NormKind::L2Norm));
  b2.emit_return();
  auto res2 = optimize_and_emit(k2);
  assert_contains(res2.msl_source, "device const packed_float4* m1_xv");
  assert_contains(res2.msl_source, "const uint m1_hd = 0u;");
}

void test_rms_sram_and_fallback() {
  for (int64_t channels : {64, 96}) {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("rms_norm");
    ir::IRBuilder b(k);
    auto x = ptr_param(b, "x_ptr");
    auto o = ptr_param(b, "out_ptr", true);
    auto attr = norm_attr(ir::NormKind::RMSNorm);
    attr.eps = 1e-5f;
    b.emit_welford_norm({x, o, dim(b, "N", 112), dim(b, "C", channels), dim(b, "HW", 262144)}, attr);
    b.emit_return();
    auto res = optimize_and_emit(k);
    if (channels == 64) {
      assert_contains(res.msl_source, "threadgroup float4 m1_tile[1024];");
      assert_contains(res.msl_source, "m1_tile[m1_i] * m1_scl4[m1_i & 15u]");
    } else {
      assert_not_contains(res.msl_source, "m1_tile");
      assert_contains(res.msl_source, "const bool m1_ok = m1_s0 + m1_pos < m1_HW;");
    }
    assert_contains(res.msl_source, "1.0f / sqrt(m1_t / float(m1_C) + 1.000000e-05f)");
  }
}

void test_frobenius_stages() {
  ir::Module mod;
  ir::Kernel& k1 = mod.add_kernel("frob_s1");
  ir::IRBuilder b1(k1);
  auto x1 = ptr_param(b1, "x_ptr");
  auto p1 = ptr_param(b1, "partial_ptr", true);
  b1.emit_welford_norm({x1, p1, dim(b1, "TOTAL", 1879048192)}, norm_attr(ir::NormKind::FrobeniusNorm, 1));
  b1.emit_return();
  auto r1 = optimize_and_emit(k1);
  assert_contains(r1.msl_source, "m1p_partial_ptr[m1_tgp.x] = m1_tot;");
  assert_contains(r1.msl_source, "m1_acc += dot(v, v);");

  ir::Kernel& k2 = mod.add_kernel("frob_s2");
  ir::IRBuilder b2(k2);
  auto x2 = ptr_param(b2, "x_ptr");
  auto o2 = ptr_param(b2, "out_ptr", true);
  auto p2 = ptr_param(b2, "partial_ptr");
  b2.emit_welford_norm({x2, o2, p2, dim(b2, "TOTAL", 1879048192), dim(b2, "NP", 2048)},
                       norm_attr(ir::NormKind::FrobeniusNorm, 2));
  b2.emit_return();
  auto r2 = optimize_and_emit(k2);
  assert_contains(r2.msl_source, "device const float* m1p_partial_ptr [[buffer(2)]]");
  assert_contains(r2.msl_source, "float4(v * m1_inv)");
}

void test_two_stage_norm(ir::NormKind kind, const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
                         const std::string& stage1_must, const std::string& stage2_must) {
  ir::Module mod;
  ir::Kernel& k1 = mod.add_kernel("stats");
  ir::IRBuilder b1(k1);
  std::vector<ir::ValueId> ops1 = {ptr_param(b1, "x_ptr"), ptr_param(b1, "partial_ptr", true)};
  for (size_t i = 0; i < dims1.size(); ++i) ops1.push_back(dim(b1, "d" + std::to_string(i), dims1[i]));
  auto a1 = norm_attr(kind, 1);
  a1.num_groups = 8;
  b1.emit_welford_norm(ops1, a1);
  b1.emit_return();
  auto r1 = optimize_and_emit(k1);
  assert_contains(r1.msl_source, stage1_must);
  assert_contains(r1.msl_source, "m1_q[2] = m1_w.m2;");

  ir::Kernel& k2 = mod.add_kernel("apply");
  ir::IRBuilder b2(k2);
  std::vector<ir::ValueId> ops2 = {ptr_param(b2, "x_ptr"), ptr_param(b2, "out_ptr", true), ptr_param(b2, "w_ptr"),
                                   ptr_param(b2, "b_ptr"), ptr_param(b2, "partial_ptr")};
  for (size_t i = 0; i < dims2.size(); ++i) ops2.push_back(dim(b2, "d" + std::to_string(i), dims2[i]));
  auto a2 = norm_attr(kind, 2, true);
  a2.num_groups = 8;
  b2.emit_welford_norm(ops2, a2);
  b2.emit_return();
  auto r2 = optimize_and_emit(k2);
  assert_contains(r2.msl_source, "m1_wf_partials(m1p_partial_ptr");
  assert_contains(r2.msl_source, stage2_must);
  assert(r2.launch_spec.param_slots.size() == 5);
  assert(r2.launch_spec.param_slots[1].writable);
  for (size_t i : {0u, 2u, 3u, 4u}) assert(!r2.launch_spec.param_slots[i].writable);
}

void test_rejects_bad_operands_and_non_norm() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("bad_bn");
  ir::IRBuilder b(k);
  auto x = ptr_param(b, "x_ptr");
  auto o = ptr_param(b, "out_ptr", true);
  b.emit_welford_norm({x, o, dim(b, "N", 2)}, norm_attr(ir::NormKind::BatchNorm, 1));
  b.emit_return();
  bool matched = false;
  auto res = codegen::metal::try_emit_norm_softmax_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && !res.ok && !res.error.empty());

  ir::Kernel& k2 = mod.add_kernel("vector_copy");
  ir::IRBuilder b2(k2);
  auto xp = ptr_param(b2, "x_ptr");
  auto op = ptr_param(b2, "out_ptr", true);
  auto offs = b2.emit_arange(0, 128);
  b2.emit_store(b2.emit_ptr_add(op, offs), b2.emit_load(b2.emit_ptr_add(xp, offs)));
  b2.emit_return();
  matched = true;
  auto res2 = codegen::metal::try_emit_norm_softmax_kernel(k2, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !res2.ok && res2.error.empty());
}

}  // namespace

int main() {
  test_pass_configures_attrs();
  test_softmax(false);
  test_softmax(true);
  test_row_norms();
  test_l2_unaligned_ptr_peels_head();
  test_rms_sram_and_fallback();
  test_frobenius_stages();
  test_two_stage_norm(ir::NormKind::BatchNorm, {64, 64, 262144, 64}, {64, 64, 262144, 64},
                      "for (uint m1_b = m1_n0; m1_b < m1_n1; ++m1_b)", "const uint m1_first = m1_c * m1_CH;");
  test_two_stage_norm(ir::NormKind::GroupNorm, {112, 64, 262144, 32}, {112, 64, 262144, 32},
                      "const long m1_S = long(m1_C / 8u) * long(262144);",
                      "const uint m1_first = ((m1_row / m1_C) * 8u + m1_c / (m1_C / 8u)) * m1_CH;");
  test_two_stage_norm(ir::NormKind::LayerNorm, {16, 4194304, 64}, {16, 4194304, 64},
                      "const long m1_S = long(4194304);", "(((v - m1_mu) * m1_inv) * w) + b");
  test_rejects_bad_operands_and_non_norm();
  std::cout << "test_norm_softmax_emitter: PASS\n";
  return 0;
}
