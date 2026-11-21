#include <cassert>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "codegen/metal/loss_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/loss_lowering.hpp"
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

ir::ValueId ptr_param(ir::IRBuilder& b, const std::string& name, ir::ScalarDType dt = ir::ScalarDType::F32,
                      bool writable = false, uint32_t div = 16) {
  return b.add_param(name, ir::Type::ptr(dt), false, 0, div, writable);
}

ir::ValueId dim_const(ir::IRBuilder& b, const std::string& name, int64_t v) {
  const auto dt = (v >= INT32_MIN && v <= INT32_MAX) ? ir::ScalarDType::I32 : ir::ScalarDType::I64;
  return b.add_param(name, ir::Type::scalar(dt), true, v);
}

const ir::Op& find_loss(const ir::Kernel& k) {
  for (const auto& op : k.ops) {
    if (!op->erased && op->kind == ir::OpKind::FusedLoss) return *op;
  }
  assert(false && "FusedLoss op not found");
  return *k.ops.front();
}

codegen::metal::CompiledMSL optimize_and_emit(ir::Kernel& k, uint32_t expected_tg = 256) {
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
  auto res = codegen::metal::try_emit_loss_kernel(k, target, &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error: " << res.error << "\n";
    assert(false);
  }
  auto via_codegen = codegen::metal::MetalCodegen::emit(k, target);
  assert(via_codegen.ok);
  assert(via_codegen.msl_source == res.msl_source);
  assert(res.launch_spec.entry_name == k.name);
  assert(res.launch_spec.tg_x == expected_tg && res.launch_spec.tg_y == 1 && res.launch_spec.tg_z == 1);
  assert(res.launch_spec.vec_width == 4);
  assert(!res.launch_spec.fast_math);
  assert_contains(res.msl_source, "kernel void " + k.name + "(");
  return res;
}

void test_loss_lowering_pass() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("mse_cfg");
  ir::IRBuilder b(k);
  auto pred = ptr_param(b, "pred_ptr");
  auto tgt = ptr_param(b, "tgt_ptr");
  auto partial = ptr_param(b, "partial_ptr", ir::ScalarDType::F32, false);
  auto total = b.add_param("total", ir::Type::scalar(ir::ScalarDType::I64));
  ir::LossAttr attr;
  attr.kind = ir::LossKind::MSE;
  attr.stage = 1;
  attr.tg_threads = 32;
  b.emit_fused_loss({pred, tgt, partial, total}, attr);
  b.emit_return();

  const auto target = ir::TargetSpec::apple_m1_max();
  passes::LossLoweringPass pass;
  passes::PassContext ctx{target};
  auto r1 = pass.run(k, ctx);
  assert(r1.ok && r1.changed);
  const auto& la = std::get<ir::LossAttr>(find_loss(k).attr);
  assert(la.tg_threads == 256);
  assert(k.params[2].writable);
  assert(!pass.run(k, ctx).changed);

  ir::Kernel& bad = mod.add_kernel("bad_stage");
  ir::IRBuilder bb(bad);
  auto p0 = ptr_param(bb, "p");
  auto p1 = ptr_param(bb, "t");
  auto p2 = ptr_param(bb, "out", ir::ScalarDType::F32, true);
  ir::LossAttr bad_attr;
  bad_attr.kind = ir::LossKind::Huber;
  bad_attr.stage = 3;
  bb.emit_fused_loss({p0, p1, p2, dim_const(bb, "N", 64)}, bad_attr);
  bb.emit_return();
  auto rbad = pass.run(bad, ctx);
  assert(!rbad.ok && !rbad.diagnostic.empty());

  ir::Kernel& bad_beta = mod.add_kernel("bad_beta");
  ir::IRBuilder bbb(bad_beta);
  auto q0 = ptr_param(bbb, "p");
  auto q1 = ptr_param(bbb, "t");
  auto q2 = ptr_param(bbb, "out", ir::ScalarDType::F32, true);
  ir::LossAttr neg_beta;
  neg_beta.kind = ir::LossKind::Huber;
  neg_beta.beta = -0.5f;
  neg_beta.stage = 1;
  bbb.emit_fused_loss({q0, q1, q2, dim_const(bbb, "N", 64)}, neg_beta);
  bbb.emit_return();
  assert(!pass.run(bad_beta, ctx).ok);
}

void test_mse_huber_kldiv_stage1_and_stage2() {
  struct Case {
    ir::LossKind kind;
    const char* name;
    float beta;
    const char* must_contain;
  };
  const std::vector<Case> cases = {
      {ir::LossKind::MSE, "mse_s1", 1.0f, "m1_acc4 = fma(m1_d, m1_d, m1_acc4);"},
      {ir::LossKind::Huber, "huber_s1", 1.0f, "select(m1_d - m1_half_beta, m1_half_inv_beta * (m1_d * m1_d)"},
      {ir::LossKind::Huber, "huber_zero_s1", 0.0f, "m1_acc4 += m1_d;"},
      {ir::LossKind::KLDiv, "kldiv_s1", 1.0f, "m1_t * (log(m1_st) - log(m1_sp))"},
  };
  for (const auto& c : cases) {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel(c.name);
    ir::IRBuilder b(k);
    auto pred = ptr_param(b, "pred_ptr");
    auto tgt = ptr_param(b, "tgt_ptr");
    auto part = ptr_param(b, "partial_ptr", ir::ScalarDType::F32, true);
    auto total = b.add_param("total", ir::Type::scalar(ir::ScalarDType::I64));
    ir::LossAttr attr;
    attr.kind = c.kind;
    attr.beta = c.beta;
    attr.stage = 1;
    b.emit_fused_loss({pred, tgt, part, total}, attr);
    b.emit_return();
    auto res = optimize_and_emit(k);
    assert_contains(res.msl_source, "device const float* m1p_pred_ptr [[buffer(0)]]");
    assert_contains(res.msl_source, "device const float* m1p_tgt_ptr [[buffer(1)]]");
    assert_contains(res.msl_source, "device float* m1p_partial_ptr [[buffer(2)]]");
    assert_contains(res.msl_source, "constant long& m1p_total [[buffer(3)]]");
    assert_contains(res.msl_source, "device const float4* m1_pv");
    assert_contains(res.msl_source, c.must_contain);
    assert(res.launch_spec.param_slots.size() == 4);
    assert(!res.launch_spec.param_slots[0].writable && !res.launch_spec.param_slots[1].writable);
    assert(res.launch_spec.param_slots[2].writable);
  }

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("loss_s2");
  ir::IRBuilder b2(k2);
  auto part = ptr_param(b2, "partial_ptr");
  auto out = ptr_param(b2, "out_ptr", ir::ScalarDType::F32, true);
  auto np = b2.add_param("num_partials", ir::Type::scalar(ir::ScalarDType::I32));
  auto denom = b2.add_param("denom", ir::Type::scalar(ir::ScalarDType::F32));
  ir::LossAttr attr2;
  attr2.kind = ir::LossKind::MSE;
  attr2.stage = 2;
  b2.emit_fused_loss({part, out, np, denom}, attr2);
  b2.emit_return();
  auto res2 = optimize_and_emit(k2);
  assert_contains(res2.msl_source, "device const float* m1p_partial_ptr [[buffer(0)]]");
  assert_contains(res2.msl_source, "device float* m1p_out_ptr [[buffer(1)]]");
  assert_contains(res2.msl_source, "constant int& m1p_num_partials [[buffer(2)]]");
  assert_contains(res2.msl_source, "constant float& m1p_denom [[buffer(3)]]");
  assert_contains(res2.msl_source, "m1p_out_ptr[0] = m1_tot / float(m1p_denom);");
  assert(res2.launch_spec.param_slots.size() == 4);
  assert(!res2.launch_spec.param_slots[0].writable && res2.launch_spec.param_slots[1].writable);
}

void test_hinge_stage1() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("hinge_s1_runtime");
  ir::IRBuilder b(k);
  auto pred = ptr_param(b, "pred_ptr");
  auto tgt = ptr_param(b, "tgt_ptr");
  auto part = ptr_param(b, "partial_ptr", ir::ScalarDType::F32, true);
  auto rows = b.add_param("rows", ir::Type::scalar(ir::ScalarDType::I64));
  auto cols = b.add_param("cols", ir::Type::scalar(ir::ScalarDType::I64));
  ir::LossAttr attr;
  attr.kind = ir::LossKind::Hinge;
  attr.stage = 1;
  b.emit_fused_loss({pred, tgt, part, rows, cols}, attr);
  b.emit_return();
  auto res = optimize_and_emit(k);
  assert_contains(res.msl_source, "max(1.0f - m1_p * m1_t, float4(0.0f))");
  assert_contains(res.msl_source, "device const float4* m1_pv");
  assert_contains(res.msl_source, "device const packed_float4* m1_pv");
  assert_contains(res.msl_source, "if ((m1_cols & 3L) == 0L || m1_rows == 1L)");

  ir::Kernel& k2 = mod.add_kernel("hinge_s1_const");
  ir::IRBuilder b2(k2);
  auto p2 = ptr_param(b2, "pred_ptr");
  auto t2 = ptr_param(b2, "tgt_ptr");
  auto o2 = ptr_param(b2, "partial_ptr", ir::ScalarDType::F32, true);
  b2.emit_fused_loss({p2, t2, o2, dim_const(b2, "R", 32768), dim_const(b2, "C", 32768)}, attr);
  b2.emit_return();
  auto res2 = optimize_and_emit(k2);
  assert_contains(res2.msl_source, "device const float4* m1_pv");
  assert_not_contains(res2.msl_source, "packed_float4");
}

void test_cross_entropy_stage1() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("ce_s1");
  ir::IRBuilder b(k);
  auto pred = ptr_param(b, "pred_ptr");
  auto tgt = ptr_param(b, "tgt_ptr", ir::ScalarDType::I64);
  auto part = ptr_param(b, "partial_ptr", ir::ScalarDType::F32, true);
  auto rows = b.add_param("B", ir::Type::scalar(ir::ScalarDType::I64));
  auto cols = b.add_param("C", ir::Type::scalar(ir::ScalarDType::I64));
  ir::LossAttr attr;
  attr.kind = ir::LossKind::CrossEntropy;
  attr.stage = 1;
  b.emit_fused_loss({pred, tgt, part, rows, cols}, attr);
  b.emit_return();
  auto res = optimize_and_emit(k);
  assert_contains(res.msl_source, "device const float* m1p_pred_ptr [[buffer(0)]]");
  assert_contains(res.msl_source, "device const long* m1p_tgt_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "device float* m1p_partial_ptr [[buffer(2)]]");
  assert_contains(res.msl_source, "simd_max(");
  assert_contains(res.msl_source, "const float4 m1_v0 =");
  assert_contains(res.msl_source, "(m1_rm + log(m1_rd)) - m1p_pred_ptr[m1_base + m1_t]");
  assert_contains(res.msl_source, "min((4L - (m1_base & 3L)) & 3L, m1_C)");
  assert(res.launch_spec.param_slots[1].dtype == ir::ScalarDType::I64);
  assert(!res.launch_spec.param_slots[1].writable);
  assert(res.launch_spec.param_slots[2].writable);
}

void test_triplet_margin_stage1() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("triplet_s1");
  ir::IRBuilder b(k);
  auto a = ptr_param(b, "anchor_ptr");
  auto p = ptr_param(b, "pos_ptr");
  auto n = ptr_param(b, "neg_ptr");
  auto part = ptr_param(b, "partial_ptr", ir::ScalarDType::F32, true);
  auto rows = b.add_param("B", ir::Type::scalar(ir::ScalarDType::I64));
  auto dim = b.add_param("D", ir::Type::scalar(ir::ScalarDType::I64));
  ir::LossAttr attr;
  attr.kind = ir::LossKind::TripletMargin;
  attr.margin = 1.0f;
  attr.eps = 1e-6f;
  attr.stage = 1;
  b.emit_fused_loss({a, p, n, part, rows, dim}, attr);
  b.emit_return();
  auto res = optimize_and_emit(k);
  assert_contains(res.msl_source, "device const float* m1p_anchor_ptr [[buffer(0)]]");
  assert_contains(res.msl_source, "device const float* m1p_pos_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "device const float* m1p_neg_ptr [[buffer(2)]]");
  assert_contains(res.msl_source, "device float* m1p_partial_ptr [[buffer(3)]]");
  assert_contains(res.msl_source, "const float4 m1_dap = (m1_va - m1_vp) + m1_eps;");
  assert_contains(res.msl_source, "max(sqrt(m1_sap) - sqrt(m1_san) + m1_margin, 0.0f)");
  assert(res.launch_spec.param_slots[3].writable);
  for (size_t i : {0u, 1u, 2u}) assert(!res.launch_spec.param_slots[i].writable);
}

void test_unaligned_and_rejection() {
  ir::Module mod;
  ir::Kernel& unaligned = mod.add_kernel("mse_unaligned");
  ir::IRBuilder bu(unaligned);
  auto p = ptr_param(bu, "pred_ptr", ir::ScalarDType::F32, false, 4);
  auto t = ptr_param(bu, "tgt_ptr", ir::ScalarDType::F32, false, 4);
  auto o = ptr_param(bu, "partial_ptr", ir::ScalarDType::F32, true, 16);
  ir::LossAttr attr;
  attr.kind = ir::LossKind::MSE;
  attr.stage = 1;
  bu.emit_fused_loss({p, t, o, dim_const(bu, "N", 1024)}, attr);
  bu.emit_return();
  auto ru = optimize_and_emit(unaligned);
  assert_contains(ru.msl_source, "device const packed_float4* m1_pv");

  ir::Kernel& non_loss = mod.add_kernel("vec_copy");
  ir::IRBuilder bn(non_loss);
  auto xp = ptr_param(bn, "x_ptr");
  auto op = ptr_param(bn, "out_ptr", ir::ScalarDType::F32, true);
  auto offs = bn.emit_arange(0, 64);
  bn.emit_store(bn.emit_ptr_add(op, offs), bn.emit_load(bn.emit_ptr_add(xp, offs)));
  bn.emit_return();
  bool matched = true;
  auto rn = codegen::metal::try_emit_loss_kernel(non_loss, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !rn.ok && rn.error.empty());

  ir::Kernel& bad_ce = mod.add_kernel("bad_ce_dtype");
  ir::IRBuilder bce(bad_ce);
  auto cp = ptr_param(bce, "pred_ptr");
  auto ct = ptr_param(bce, "tgt_ptr", ir::ScalarDType::F32);
  auto co = ptr_param(bce, "partial_ptr", ir::ScalarDType::F32, true);
  ir::LossAttr ce_attr;
  ce_attr.kind = ir::LossKind::CrossEntropy;
  ce_attr.stage = 1;
  bce.emit_fused_loss({cp, ct, co, dim_const(bce, "B", 8), dim_const(bce, "C", 16)}, ce_attr);
  bce.emit_return();
  matched = false;
  auto rce = codegen::metal::try_emit_loss_kernel(bad_ce, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && !rce.ok && !rce.error.empty());
}

}  // namespace

int main() {
  test_loss_lowering_pass();
  test_mse_huber_kldiv_stage1_and_stage2();
  test_hinge_stage1();
  test_cross_entropy_stage1();
  test_triplet_margin_stage1();
  test_unaligned_and_rejection();
  std::cout << "test_loss_emitter: PASS\n";
  return 0;
}
