#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "codegen/metal/msl_common.hpp"
#include "ir/builder.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/algebraic_norm_fold.hpp"
#include "passes/epilogue_dag_fusion.hpp"
#include "passes/pass_manager.hpp"

namespace {

using namespace m1triton;

void test_ir_roundtrip_and_verifier() {
  const auto target = ir::TargetSpec::apple_m1_max();
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("fused_roundtrip");
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  b.emit_return();

  ir::EpilogueDAGAttr dag;
  dag.steps = {
      {ir::EpilogueOpKind::AddBias, 0.0f, 0.0f, 1},
      {ir::EpilogueOpKind::SaveReg, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::Sigmoid, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::MulConst, 2.0f, 0.0f, -1},
      {ir::EpilogueOpKind::AddSaved, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::Clamp, -1.0f, 1.0f, -1},
  };
  k.hints.epilogue_dag = dag;
  k.hints.fused_tail = ir::FusedTailAttr{ir::TailKind::RowSum, 1, 1, 0.5f, true};

  auto vr = ir::IRVerifier::verify(mod, target);
  assert(vr.ok && "Valid epilogue_dag + fused_tail should verify");

  const std::string printed = ir::IRPrinter::print(mod);
  auto parsed = ir::IRParser::parse(printed);
  assert(parsed.ok && parsed.module && "Parsed roundtrip failed");
  const ir::Kernel& rk = *parsed.module->kernels[0];
  assert(rk.hints.epilogue_dag.has_value());
  assert(rk.hints.epilogue_dag->steps.size() == dag.steps.size());
  assert(rk.hints.epilogue_dag->steps[0].op == ir::EpilogueOpKind::AddBias);
  assert(rk.hints.epilogue_dag->steps[0].arg_idx == 1);
  assert(rk.hints.epilogue_dag->steps[5].op == ir::EpilogueOpKind::Clamp);
  assert(rk.hints.epilogue_dag->steps[5].param0 == -1.0f && rk.hints.epilogue_dag->steps[5].param1 == 1.0f);
  assert(rk.hints.fused_tail.has_value());
  assert(rk.hints.fused_tail->kind == ir::TailKind::RowSum);
  assert(std::fabs(rk.hints.fused_tail->post_scale - 0.5f) < 1e-6f);

  // Verifier must reject unsaved register read and out-of-range param index.
  ir::Kernel bad_k;
  bad_k.name = "bad_epilogue";
  ir::IRBuilder bb(bad_k);
  bb.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  bb.emit_return();
  bad_k.hints.epilogue_dag = ir::EpilogueDAGAttr{{{ir::EpilogueOpKind::AddSaved, 0.0f, 0.0f, 0}}};
  assert(!ir::IRVerifier::verify(bad_k, target).ok);

  bad_k.hints.epilogue_dag = ir::EpilogueDAGAttr{{{ir::EpilogueOpKind::AddBias, 0.0f, 0.0f, 9}}};
  assert(!ir::IRVerifier::verify(bad_k, target).ok);

  bad_k.hints.epilogue_dag = ir::EpilogueDAGAttr{{{ir::EpilogueOpKind::Clamp, 2.0f, -1.0f, -1}}};
  assert(!ir::IRVerifier::verify(bad_k, target).ok);
}

void test_algebraic_norm_folding_src_dst() {
  const auto target = ir::TargetSpec::apple_m1_max();
  ir::Kernel k;
  k.name = "alg_fold";
  ir::IRBuilder b(k);
  b.emit_return();

  k.hints.epilogue_dag = ir::EpilogueDAGAttr{{
      {ir::EpilogueOpKind::SaveReg, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::MulConst, 1.5f, 0.0f, -1},
      {ir::EpilogueOpKind::AddSaved, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::DivConst, 2.5f, 0.0f, -1},
      {ir::EpilogueOpKind::SubConst, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::Clamp, -2.0f, 2.0f, -1},
      {ir::EpilogueOpKind::Clamp, -1.0f, 3.0f, -1},
      {ir::EpilogueOpKind::ReLU, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::ReLU, 0.0f, 0.0f, -1},
  }};

  passes::AlgebraicNormFoldingPass pass;
  passes::PassContext ctx{target, true, false, nullptr};
  auto res = pass.run(k, ctx);
  assert(res.ok && res.changed);
  assert(k.hints.epilogue_dag.has_value());
  // (1.5 + 1.0) / 2.5 == 1.0 (dropped!), Clamp(-2,2)+Clamp(-1,3) -> Clamp(-1,2), ReLU+ReLU -> ReLU.
  const auto& steps = k.hints.epilogue_dag->steps;
  assert(steps.size() == 2);
  assert(steps[0].op == ir::EpilogueOpKind::Clamp && steps[0].param0 == -1.0f && steps[0].param1 == 2.0f);
  assert(steps[1].op == ir::EpilogueOpKind::ReLU);
}

void test_post_dot_ssa_auto_fusion() {
  const auto target = ir::TargetSpec::apple_m1_max();
  ir::Kernel k;
  k.name = "dot_autofuse";
  ir::IRBuilder b(k);
  ir::ValueId a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  ir::ValueId b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  ir::ValueId c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  ir::ValueId r = b.emit_arange(0, 16);
  ir::ValueId c = b.emit_arange(0, 16);
  ir::ValueId r2 = b.emit_expand_dims(r, 1);
  ir::ValueId c2 = b.emit_expand_dims(c, 0);
  ir::ValueId idx = b.emit_binary(ir::BinaryOp::Add, b.emit_broadcast(r2, {16, 16}), b.emit_broadcast(c2, {16, 16}));
  ir::ValueId pa = b.emit_ptr_add(b.emit_splat(a_ptr, {16, 16}), idx);
  ir::ValueId pb = b.emit_ptr_add(b.emit_splat(b_ptr, {16, 16}), idx);
  ir::ValueId pc = b.emit_ptr_add(b.emit_splat(c_ptr, {16, 16}), idx);
  ir::ValueId ta = b.emit_load(pa);
  ir::ValueId tb = b.emit_load(pb);
  ir::ValueId dot = b.emit_dot(ta, tb);
  ir::ValueId c_sub = b.emit_splat(b.emit_const_f32(0.5), {16, 16});
  ir::ValueId v1 = b.emit_binary(ir::BinaryOp::Sub, dot, c_sub);
  ir::ValueId c_mul = b.emit_splat(b.emit_const_f32(2.0), {16, 16});
  ir::ValueId v2 = b.emit_binary(ir::BinaryOp::Mul, v1, c_mul);
  ir::ValueId c_zero = b.emit_splat(b.emit_const_f32(0.0), {16, 16});
  ir::ValueId v3 = b.emit_binary(ir::BinaryOp::Max, v2, c_zero);
  b.emit_store(pc, v3);
  b.emit_return();

  assert(ir::IRVerifier::verify(k, target).ok);

  passes::EpilogueDAGFusionPass fuse;
  passes::PassContext ctx{target, true, false, nullptr};
  auto res = fuse.run(k, ctx);
  assert(res.ok && res.changed);
  assert(ir::IRVerifier::verify(k, target).ok);
  assert(k.hints.epilogue_dag.has_value());
  const auto& steps = k.hints.epilogue_dag->steps;
  assert(steps.size() == 3);
  assert(steps[0].op == ir::EpilogueOpKind::SubConst && steps[0].param0 == 0.5f);
  assert(steps[1].op == ir::EpilogueOpKind::MulConst && steps[1].param0 == 2.0f);
  assert(steps[2].op == ir::EpilogueOpKind::ReLU);
}

void test_shared_msl_epilogue_emission() {
  ir::Kernel k;
  k.name = "msl_epilogue_test";
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.add_param("bias_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();

  ir::EpilogueDAGAttr dag;
  dag.steps = {
      {ir::EpilogueOpKind::AddBias, 0.0f, 0.0f, 1},
      {ir::EpilogueOpKind::GELU, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::Mish, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::ReLU, 0.0f, 0.0f, -1},
  };
  std::ostringstream os;
  codegen::metal::HelperFlags hf;
  codegen::metal::emit_epilogue_dag_msl(
      os, k, dag, "v", "c", "idx", [&](uint32_t i) { return k.params[i].name; }, hf, "  ");
  const std::string code = os.str();
  assert(hf.needs_erf && hf.needs_tanh);
  assert(code.find("bias_ptr[c]") != std::string::npos);
  assert(code.find("m1_erf(") != std::string::npos);
  assert(code.find("m1_tanh(") != std::string::npos);
  assert(code.find("(v < 0.0f) ? 0.0f : v") != std::string::npos);
}

}  // namespace

int main() {
  test_ir_roundtrip_and_verifier();
  test_algebraic_norm_folding_src_dst();
  test_post_dot_ssa_auto_fusion();
  test_shared_msl_epilogue_emission();
  std::cout << "test_v2_fusion_passes: ALL PASSED\n";
  return 0;
}
