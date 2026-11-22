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
#include "passes/fusion_cost_model.hpp"
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

ir::Kernel make_dot_tile_kernel(const std::string& name, int64_t bm, int64_t bn, int64_t bk,
                                ir::ScalarDType dt, ir::TileSpecAttr ts) {
  ir::Kernel k;
  k.name = name;
  ir::IRBuilder b(k);
  const ir::ValueId a = b.add_param("a", ir::Type::tile({bm, bk}, dt));
  const ir::ValueId w = b.add_param("b", ir::Type::tile({bk, bn}, dt));
  b.add_param("bias", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_dot(a, w, ir::kInvalidId, ir::DotAttr{});
  b.emit_return();
  k.hints.tile_spec = ts;
  return k;
}

void test_fusion_cost_model_pso_simdgroup_clamp() {
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FusionOccupancyCostModelPass pass;
  passes::PassContext ctx{target, true, false, nullptr};

  // Kernel-level PSO clamp: num_simdgroups=64 (2048 threads > 1024) -> clamped to <= 32.
  ir::Kernel k = make_dot_tile_kernel("pso_clamp_kernel", 64, 64, 16, ir::ScalarDType::F32,
                                      ir::TileSpecAttr{64, 64, 16, 64, false, 1, 8});
  assert(ir::IRVerifier::verify(k, target).ok);

  const auto res = pass.run(k, ctx);
  assert(res.ok && res.changed);
  assert(ir::IRVerifier::verify(k, target).ok);
  assert(k.hints.tile_spec.has_value());
  assert(k.hints.tile_spec->num_simdgroups <= 32u);
  assert(k.hints.tile_spec->num_simdgroups == 32u);
  assert(pass.last_report().clamped_simdgroups);
  assert(pass.last_report().threads_per_threadgroup <= target.max_threads_per_threadgroup);

  // Idempotent on second run.
  const auto res2 = pass.run(k, ctx);
  assert(res2.ok && !res2.changed);

  // Per-op Dot tile_spec clamp.
  ir::Kernel k_op = make_dot_tile_kernel("pso_clamp_op", 64, 64, 16, ir::ScalarDType::F32,
                                         ir::TileSpecAttr{64, 64, 16, 4, false, 1, 8});
  k_op.hints.tile_spec.reset();
  k_op.op(0).hints.tile_spec = ir::TileSpecAttr{64, 64, 16, 64, false, 1, 8};
  const auto res_op = pass.run(k_op, ctx);
  assert(res_op.ok && res_op.changed);
  assert(k_op.op(0).hints.tile_spec->num_simdgroups == 32u);
  assert(!pass.run(k_op, ctx).changed);
}

void test_fusion_cost_model_double_buffer_downgrade() {
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FusionOccupancyCostModelPass pass;
  passes::PassContext ctx{target, true, false, nullptr};

  // F16 128x128x32: 1-stage=18944 B <= 32768 B, 2-stage=37888 B > 32768 B -> double_buffer=false, tile preserved.
  {
    ir::Kernel k = make_dot_tile_kernel("dbl_downgrade_f16", 128, 128, 32, ir::ScalarDType::F16,
                                        ir::TileSpecAttr{128, 128, 32, 8, true, 1, 8});
    const auto before = passes::FusionOccupancyCostModelPass::analyze(k, target);
    assert(before.estimated_sram_bytes == 37888u);

    const auto res = pass.run(k, ctx);
    assert(res.ok && res.changed);
    assert(!k.hints.tile_spec->double_buffer);
    assert(k.hints.tile_spec->bm == 128u && k.hints.tile_spec->bn == 128u && k.hints.tile_spec->bk == 32u);
    assert(pass.last_report().disabled_double_buffer);
    assert(!pass.last_report().clamped_tile);

    const auto after = passes::FusionOccupancyCostModelPass::analyze(k, target);
    assert(after.estimated_sram_bytes == 18944u);
    assert(after.estimated_sram_bytes <= target.max_threadgroup_memory_bytes);
    assert(!pass.run(k, ctx).changed);
  }

  // F32 128x128x32 with double_buffer=true: disables double_buffer and clamps bk so SRAM fits <= 32768 B.
  {
    ir::Kernel k = make_dot_tile_kernel("dbl_downgrade_f32", 128, 128, 32, ir::ScalarDType::F32,
                                        ir::TileSpecAttr{128, 128, 32, 8, true, 1, 8});
    const auto res = pass.run(k, ctx);
    assert(res.ok && res.changed);
    assert(!k.hints.tile_spec->double_buffer);
    assert(pass.last_report().disabled_double_buffer);
    const auto after = passes::FusionOccupancyCostModelPass::analyze(k, target);
    assert(after.estimated_sram_bytes <= target.max_threadgroup_memory_bytes);
    assert(!pass.run(k, ctx).changed);
  }
}

void test_fusion_cost_model_single_buffer_sram_clamp() {
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FusionOccupancyCostModelPass pass;
  passes::PassContext ctx{target, true, false, nullptr};

  // F32 128x128x64 single-buffered: 68608 B > 32768 B -> halves bk down to 16 (18688 B <= 32768 B).
  {
    ir::Kernel k = make_dot_tile_kernel("sram_clamp_bk", 128, 128, 64, ir::ScalarDType::F32,
                                        ir::TileSpecAttr{128, 128, 64, 8, false, 1, 8});
    const auto before = passes::FusionOccupancyCostModelPass::analyze(k, target);
    assert(before.estimated_sram_bytes > target.max_threadgroup_memory_bytes);

    const auto res = pass.run(k, ctx);
    assert(res.ok && res.changed);
    assert(pass.last_report().clamped_tile);
    assert(k.hints.tile_spec->bk == 16u);
    const auto after = passes::FusionOccupancyCostModelPass::analyze(k, target);
    assert(after.estimated_sram_bytes <= target.max_threadgroup_memory_bytes);
    assert(!pass.run(k, ctx).changed);
  }

  // Tight SRAM budget forces bk down to 8 and halves bn/bm down until <= budget.
  {
    ir::TargetSpec tight = target;
    tight.max_threadgroup_memory_bytes = 8192u;
    passes::PassContext tight_ctx{tight, true, false, nullptr};

    ir::Kernel k = make_dot_tile_kernel("sram_clamp_bm_bn", 128, 128, 64, ir::ScalarDType::F32,
                                        ir::TileSpecAttr{128, 128, 64, 8, false, 1, 8});
    k.hints.fused_tail = ir::FusedTailAttr{ir::TailKind::RowSum, 1, 1, 1.0f, true};

    const auto res = pass.run(k, tight_ctx);
    assert(res.ok && res.changed);
    assert(pass.last_report().clamped_tile);
    assert(k.hints.tile_spec->bk == 8u);
    assert(k.hints.tile_spec->bm < 128u || k.hints.tile_spec->bn < 128u);
    assert(k.hints.tile_spec->bm >= 32u && k.hints.tile_spec->bn >= 32u);

    const auto after = passes::FusionOccupancyCostModelPass::analyze(k, tight);
    assert(after.has_fused_tail);
    assert(after.estimated_sram_bytes <= tight.max_threadgroup_memory_bytes);
    assert(!pass.run(k, tight_ctx).changed);
  }
}

void test_fusion_cost_model_deep_epilogue_reg_pressure() {
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FusionOccupancyCostModelPass pass;
  passes::PassContext ctx{target, true, false, nullptr};

  ir::Kernel k = make_dot_tile_kernel("deep_epilogue_guard", 128, 128, 16, ir::ScalarDType::F32,
                                      ir::TileSpecAttr{128, 128, 16, 4, false, 1, 8});
  k.hints.epilogue_dag = ir::EpilogueDAGAttr{{
      {ir::EpilogueOpKind::AddBias, 0.0f, 0.0f, 2},
      {ir::EpilogueOpKind::SaveReg, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::GELU, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::Mish, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::AddSaved, 0.0f, 0.0f, 0},
      {ir::EpilogueOpKind::ReLU, 0.0f, 0.0f, -1},
  }};
  assert(ir::IRVerifier::verify(k, target).ok);

  const auto before = passes::FusionOccupancyCostModelPass::analyze(k, target);
  assert(before.fused_epilogue_ops == 6u);
  assert(before.epilogue_saved_regs == 1u);
  assert(before.threads_per_threadgroup == 128u);
  assert(before.estimated_registers_per_thread == 16u + (2u * 64u + 8u) + (2u * 1u + 4u + 1u));

  const auto res = pass.run(k, ctx);
  assert(res.ok && res.changed);
  assert(k.hints.tile_spec->num_simdgroups > 4u);
  assert(k.hints.tile_spec->num_simdgroups == 16u);
  assert(pass.last_report().clamped_simdgroups);

  const auto after = passes::FusionOccupancyCostModelPass::analyze(k, target);
  assert(after.estimated_registers_per_thread < before.estimated_registers_per_thread);
  assert(after.estimated_occupancy > before.estimated_occupancy);
  assert(after.arithmetic_intensity > 0.0);

  // Idempotent on second run.
  const auto res2 = pass.run(k, ctx);
  assert(res2.ok && !res2.changed);
}

void test_fusion_cost_model_preserves_well_sized_tiles_and_pointwise() {
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::FusionOccupancyCostModelPass pass;
  passes::PassContext ctx{target, true, false, nullptr};

  struct WellSizedCase {
    const char* name;
    uint32_t bm;
    uint32_t bn;
    uint32_t bk;
    uint32_t nsg;
    bool dbl;
  };
  const WellSizedCase cases[] = {
      {"ws_64x64x16_4", 64, 64, 16, 4, false},
      {"ws_64x64x16_4_dbl", 64, 64, 16, 4, true},
      {"ws_64x128x16_8", 64, 128, 16, 8, false},
      {"ws_128x64x16_8_dbl", 128, 64, 16, 8, true},
      {"ws_128x64x32_8", 128, 64, 32, 8, false},
  };

  for (const auto& c : cases) {
    const ir::TileSpecAttr orig{c.bm, c.bn, c.bk, c.nsg, c.dbl, 1, 8};
    ir::Kernel k = make_dot_tile_kernel(c.name, c.bm, c.bn, c.bk, ir::ScalarDType::F32, orig);
    k.hints.epilogue_dag = ir::EpilogueDAGAttr{{
        {ir::EpilogueOpKind::AddBias, 0.0f, 0.0f, 2},
        {ir::EpilogueOpKind::SaveReg, 0.0f, 0.0f, 0},
        {ir::EpilogueOpKind::GELU, 0.0f, 0.0f, -1},
        {ir::EpilogueOpKind::AddSaved, 0.0f, 0.0f, 0},
    }};
    const auto res = pass.run(k, ctx);
    assert(res.ok && !res.changed);
    assert(*k.hints.tile_spec == orig);
  }

  // Pointwise kernel cost model analysis (AI = epilogue FLOPs / 8.0).
  ir::Kernel pw;
  pw.name = "pointwise_epi";
  ir::IRBuilder b(pw);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();
  pw.hints.epilogue_dag = ir::EpilogueDAGAttr{{
      {ir::EpilogueOpKind::GELU, 0.0f, 0.0f, -1},
      {ir::EpilogueOpKind::ReLU, 0.0f, 0.0f, -1},
  }};
  const auto pw_rep = passes::FusionOccupancyCostModelPass::analyze(pw, target);
  assert(pw_rep.fused_epilogue_ops == 2u);
  assert(pw_rep.estimated_sram_bytes == 0u);
  assert(pw_rep.estimated_registers_per_thread == 20u);
  assert(pw_rep.arithmetic_intensity > 1.0);
  assert(pw_rep.estimated_occupancy > 0.0 && pw_rep.estimated_occupancy <= 1.0);
  assert(!pass.run(pw, ctx).changed);

  // GroupedGemm and QuantDot cost analysis.
  ir::Kernel gg_k;
  gg_k.name = "grouped_gemm_cost";
  ir::IRBuilder gb(gg_k);
  const auto ptr_f32 = ir::Type::ptr(ir::ScalarDType::F32);
  const auto ptr_i32 = ir::Type::ptr(ir::ScalarDType::I32);
  const auto i32 = ir::Type::scalar(ir::ScalarDType::I32);
  ir::GroupedGemmAttr gg_attr;
  gg_attr.bm = 64;
  gg_attr.bn = 64;
  gg_attr.bk = 16;
  gg_attr.num_simdgroups = 4;
  gb.emit_grouped_gemm({gb.add_param("a", ptr_f32), gb.add_param("w", ptr_f32), gb.add_param("off", ptr_i32),
                        gb.add_param("out", ptr_f32, false, 0, 1, true), gb.add_param("E", i32),
                        gb.add_param("N", i32), gb.add_param("K", i32)},
                       gg_attr);
  gb.emit_return();
  const auto gg_rep = passes::FusionOccupancyCostModelPass::analyze(gg_k, target);
  assert(gg_rep.estimated_sram_bytes == 9472u);
  assert(gg_rep.threads_per_threadgroup == 128u);
  assert(gg_rep.max_resident_tgs_per_core == 3u);
  assert(std::fabs(gg_rep.arithmetic_intensity - 16.0) < 1e-6);

  ir::Kernel qd_k;
  qd_k.name = "quant_dot_cost";
  ir::IRBuilder qb(qd_k);
  const auto ptr_f16 = ir::Type::ptr(ir::ScalarDType::F16);
  ir::QuantDotAttr qd_attr;
  qd_attr.bm = 64;
  qd_attr.bn = 64;
  qd_attr.bk = 32;
  qd_attr.num_simdgroups = 4;
  qb.emit_quant_dot({qb.add_param("a", ptr_f16), qb.add_param("qw", ptr_i32), qb.add_param("sc", ptr_f16),
                     qb.add_param("zp", ptr_f16), qb.add_param("out", ptr_f16, false, 0, 1, true),
                     qb.add_param("M", i32), qb.add_param("N", i32), qb.add_param("K", i32)},
                    qd_attr);
  qb.emit_return();
  const auto qd_rep = passes::FusionOccupancyCostModelPass::analyze(qd_k, target);
  assert(qd_rep.estimated_sram_bytes == 9728u);
  assert(qd_rep.threads_per_threadgroup == 128u);
  assert(std::fabs(qd_rep.arithmetic_intensity - 32.0) < 1e-6);
}

}  // namespace

int main() {
  test_ir_roundtrip_and_verifier();
  test_algebraic_norm_folding_src_dst();
  test_post_dot_ssa_auto_fusion();
  test_shared_msl_epilogue_emission();
  test_fusion_cost_model_pso_simdgroup_clamp();
  test_fusion_cost_model_double_buffer_downgrade();
  test_fusion_cost_model_single_buffer_sram_clamp();
  test_fusion_cost_model_deep_epilogue_reg_pressure();
  test_fusion_cost_model_preserves_well_sized_tiles_and_pointwise();
  std::cout << "test_v2_fusion_passes: ALL PASSED\n";
  return 0;
}
