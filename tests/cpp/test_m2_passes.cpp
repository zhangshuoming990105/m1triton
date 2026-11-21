#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ir/attrs.hpp"
#include "ir/builder.hpp"
#include "ir/ir.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/types.hpp"
#include "ir/verifier.hpp"
#include "passes/pass_manager.hpp"
#include "passes/simdgroup_matmul.hpp"
#include "passes/structure_pruning.hpp"
#include "passes/threadgroup_staging.hpp"
#include "passes/tile_layout.hpp"

using namespace m1triton::ir;
using namespace m1triton::passes;

namespace {

std::unique_ptr<Module> parse_or_die(const std::string& text) {
  auto parsed = IRParser::parse(text);
  if (!parsed.ok || !parsed.module) {
    std::cerr << "IR parse failed: " << parsed.error << "\n--- IR ---\n" << text << "\n";
    assert(false);
  }
  return std::move(parsed.module);
}

void assert_ir_roundtrip(const Module& mod, const TargetSpec& target, VerifyStage stage) {
  const std::string printed1 = IRPrinter::print(mod);
  const auto parsed = IRParser::parse(printed1);
  if (!parsed.ok || !parsed.module) {
    std::cerr << "Roundtrip parse failed: " << parsed.error << "\n--- IR ---\n" << printed1 << "\n";
    assert(false);
  }
  const auto vr = IRVerifier::verify(*parsed.module, target, stage);
  if (!vr.ok) {
    std::cerr << "Roundtrip verify failed:\n" << vr.summary() << "\n--- IR ---\n" << printed1 << "\n";
    assert(false);
  }
  const std::string printed2 = IRPrinter::print(*parsed.module);
  if (printed1 != printed2) {
    std::cerr << "Roundtrip mismatch!\n--- PRINTED 1 ---\n" << printed1 << "\n--- PRINTED 2 ---\n" << printed2 << "\n";
    assert(false);
  }
}

void assert_ir_diff(const std::string& src_ir, const std::string& expected_dst_ir, const PassManager& pm,
                    const TargetSpec& target, VerifyStage final_stage, bool check_idempotent = true) {
  auto mod = parse_or_die(src_ir);
  PassContext ctx{target, true, false, nullptr};
  const auto pr = pm.run(*mod, ctx);
  if (!pr.ok) {
    std::cerr << "PassManager failed: " << pr.diagnostic << "\n--- SRC IR ---\n" << src_ir << "\n";
    assert(false);
  }
  const std::string actual = IRPrinter::print(*mod);
  if (actual != expected_dst_ir) {
    std::cerr << "IR diff mismatch!\n--- EXPECTED ---\n"
              << expected_dst_ir << "--- ACTUAL ---\n"
              << actual << "\n";
    assert(false);
  }
  assert_ir_roundtrip(*mod, target, final_stage);

  if (check_idempotent) {
    const auto pr2 = pm.run(*mod, ctx);
    assert(pr2.ok && !pr2.changed);
    assert(IRPrinter::print(*mod) == expected_dst_ir);
  }
}

void test_structure_pruning_upper_triangular() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<StructureAwareDeadTilePruningPass>();

  const std::string src_ir = R"(kernel @triu_gemm(%a: tile<64x16, f32>, %b: tile<16x64, f32>) hints {structure=upper_triangular} {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";

  const std::string expected_dst_ir =
      R"(kernel @triu_gemm(%a: tile<64x16, f32>, %b: tile<16x64, f32>) hints {structure=upper_triangular} {
  %0 = dot %a, %b {acc=f32, lhs_struct=upper_triangular, rhs_struct=upper_triangular, out_struct=upper_triangular} : tile<64x64, f32, struct=upper_triangular>
  return
}
)";

  assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  std::cout << "[PASS] test_structure_pruning_upper_triangular\n";
}

void test_structure_pruning_lower_triangular() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<StructureAwareDeadTilePruningPass>();

  const std::string src_ir =
      R"(kernel @tril_gemm(%a: tile<64x16, f32, struct=lower_triangular>, %b: tile<16x64, f32, struct=lower_triangular>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";

  const std::string expected_dst_ir =
      R"(kernel @tril_gemm(%a: tile<64x16, f32, struct=lower_triangular>, %b: tile<16x64, f32, struct=lower_triangular>) {
  %0 = dot %a, %b {acc=f32, lhs_struct=lower_triangular, rhs_struct=lower_triangular, out_struct=lower_triangular} : tile<64x64, f32, struct=lower_triangular>
  return
}
)";

  assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  std::cout << "[PASS] test_structure_pruning_lower_triangular\n";
}

void test_structure_pruning_symmetric_invariant() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<StructureAwareDeadTilePruningPass>();

  // Strict negative test: Symmetric(A) @ Symmetric(B) with A != B MUST NOT set out_struct=symmetric!
  {
    const std::string src_ir =
        R"(kernel @sym_ab(%a: tile<64x64, f32, struct=symmetric>, %b: tile<64x64, f32, struct=symmetric>) hints {structure=symmetric} {
  %0 = dot %a, %b {acc=f32} hints {structure=symmetric} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @sym_ab(%a: tile<64x64, f32, struct=symmetric>, %b: tile<64x64, f32, struct=symmetric>) hints {structure=symmetric} {
  %0 = dot %a, %b {acc=f32, lhs_struct=symmetric, rhs_struct=symmetric} hints {structure=symmetric} : tile<64x64, f32>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  // Positive test: Symmetric(A) @ Symmetric(A) with lhs == rhs DOES propagate out_struct=symmetric!
  {
    const std::string src_ir = R"(kernel @sym_aa(%a: tile<64x64, f32, struct=symmetric>) {
  %0 = dot %a, %a {acc=f32} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir = R"(kernel @sym_aa(%a: tile<64x64, f32, struct=symmetric>) {
  %0 = dot %a, %a {acc=f32, lhs_struct=symmetric, rhs_struct=symmetric, out_struct=symmetric} : tile<64x64, f32, struct=symmetric>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  // Explicit out_struct=symmetric on the Dot op itself is preserved and synced onto result TileType.
  {
    const std::string src_ir =
        R"(kernel @sym_explicit(%a: tile<64x64, f32, struct=symmetric>, %b: tile<64x64, f32, struct=symmetric>) {
  %0 = dot %a, %b {acc=f32, out_struct=symmetric} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @sym_explicit(%a: tile<64x64, f32, struct=symmetric>, %b: tile<64x64, f32, struct=symmetric>) {
  %0 = dot %a, %b {acc=f32, lhs_struct=symmetric, rhs_struct=symmetric, out_struct=symmetric} : tile<64x64, f32, struct=symmetric>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  std::cout << "[PASS] test_structure_pruning_symmetric_invariant\n";
}

void test_structure_pruning_diagonal_and_transpose_and_negatives() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<StructureAwareDeadTilePruningPass>();

  // Diagonal LHS * Dense RHS -> lhs_struct=diagonal, out_struct remains Dense.
  {
    const std::string src_ir = R"(kernel @diag_lhs(%a: tile<64x64, f32, struct=diagonal>, %b: tile<64x64, f32>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @diag_lhs(%a: tile<64x64, f32, struct=diagonal>, %b: tile<64x64, f32>) {
  %0 = dot %a, %b {acc=f32, lhs_struct=diagonal} : tile<64x64, f32>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  // Diagonal * Diagonal -> out_struct=diagonal.
  {
    const std::string src_ir =
        R"(kernel @diag_both(%a: tile<64x64, f32, struct=diagonal>, %b: tile<64x64, f32, struct=diagonal>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @diag_both(%a: tile<64x64, f32, struct=diagonal>, %b: tile<64x64, f32, struct=diagonal>) {
  %0 = dot %a, %b {acc=f32, lhs_struct=diagonal, rhs_struct=diagonal, out_struct=diagonal} : tile<64x64, f32, struct=diagonal>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  // Transpose-aware closure: upper_triangular^T (= lower_triangular) @ lower_triangular -> lower_triangular.
  {
    const std::string src_ir =
        R"(kernel @triu_t_tril(%a: tile<16x64, f32, struct=upper_triangular>, %b: tile<16x64, f32, struct=lower_triangular>) {
  %0 = dot %a, %b {acc=f32, ta=true} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @triu_t_tril(%a: tile<16x64, f32, struct=upper_triangular>, %b: tile<16x64, f32, struct=lower_triangular>) {
  %0 = dot %a, %b {acc=f32, ta=true, lhs_struct=upper_triangular, rhs_struct=lower_triangular, out_struct=lower_triangular} : tile<64x64, f32, struct=lower_triangular>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::HighLevel);
  }

  // Negative 1: invalid diagonal metadata (unit=true with off!=0).
  {
    const std::string bad_ir =
        R"(kernel @bad_diag(%a: tile<64x64, f32, struct=diagonal(off=1, unit=true)>, %b: tile<64x64, f32>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassContext ctx{target, true, false, nullptr};
    const auto pr = pm.run(*mod, ctx);
    assert(!pr.ok);
  }

  // Negative 2: conflicting structure between DotAttr and operand TileType.
  {
    const std::string bad_ir =
        R"(kernel @conflict_struct(%a: tile<64x64, f32, struct=lower_triangular>, %b: tile<64x64, f32>) {
  %0 = dot %a, %b {acc=f32, lhs_struct=upper_triangular} : tile<64x64, f32>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassContext ctx{target, true, false, nullptr};
    const auto pr = pm.run(*mod, ctx);
    assert(!pr.ok);
  }

  std::cout << "[PASS] test_structure_pruning_diagonal_and_transpose_and_negatives\n";
}

void test_tile_layout_sgmm_and_fallbacks_and_reduce() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<TileAndLayoutAssignmentPass>();

  // 1. 64x64x16 -> 4 SGs (sgrid=2x2, ftiles=4x4, tg=128x1x1)
  {
    const std::string src_ir = R"(kernel @sgmm_64x64(%a: tile<64x16, f32>, %b: tile<16x64, f32>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @sgmm_64x64(%a: tile<64x16, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1)>, %b: tile<16x64, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::PreCodegen);
  }

  // 2. 128x64x16 with TileSpecAttr(simds=8) -> 8 SGs (sgrid=4x2, ftiles=4x4, tg=256x1x1)
  {
    const std::string src_ir =
        R"(kernel @sgmm_128x64(%a: tile<128x16, f32>, %b: tile<16x64, f32>) hints {tile_spec(bm=128, bn=64, bk=16, simds=8, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32>
  return
}
)";

    const std::string expected_dst_ir =
        R"(kernel @sgmm_128x64(%a: tile<128x16, f32, layout=blocked(vec=1, align=4, contig=1, tg=256x1x1, ept=128x1)>, %b: tile<16x64, f32, layout=blocked(vec=1, align=4, contig=1, tg=256x1x1, ept=16x1)>) hints {tile_spec(bm=128, bn=64, bk=16, simds=8, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=256x1x1, ept=8x4, ftiles=4x4, sgrid=4x2)>
  return
}
)";

    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::PreCodegen);
  }

  // 3. Non-SGMM small fallback: 16x16x16 -> Blocked tg=8x8x1; 24x24x16 -> Blocked tg=16x16x1
  {
    const std::string src_16 = R"(kernel @dot_16(%a: tile<16x16, f32>, %b: tile<16x16, f32>) {
  %0 = dot %a, %b {acc=f32} : tile<16x16, f32>
  return
}
)";
    auto mod16 = parse_or_die(src_16);
    PassContext ctx{target, true, false, nullptr};
    assert(pm.run(*mod16, ctx).ok);
    const auto& l16 = mod16->kernels[0]->value(mod16->kernels[0]->op(0).results[0]).type.as_tile().layout;
    assert(l16.kind == LayoutKind::Blocked);
    assert(l16.tg_threads[0] == 8 && l16.tg_threads[1] == 8 && l16.tg_threads[2] == 1);

    const std::string src_24 = R"(kernel @dot_24(%a: tile<24x16, f32>, %b: tile<16x24, f32>) {
  %0 = dot %a, %b {acc=f32} : tile<24x24, f32>
  return
}
)";
    auto mod24 = parse_or_die(src_24);
    assert(pm.run(*mod24, ctx).ok);
    const auto& l24 = mod24->kernels[0]->value(mod24->kernels[0]->op(0).results[0]).type.as_tile().layout;
    assert(l24.kind == LayoutKind::Blocked);
    assert(l24.tg_threads[0] == 16 && l24.tg_threads[1] == 16 && l24.tg_threads[2] == 1);
  }

  // 4. TargetSpec thread limit clamping: max_threads_per_threadgroup=128 clamps simds=8 down to 4 SGs (2x2)
  {
    TargetSpec small_target = TargetSpec::apple_m1_max();
    small_target.max_threads_per_threadgroup = 128;
    const std::string src_ir =
        R"(kernel @sgmm_clamped(%a: tile<128x16, f32>, %b: tile<16x64, f32>) hints {tile_spec(bm=128, bn=64, bk=16, simds=8, dbl=false, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32>
  return
}
)";
    auto mod = parse_or_die(src_ir);
    PassContext ctx{small_target, true, false, nullptr};
    assert(pm.run(*mod, ctx).ok);
    const auto& l = mod->kernels[0]->value(mod->kernels[0]->op(0).results[0]).type.as_tile().layout;
    assert(l.kind == LayoutKind::SimdgroupFragment);
    assert(l.tg_threads[0] == 128 && l.tg_threads[1] == 1 && l.tg_threads[2] == 1);
    assert(l.simd_grid[0] == 2 && l.simd_grid[1] == 2);
    assert(l.frag_tiles_m == 8 && l.frag_tiles_n == 4);
  }

  // 5. 1D & 2D Row Reduce tile assignment
  {
    const std::string src_1d = R"(kernel @reduce_1d(%x: tile<1024, f32>) {
  %0 = reduce %x {kind=sum, mode=value, axis=0} : f32
  return
}
)";
    const std::string dst_1d =
        R"(kernel @reduce_1d(%x: tile<1024, f32, layout=blocked(vec=1, align=4, contig=1, tg=256x1x1, ept=4x1)>) {
  %0 = reduce %x {kind=sum, mode=value, axis=0} : f32
  return
}
)";
    assert_ir_diff(src_1d, dst_1d, pm, target, VerifyStage::PreCodegen);

    const std::string src_row = R"(kernel @reduce_row(%x: tile<4x128, f32>) {
  %0 = reduce %x {kind=sum, mode=value, axis=1} : tile<4, f32>
  return
}
)";
    const std::string dst_row =
        R"(kernel @reduce_row(%x: tile<4x128, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=4x1)>) {
  %0 = reduce %x {kind=sum, mode=value, axis=1} : tile<4, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=1x1)>
  return
}
)";
    assert_ir_diff(src_row, dst_row, pm, target, VerifyStage::PreCodegen);

    // Small reduce (BLOCK_K=16) clamps up to simd_width=32.
    const std::string src_small = R"(kernel @reduce_small(%x: tile<16, f32>) {
  %0 = reduce %x {kind=max, mode=value, axis=0} : f32
  return
}
)";
    auto mod_small = parse_or_die(src_small);
    PassContext ctx{target, true, false, nullptr};
    assert(pm.run(*mod_small, ctx).ok);
    const auto& ls = mod_small->kernels[0]->value(0).type.as_tile().layout;
    assert(ls.tg_threads[0] == 32 && ls.tg_threads[1] == 1 && ls.tg_threads[2] == 1);
  }

  std::cout << "[PASS] test_tile_layout_sgmm_and_fallbacks_and_reduce\n";
}

void test_threadgroup_staging_and_sram_downgrade() {
  const auto target = TargetSpec::apple_m1_max();
  PassManager pm;
  pm.add_pass<TileAndLayoutAssignmentPass>();
  pm.add_pass<ThreadgroupStagingAndSwizzlePass>();

  // 1. F32 64x64x16 double-buffered: 2 * ((64*20 + 16*68) * 4) = 18944 B <= 32768 B -> stages=2, pad4, pad=4
  {
    assert(ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(
               64, 64, 16, ScalarDType::F32, ScalarDType::F32, 4, 4, 2) == 18944);

    const std::string src_ir =
        R"(kernel @stage_f32_dbl(%a: tile<64x16, f32>, %b: tile<16x64, f32>) hints {tile_spec(bm=64, bn=64, bk=16, simds=4, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";
    const std::string expected_dst_ir =
        R"(kernel @stage_f32_dbl(%a: tile<64x16, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1, swizzle=pad4, pad=4, stages=2)>, %b: tile<16x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1, swizzle=pad4, pad=4, stages=2)>) hints {tile_spec(bm=64, bn=64, bk=16, simds=4, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::PreCodegen, /*check_idempotent=*/false);
  }

  // 2. F16 64x64x16 double-buffered: 2 * ((64*24 + 16*72) * 2) = 10752 B <= 32768 B -> stages=2, pad8, pad=8
  {
    assert(ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(
               64, 64, 16, ScalarDType::F16, ScalarDType::F16, 8, 8, 2) == 10752);

    const std::string src_ir =
        R"(kernel @stage_f16_dbl(%a: tile<64x16, f16>, %b: tile<16x64, f16>) hints {tile_spec(bm=64, bn=64, bk=16, simds=4, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32>
  return
}
)";
    const std::string expected_dst_ir =
        R"(kernel @stage_f16_dbl(%a: tile<64x16, f16, layout=tg_staged(vec=1, align=2, contig=1, tg=128x1x1, ept=64x1, swizzle=pad8, pad=8, stages=2)>, %b: tile<16x64, f16, layout=tg_staged(vec=1, align=2, contig=1, tg=128x1x1, ept=16x1, swizzle=pad8, pad=8, stages=2)>) hints {tile_spec(bm=64, bn=64, bk=16, simds=4, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::PreCodegen, /*check_idempotent=*/false);
  }

  // 3. 32 KB SRAM auto-downgrade: 128x64x32 F32 with double_buffer=true!
  // 1-stage = (128*36 + 32*68)*4 = 27136 B <= 32768 B; 2-stage = 54272 B > 32768 B -> auto-downgrade stages=1!
  {
    assert(ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(
               128, 64, 32, ScalarDType::F32, ScalarDType::F32, 4, 4, 1) == 27136);
    assert(ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(
               128, 64, 32, ScalarDType::F32, ScalarDType::F32, 4, 4, 2) == 54272);

    const std::string src_ir =
        R"(kernel @stage_downgrade(%a: tile<128x32, f32>, %b: tile<32x64, f32>) hints {tile_spec(bm=128, bn=64, bk=32, simds=8, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32>
  return
}
)";
    // Note: IRPrinter omits `stages=1` when stages == 1 (default).
    const std::string expected_dst_ir =
        R"(kernel @stage_downgrade(%a: tile<128x32, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=256x1x1, ept=128x1, swizzle=pad4, pad=4)>, %b: tile<32x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=256x1x1, ept=32x1, swizzle=pad4, pad=4)>) hints {tile_spec(bm=128, bn=64, bk=32, simds=8, dbl=true, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=256x1x1, ept=8x4, ftiles=4x4, sgrid=4x2)>
  return
}
)";
    assert_ir_diff(src_ir, expected_dst_ir, pm, target, VerifyStage::PreCodegen, /*check_idempotent=*/false);

    // Also confirm standalone ThreadgroupStagingAndSwizzlePass is idempotent after downgrade.
    auto mod = parse_or_die(expected_dst_ir);
    PassManager stage_only;
    stage_only.add_pass<ThreadgroupStagingAndSwizzlePass>();
    PassContext ctx{target, true, false, nullptr};
    const auto pr = stage_only.run(*mod, ctx);
    assert(pr.ok && !pr.changed);
    assert(mod->kernels[0]->value(0).type.as_tile().layout.stages == 1);
    assert(mod->kernels[0]->value(1).type.as_tile().layout.stages == 1);
  }

  // 4. Negative: 128x128x64 F32 exceeds 32 KB SRAM even at stages=1 -> fails cleanly!
  {
    const std::string overflow_ir =
        R"(kernel @stage_overflow(%a: tile<128x64, f32>, %b: tile<64x128, f32>) hints {tile_spec(bm=128, bn=128, bk=64, simds=8, dbl=false, split_k=1)} {
  %0 = dot %a, %b {acc=f32} : tile<128x128, f32>
  return
}
)";
    auto mod = parse_or_die(overflow_ir);
    PassContext ctx{target, true, false, nullptr};
    const auto pr = pm.run(*mod, ctx);
    assert(!pr.ok);
    assert(pr.diagnostic.find("32768") != std::string::npos);
  }

  std::cout << "[PASS] test_threadgroup_staging_and_sram_downgrade\n";
}

void test_simdgroup_matmul_validation() {
  const auto target = TargetSpec::apple_m1_max();
  PassContext ctx{target, true, false, nullptr};

  // Positive: valid staged SGMM passes SimdgroupMatrixLoweringPass + compute_sgmm_attr matches.
  {
    const std::string valid_ir =
        R"(kernel @valid_sgmm(%a: tile<64x16, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1, swizzle=pad4, pad=4, stages=2)>, %b: tile<16x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1, swizzle=pad4, pad=4, stages=2)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    auto mod = parse_or_die(valid_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    const auto pr = pm.run(*mod, ctx);
    assert(pr.ok && !pr.changed);
    const auto attr = SimdgroupMatrixLoweringPass::compute_sgmm_attr(*mod->kernels[0], mod->kernels[0]->op(0), target);
    assert(attr.has_value());
    assert(attr->m_tiles == 4 && attr->n_tiles == 4 && attr->k_step == 8);
    assert(attr->ldm_a == 20 && attr->ldm_b == 68 && attr->acc_dtype == ScalarDType::F32);
  }

  // Negative 1: SimdgroupFragment result with non-staged (Blocked) operands rejected.
  {
    const std::string bad_ir =
        R"(kernel @missing_staging(%a: tile<64x16, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1)>, %b: tile<16x64, f32, layout=blocked(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    assert(!pm.run(*mod, ctx).ok);
  }

  // Negative 2: Mismatched frag_tiles (2x2 instead of 4x4 for 64x64 with sgrid=2x2) rejected.
  {
    const std::string bad_ir =
        R"(kernel @bad_ftiles(%a: tile<64x16, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1, swizzle=pad4, pad=4)>, %b: tile<16x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1, swizzle=pad4, pad=4)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=4x2, ftiles=2x2, sgrid=2x2)>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    assert(!pm.run(*mod, ctx).ok);
  }

  // Negative 3: Swizzle/pad mismatch (swizzle=pad4 with pad=8) rejected.
  {
    const std::string bad_ir =
        R"(kernel @bad_pad(%a: tile<64x16, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1, swizzle=pad4, pad=8)>, %b: tile<16x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=16x1, swizzle=pad4, pad=4)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    assert(!pm.run(*mod, ctx).ok);
  }

  // Negative 4: Forged 2-stage 128x64x32 exceeding 32 KB SRAM rejected.
  {
    const std::string bad_ir =
        R"(kernel @forged_sram(%a: tile<128x32, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=256x1x1, ept=128x1, swizzle=pad4, pad=4, stages=2)>, %b: tile<32x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=256x1x1, ept=32x1, swizzle=pad4, pad=4, stages=2)>) {
  %0 = dot %a, %b {acc=f32} : tile<128x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=256x1x1, ept=8x4, ftiles=4x4, sgrid=4x2)>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    const auto pr = pm.run(*mod, ctx);
    assert(!pr.ok);
    assert(pr.diagnostic.find("32768") != std::string::npos);
  }

  // Negative 5: Custom TargetSpec with sgmm_k=16 rejects BK=8.
  {
    TargetSpec custom = TargetSpec::apple_m1_max();
    custom.sgmm_k = 16;
    const std::string bad_ir =
        R"(kernel @bad_k(%a: tile<64x8, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=64x1, swizzle=pad4, pad=4)>, %b: tile<8x64, f32, layout=tg_staged(vec=1, align=4, contig=1, tg=128x1x1, ept=8x1, swizzle=pad4, pad=4)>) {
  %0 = dot %a, %b {acc=f32} : tile<64x64, f32, layout=simd_frag(vec=1, align=4, contig=1, tg=128x1x1, ept=8x4, ftiles=4x4, sgrid=2x2)>
  return
}
)";
    auto mod = parse_or_die(bad_ir);
    PassManager pm;
    pm.add_pass<SimdgroupMatrixLoweringPass>();
    PassContext cctx{custom, true, false, nullptr};
    assert(!pm.run(*mod, cctx).ok);
  }

  std::cout << "[PASS] test_simdgroup_matmul_validation\n";
}

void test_full_m2_and_m1_pipelines_e2e() {
  const auto target = TargetSpec::apple_m1_max();
  PassContext ctx{target, true, false, nullptr};

  // 1. Full M2 pipeline on UpperTriangular 64x64x16 with double_buffer=true inside ForLoop.
  {
    Module mod;
    auto& k = mod.add_kernel("triu_sgmm_loop");
    k.hints.structure = StructureAttr{MatrixStructure::UpperTriangular, 0, false};
    k.hints.tile_spec = TileSpecAttr{64, 64, 16, 4, true, 1, 8};

    IRBuilder b(k);
    const ValueId a_ptr = b.add_param("a_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, false);
    const ValueId b_ptr = b.add_param("b_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, false);
    const ValueId c_ptr = b.add_param("c_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, true);
    const ValueId n = b.add_param("N", Type::scalar(ScalarDType::I32), false, 0, 16, false);

    const ValueId pid_m = b.emit_program_id(0);
    const ValueId pid_n = b.emit_program_id(1);
    const ValueId c64 = b.emit_const_i32(64);
    const ValueId c16 = b.emit_const_i32(16);
    const ValueId c0 = b.emit_const_i32(0);

    const ValueId offs_m = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, pid_m, c64), b.emit_arange(0, 64));
    const ValueId offs_n = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, pid_n, c64), b.emit_arange(0, 64));
    const ValueId offs_k = b.emit_arange(0, 16);

    const ValueId om2 = b.emit_expand_dims(offs_m, 1);
    const ValueId on2 = b.emit_expand_dims(offs_n, 0);
    const ValueId ok_row = b.emit_expand_dims(offs_k, 0);
    const ValueId ok_col = b.emit_expand_dims(offs_k, 1);

    const ValueId a_ptrs_init = b.emit_ptr_add(a_ptr, b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, om2, n), ok_row));
    const ValueId b_ptrs_init = b.emit_ptr_add(b_ptr, b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, ok_col, n), on2));
    const ValueId acc_init = b.emit_splat(b.emit_const_f32(0.0f), {64, 64});

    const BlockId entry = b.current_block();
    const auto loop = b.begin_for_loop(c0, n, c16, {acc_init, a_ptrs_init, b_ptrs_init});
    const ValueId b_step = b.emit_binary(BinaryOp::Mul, c16, n);
    const ValueId a_tile = b.emit_load(loop.iter_args[1]);
    const ValueId b_tile = b.emit_load(loop.iter_args[2]);
    const ValueId acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], DotAttr{});
    b.end_for_loop(loop, {acc_next, b.emit_ptr_add(loop.iter_args[1], c16), b.emit_ptr_add(loop.iter_args[2], b_step)}, entry);

    const ValueId c_ptrs = b.emit_ptr_add(c_ptr, b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, om2, n), on2));
    b.emit_store(c_ptrs, loop.results[0]);
    b.emit_return();

    // Also run via build_m1_pipeline() to verify tile_spec-hinted kernels get full M2 SGMM + staging + structure!
    const PassManager pm = PassManager::build_m1_pipeline();
    const auto pr = pm.run(k, ctx);
    if (!pr.ok) std::cerr << pr.diagnostic << "\n";
    assert(pr.ok && pr.changed);
    assert_ir_roundtrip(mod, target, VerifyStage::PreCodegen);

    bool found_dot = false;
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Dot) {
        found_dot = true;
        const auto& da = std::get<DotAttr>(op->attr);
        assert(da.lhs_struct.kind == MatrixStructure::UpperTriangular);
        assert(da.rhs_struct.kind == MatrixStructure::UpperTriangular);
        assert(da.out_struct.kind == MatrixStructure::UpperTriangular);

        const auto& out_tile = k.value(op->results[0]).type.as_tile();
        assert(out_tile.structure.kind == MatrixStructure::UpperTriangular);
        assert(out_tile.layout.kind == LayoutKind::SimdgroupFragment);
        assert(out_tile.layout.frag_role == FragRole::Accumulator);
        assert(out_tile.layout.simd_grid[0] == 2 && out_tile.layout.simd_grid[1] == 2);
        assert(out_tile.layout.frag_tiles_m == 4 && out_tile.layout.frag_tiles_n == 4);
        assert(out_tile.layout.tg_threads[0] == 128 && out_tile.layout.tg_threads[1] == 1 &&
               out_tile.layout.tg_threads[2] == 1);

        const auto& lhs_l = k.value(op->operands[0]).type.as_tile().layout;
        const auto& rhs_l = k.value(op->operands[1]).type.as_tile().layout;
        assert(lhs_l.kind == LayoutKind::ThreadgroupStaged && rhs_l.kind == LayoutKind::ThreadgroupStaged);
        assert(lhs_l.swizzle == SwizzleMode::PadBank4 && rhs_l.swizzle == SwizzleMode::PadBank4);
        assert(lhs_l.pad_elems == 4 && rhs_l.pad_elems == 4);
        assert(lhs_l.stages == 2 && rhs_l.stages == 2);
        assert(lhs_l.vec_width == 4 && rhs_l.vec_width == 4);
      }
    }
    assert(found_dot);
  }

  // 2. Full M2 pipeline on unhinted 3D batched transposed Dot (live via Store through DCE).
  {
    const std::string src_ir =
        R"(kernel @m2_auto_and_batched(%a: tile<4x64x16, f32>, %b: tile<4x64x16, f32>, %out: tile<4x64x64, ptr<f32, device>>) {
  %0 = dot %a, %b {acc=f32, tb=true} : tile<4x64x64, f32>
  store %out, %0 {align=4, vec=1}
  return
}
)";
    auto mod = parse_or_die(src_ir);
    const PassManager pm = PassManager::build_m2_pipeline();
    const auto pr = pm.run(*mod, ctx);
    if (!pr.ok) std::cerr << pr.diagnostic << "\n";
    assert(pr.ok && pr.changed);
    assert_ir_roundtrip(*mod, target, VerifyStage::PreCodegen);

    const auto& k = *mod->kernels[0];
    bool found_dot = false;
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Dot) {
        found_dot = true;
        const auto& out_l = k.value(op->results[0]).type.as_tile().layout;
        const auto& lhs_l = k.value(op->operands[0]).type.as_tile().layout;
        const auto& rhs_l = k.value(op->operands[1]).type.as_tile().layout;
        assert(out_l.kind == LayoutKind::SimdgroupFragment);
        assert(out_l.simd_grid[0] == 2 && out_l.simd_grid[1] == 2);
        assert(out_l.frag_tiles_m == 4 && out_l.frag_tiles_n == 4);
        assert(out_l.tg_threads[0] == 128 && out_l.tg_threads[1] == 1 && out_l.tg_threads[2] == 1);
        assert(lhs_l.kind == LayoutKind::ThreadgroupStaged && lhs_l.pad_elems == 4 && lhs_l.stages == 1);
        assert(rhs_l.kind == LayoutKind::ThreadgroupStaged && rhs_l.pad_elems == 4 && rhs_l.stages == 1);
      }
    }
    assert(found_dot);
  }

  // 3. Full M2 pipeline on Symmetric(A) @ Symmetric(B) (A != B, two distinct loads): out_struct stays Dense!
  {
    const std::string src_ir =
        R"(kernel @sym_ab_m2(%ap: tile<64x16, ptr<f32, device>>, %bp: tile<16x64, ptr<f32, device>>, %cp: tile<64x64, ptr<f32, device>>) hints {structure=symmetric} {
  %0 = load %ap {align=4, vec=1} : tile<64x16, f32, struct=symmetric>
  %1 = load %bp {align=4, vec=1} : tile<16x64, f32, struct=symmetric>
  %2 = dot %0, %1 {acc=f32} : tile<64x64, f32>
  store %cp, %2 {align=4, vec=1}
  return
}
)";
    auto mod = parse_or_die(src_ir);
    const PassManager pm = PassManager::build_m2_pipeline();
    const auto pr = pm.run(*mod, ctx);
    if (!pr.ok) std::cerr << pr.diagnostic << "\n";
    assert(pr.ok && pr.changed);
    assert_ir_roundtrip(*mod, target, VerifyStage::PreCodegen);

    const auto& k = *mod->kernels[0];
    bool found_dot = false;
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Dot) {
        found_dot = true;
        const auto& da = std::get<DotAttr>(op->attr);
        assert(da.lhs_struct.kind == MatrixStructure::Symmetric);
        assert(da.rhs_struct.kind == MatrixStructure::Symmetric);
        assert(da.out_struct.kind == MatrixStructure::Dense);
        assert(k.value(op->results[0]).type.as_tile().structure.kind == MatrixStructure::Dense);
        assert(k.value(op->results[0]).type.as_tile().layout.kind == LayoutKind::SimdgroupFragment);
      }
    }
    assert(found_dot);
  }

  std::cout << "[PASS] test_full_m2_and_m1_pipelines_e2e\n";
}

}  // namespace

int main() {
  test_structure_pruning_upper_triangular();
  test_structure_pruning_lower_triangular();
  test_structure_pruning_symmetric_invariant();
  test_structure_pruning_diagonal_and_transpose_and_negatives();
  test_tile_layout_sgmm_and_fallbacks_and_reduce();
  test_threadgroup_staging_and_sram_downgrade();
  test_simdgroup_matmul_validation();
  test_full_m2_and_m1_pipelines_e2e();
  std::cout << "ALL M2 PASS TESTS PASSED!\n";
  return 0;
}
