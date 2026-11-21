#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "codegen/metal/reduce_scan_emitter.hpp"
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

void verify_and_opt(ir::Kernel& k) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  auto pm = passes::PassManager::build_m1_pipeline();
  passes::PassContext ctx{target};
  pm.run(k, ctx);
  auto vr2 = ir::IRVerifier::verify(k, target);
  if (!vr2.ok) {
    std::cerr << "Post-pass verify error: " << vr2.summary() << "\n";
    assert(false);
  }
}

void test_non_reduce_returns_unmatched() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("vector_add");
  ir::IRBuilder b(k);
  constexpr int64_t kBlock = 1024;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto n = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
  auto pid = b.emit_program_id(0);
  auto offs = b.emit_binary(ir::BinaryOp::Add, b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(kBlock)),
                            b.emit_arange(0, kBlock));
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, n);
  auto zero = b.emit_const_f32(0.0f);
  auto x = b.emit_load(b.emit_ptr_add(x_ptr, offs), mask, zero);
  auto y = b.emit_load(b.emit_ptr_add(y_ptr, offs), mask, zero);
  b.emit_store(b.emit_ptr_add(out_ptr, offs), b.emit_binary(ir::BinaryOp::Add, x, y), mask);
  b.emit_return();

  bool matched = true;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched);
  assert(!res.ok);
  assert(res.msl_source.empty());
  assert(res.error.empty());
}

void test_gemv_r4_float4_peeled() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("gemv_r4");
  ir::IRBuilder b(k);

  constexpr int64_t kBlockK = 1024;
  auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_am = b.add_param("stride_am", ir::Type::scalar(ir::ScalarDType::I64));
  auto bk_c = b.add_param("BLOCK_K", ir::Type::scalar(ir::ScalarDType::I32), true, kBlockK);

  auto pid = b.emit_program_id(0);
  auto base_row = b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(4));

  ir::ValueId rows[4];
  ir::ValueId row_ok[4];
  ir::ValueId row_offs[4];
  for (int r = 0; r < 4; ++r) {
    rows[r] = (r == 0) ? base_row : b.emit_binary(ir::BinaryOp::Add, base_row, b.emit_const_i32(r));
    row_ok[r] = b.emit_binary(ir::BinaryOp::CmpLt, rows[r], m_dim);
    row_offs[r] = b.emit_binary(ir::BinaryOp::Mul, b.emit_cast(rows[r], ir::ScalarDType::I64), stride_am);
  }

  auto zero_f = b.emit_const_f32(0.0f);
  auto acc_init = b.emit_splat(zero_f, {kBlockK});
  auto ar_k = b.emit_arange(0, kBlockK);

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, bk_c, {acc_init, acc_init, acc_init, acc_init});
  auto k_offs = b.emit_binary(ir::BinaryOp::Add, loop.iv, ar_k);
  auto k_mask = b.emit_binary(ir::BinaryOp::CmpLt, k_offs, k_dim);
  auto b_vec = b.emit_load(b.emit_ptr_add(b_ptr, k_offs), k_mask, zero_f);

  std::vector<ir::ValueId> yields;
  for (int r = 0; r < 4; ++r) {
    auto a_idx = b.emit_binary(ir::BinaryOp::Add, row_offs[r], b.emit_cast(k_offs, ir::ScalarDType::I64));
    auto a_mask = b.emit_binary(ir::BinaryOp::And, k_mask, row_ok[r]);
    auto a_vec = b.emit_load(b.emit_ptr_add(a_ptr, a_idx), a_mask, zero_f);
    auto prod = b.emit_binary(ir::BinaryOp::Mul, a_vec, b_vec);
    yields.push_back(b.emit_binary(ir::BinaryOp::Add, loop.iter_args[static_cast<size_t>(r)], prod));
  }
  b.end_for_loop(loop, yields, entry);

  for (int r = 0; r < 4; ++r) {
    auto sum_r = b.emit_reduce(loop.results[static_cast<size_t>(r)],
                               ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
    b.emit_store(b.emit_ptr_add(c_ptr, rows[r]), sum_r, row_ok[r]);
  }
  b.emit_return();

  verify_and_opt(k);

  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  assert(res.ok);
  assert(res.launch_spec.entry_name == "gemv_r4");
  assert(res.launch_spec.tg_x == 256);
  assert(res.launch_spec.tg_y == 1);
  assert(res.launch_spec.vec_width == 4);
  assert(res.launch_spec.param_slots.size() == 6);
  assert(!res.launch_spec.param_slots[0].writable);
  assert(!res.launch_spec.param_slots[1].writable);
  assert(res.launch_spec.param_slots[2].writable);

  assert_contains(res.msl_source, "device const float* a_ptr [[buffer(0)]]");
  assert_contains(res.msl_source, "device const float* b_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "device float* c_ptr [[buffer(2)]]");
  assert_contains(res.msl_source, "threadgroup float m1_smem[8];");
  assert_contains(res.msl_source, "*(const device float4*)");
  assert_contains(res.msl_source, "simd_sum(");
  assert_contains(res.msl_source, "threadgroup_barrier(mem_flags::mem_threadgroup);");
  assert_contains(res.msl_source, "m1_fast_max");
  assert_contains(res.msl_source, "m1_rptr");
  assert_contains(res.msl_source, "(m1_tid_linear == 0u)");
}

void test_gemv_r4_block2048_collapsible() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("gemv_r4_2048");
  ir::IRBuilder b(k);

  constexpr int64_t kBlockK = 2048;
  auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
  auto bk_c = b.add_param("BLOCK_K", ir::Type::scalar(ir::ScalarDType::I32), true, kBlockK);

  auto pid = b.emit_program_id(0);
  auto row0 = b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(2));
  auto row1 = b.emit_binary(ir::BinaryOp::Add, row0, b.emit_const_i32(1));
  auto ok0 = b.emit_binary(ir::BinaryOp::CmpLt, row0, m_dim);
  auto ok1 = b.emit_binary(ir::BinaryOp::CmpLt, row1, m_dim);
  auto k64 = b.emit_cast(k_dim, ir::ScalarDType::I64);
  auto off0 = b.emit_binary(ir::BinaryOp::Mul, b.emit_cast(row0, ir::ScalarDType::I64), k64);
  auto off1 = b.emit_binary(ir::BinaryOp::Mul, b.emit_cast(row1, ir::ScalarDType::I64), k64);

  auto zero_f = b.emit_const_f32(0.0f);
  auto acc_init = b.emit_splat(zero_f, {kBlockK});
  auto ar_k = b.emit_arange(0, kBlockK);

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, bk_c, {acc_init, acc_init});
  auto k_offs = b.emit_binary(ir::BinaryOp::Add, loop.iv, ar_k);
  auto k_mask = b.emit_binary(ir::BinaryOp::CmpLt, k_offs, k_dim);
  auto bv = b.emit_load(b.emit_ptr_add(b_ptr, k_offs), k_mask, zero_f);
  auto a0 = b.emit_load(b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, off0, b.emit_cast(k_offs, ir::ScalarDType::I64))),
                        b.emit_binary(ir::BinaryOp::And, k_mask, ok0), zero_f);
  auto a1 = b.emit_load(b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, off1, b.emit_cast(k_offs, ir::ScalarDType::I64))),
                        b.emit_binary(ir::BinaryOp::And, k_mask, ok1), zero_f);
  auto n0 = b.emit_binary(ir::BinaryOp::Add, loop.iter_args[0], b.emit_binary(ir::BinaryOp::Mul, a0, bv));
  auto n1 = b.emit_binary(ir::BinaryOp::Add, loop.iter_args[1], b.emit_binary(ir::BinaryOp::Mul, a1, bv));
  b.end_for_loop(loop, {n0, n1}, entry);

  auto s0 = b.emit_reduce(loop.results[0], ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
  auto s1 = b.emit_reduce(loop.results[1], ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
  b.emit_store(b.emit_ptr_add(c_ptr, row0), s0, ok0);
  b.emit_store(b.emit_ptr_add(c_ptr, row1), s1, ok1);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.tg_x == 256);
  assert(res.launch_spec.vec_width == 4);
  assert_contains(res.msl_source, "#pragma unroll");
  assert_contains(res.msl_source, "for (uint m1_c = 0u; m1_c < 2u; ++m1_c)");
  assert_contains(res.msl_source, "*(const device float4*)");
  assert_contains(res.msl_source, "simd_sum(");
}

void test_odd_block_k_scalar_fallback() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("row_sum_1003");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 1003;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto n_cols = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));

  auto row = b.emit_program_id(0);
  auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, n_cols);
  auto zero_f = b.emit_const_f32(0.0f);
  auto acc_init = b.emit_splat(zero_f, {kBlock});
  auto ar = b.emit_arange(0, kBlock);

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), n_cols, b.emit_const_i32(kBlock), {acc_init});
  auto cols = b.emit_binary(ir::BinaryOp::Add, loop.iv, ar);
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, cols, n_cols);
  auto xv = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, cols)), mask, zero_f);
  auto next = b.emit_binary(ir::BinaryOp::Add, loop.iter_args[0], xv);
  b.end_for_loop(loop, {next}, entry);

  auto sum_v = b.emit_reduce(loop.results[0], ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
  auto mean_v = b.emit_binary(ir::BinaryOp::Div, sum_v, b.emit_cast(n_cols, ir::ScalarDType::F32));
  b.emit_store(b.emit_ptr_add(out_ptr, row), mean_v);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.vec_width == 1);
  assert(res.launch_spec.tg_x == 256);
  assert_contains(res.msl_source, "m1_lane < 1003u");
  assert_contains(res.msl_source, "simd_sum(");
  assert_contains(res.msl_source, "if ((m1_tid_linear == 0u))");
}

void test_max_min_argmax_argmin() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("row_extrema");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 512;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_max = b.add_param("out_max", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_min = b.add_param("out_min", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_amax = b.add_param("out_amax", ir::Type::ptr(ir::ScalarDType::I32));
  auto out_amin = b.add_param("out_amin", ir::Type::ptr(ir::ScalarDType::I32));
  auto n_cols = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));

  auto row = b.emit_program_id(0);
  auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, n_cols);
  auto offs = b.emit_arange(0, kBlock);
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, n_cols);
  auto ninf = b.emit_const_f32(-std::numeric_limits<float>::infinity());
  auto pinf = b.emit_const_f32(std::numeric_limits<float>::infinity());
  auto x_max = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), mask, ninf);
  auto x_min = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), mask, pinf);

  auto vmax = b.emit_reduce(x_max, ir::ReduceAttr{ir::ReduceKind::Max, ir::ReduceResultMode::ValueOnly, 0, false});
  auto amax = b.emit_reduce(x_max, ir::ReduceAttr{ir::ReduceKind::Argmax, ir::ReduceResultMode::IndexOnly, 0, false});
  auto vmin = b.emit_reduce(x_min, ir::ReduceAttr{ir::ReduceKind::Min, ir::ReduceResultMode::ValueOnly, 0, false});
  auto amin = b.emit_reduce(x_min, ir::ReduceAttr{ir::ReduceKind::Argmin, ir::ReduceResultMode::IndexOnly, 0, false});

  b.emit_store(b.emit_ptr_add(out_max, row), vmax);
  b.emit_store(b.emit_ptr_add(out_min, row), vmin);
  b.emit_store(b.emit_ptr_add(out_amax, row), amax);
  b.emit_store(b.emit_ptr_add(out_amin, row), amin);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.tg_x == 128);
  assert(res.launch_spec.vec_width == 4);
  assert_contains(res.msl_source, "simd_max(");
  assert_contains(res.msl_source, "simd_min(");
  assert_contains(res.msl_source, "simd_shuffle_xor(");
  assert_contains(res.msl_source, "threadgroup float m1_smem[4];");
  assert_contains(res.msl_source, "threadgroup int m1_smem_idx[4];");
  assert_contains(res.msl_source, "(-INFINITY)");
  assert_contains(res.msl_source, "INFINITY");
}

void test_softmax_chained_reduces_vec4() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("row_softmax");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 1024;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto n_cols = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));

  auto row = b.emit_program_id(0);
  auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, n_cols);
  auto offs = b.emit_arange(0, kBlock);
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, n_cols);
  auto ninf = b.emit_const_f32(-std::numeric_limits<float>::infinity());
  auto x = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), mask, ninf);
  auto m = b.emit_reduce(x, ir::ReduceAttr{ir::ReduceKind::Max, ir::ReduceResultMode::ValueOnly, 0, false});
  auto ex = b.emit_unary(ir::UnaryOp::Exp, b.emit_binary(ir::BinaryOp::Sub, x, m));
  auto s = b.emit_reduce(ex, ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
  auto y = b.emit_binary(ir::BinaryOp::Div, ex, s);
  b.emit_store(b.emit_ptr_add(y_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), y, mask);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.tg_x == 256);
  assert(res.launch_spec.vec_width == 4);
  assert_contains(res.msl_source, "*(const device float4*)");
  assert_contains(res.msl_source, "*(device float4*)");
  assert_contains(res.msl_source, "simd_max(");
  assert_contains(res.msl_source, "simd_sum(");
  assert_contains(res.msl_source, "metal::exp(");
}

void test_tiled_argmax_reduce_inside_forloop() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("tiled_argmax");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 1024;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::I32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));

  auto row = b.emit_program_id(0);
  auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, k_dim);
  auto ninf = b.emit_const_f32(-std::numeric_limits<float>::infinity());
  auto z_idx = b.emit_const_i32(0);
  auto ar = b.emit_arange(0, kBlock);

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, b.emit_const_i32(kBlock), {ninf, z_idx});
  auto cols = b.emit_binary(ir::BinaryOp::Add, loop.iv, ar);
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, cols, k_dim);
  auto x = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, cols)), mask, ninf);
  auto tm = b.emit_reduce(x, ir::ReduceAttr{ir::ReduceKind::Max, ir::ReduceResultMode::ValueOnly, 0, false});
  auto ta = b.emit_reduce(x, ir::ReduceAttr{ir::ReduceKind::Argmax, ir::ReduceResultMode::IndexOnly, 0, false});
  auto better = b.emit_binary(ir::BinaryOp::CmpGt, tm, loop.iter_args[0]);
  auto next_v = b.emit_select(better, tm, loop.iter_args[0]);
  auto next_i = b.emit_select(better, b.emit_binary(ir::BinaryOp::Add, loop.iv, ta), loop.iter_args[1]);
  b.end_for_loop(loop, {next_v, next_i}, entry);

  b.emit_store(b.emit_ptr_add(out_ptr, row), loop.results[1]);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.tg_x == 256);
  assert_contains(res.msl_source, "simd_max(");
  assert_contains(res.msl_source, "simd_shuffle_xor(");
}

void test_scan_cumsum_and_cumprod() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("row_cumsum");
    ir::IRBuilder b(k);
    constexpr int64_t kBlock = 256;
    auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto n_cols = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
    auto row = b.emit_program_id(0);
    auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, n_cols);
    auto offs = b.emit_arange(0, kBlock);
    auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, n_cols);
    auto x = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), mask,
                         b.emit_const_f32(0.0f));
    auto y = b.emit_scan(x, ir::ScanAttr{ir::ScanKind::Cumsum, 0, false, false, false});
    b.emit_store(b.emit_ptr_add(y_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), y, mask);
    b.emit_return();

    verify_and_opt(k);
    bool matched = false;
    auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched && res.ok);
    assert(res.launch_spec.tg_x == 256);
    assert_contains(res.msl_source, "simd_prefix_inclusive_sum(");
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("row_cumprod");
    ir::IRBuilder b(k);
    constexpr int64_t kBlock = 128;
    auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto n_cols = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
    auto row = b.emit_program_id(0);
    auto row_off = b.emit_binary(ir::BinaryOp::Mul, row, n_cols);
    auto offs = b.emit_arange(0, kBlock);
    auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, n_cols);
    auto x = b.emit_load(b.emit_ptr_add(x_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), mask,
                         b.emit_const_f32(1.0f));
    auto y = b.emit_scan(x, ir::ScanAttr{ir::ScanKind::Cumprod, 0, false, false, false});
    b.emit_store(b.emit_ptr_add(y_ptr, b.emit_binary(ir::BinaryOp::Add, row_off, offs)), y, mask);
    b.emit_return();

    verify_and_opt(k);
    bool matched = false;
    auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched && res.ok);
    assert_contains(res.msl_source, "simd_shuffle_up(");
  }
}

void test_split_k_atomic_add_epilogue() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("split_k_reduce");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 256;
  auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
  auto k_start = b.emit_binary(ir::BinaryOp::Mul, b.emit_program_id(1), b.emit_const_i32(kBlock));
  auto row = b.emit_program_id(0);
  auto offs = b.emit_binary(ir::BinaryOp::Add, k_start, b.emit_arange(0, kBlock));
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, k_dim);
  auto zero = b.emit_const_f32(0.0f);
  auto av = b.emit_load(b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, b.emit_binary(ir::BinaryOp::Mul, row, k_dim), offs)),
                        mask, zero);
  auto bv = b.emit_load(b.emit_ptr_add(b_ptr, offs), mask, zero);
  auto partial = b.emit_reduce(b.emit_binary(ir::BinaryOp::Mul, av, bv),
                               ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 0, false});
  b.emit_atomic_add(b.emit_ptr_add(c_ptr, row), partial);
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.param_slots[2].writable);
  assert_contains(res.msl_source, "device float* c_ptr [[buffer(2)]]");
  assert_contains(res.msl_source, "atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>");
  assert_contains(res.msl_source, "if (m1_tid_linear == 0u)");
}

void test_2d_row_reduce() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("row_sum_2d");
  ir::IRBuilder b(k);

  constexpr int64_t kBM = 8;
  constexpr int64_t kBN = 128;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
  auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));

  auto pid = b.emit_program_id(0);
  auto rows = b.emit_binary(ir::BinaryOp::Add, b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(kBM)),
                            b.emit_arange(0, kBM));
  auto cols = b.emit_arange(0, kBN);
  auto r2d = b.emit_broadcast(b.emit_expand_dims(rows, 1), {kBM, kBN});
  auto c2d = b.emit_broadcast(b.emit_expand_dims(cols, 0), {kBM, kBN});
  auto idx = b.emit_binary(ir::BinaryOp::Add, b.emit_binary(ir::BinaryOp::Mul, r2d, n_dim), c2d);
  auto mask = b.emit_binary(ir::BinaryOp::And, b.emit_binary(ir::BinaryOp::CmpLt, r2d, m_dim),
                            b.emit_binary(ir::BinaryOp::CmpLt, c2d, n_dim));
  auto x = b.emit_load(b.emit_ptr_add(x_ptr, idx), mask, b.emit_const_f32(0.0f));
  auto s = b.emit_reduce(x, ir::ReduceAttr{ir::ReduceKind::Sum, ir::ReduceResultMode::ValueOnly, 1, false});
  b.emit_store(b.emit_ptr_add(out_ptr, rows), s, b.emit_binary(ir::BinaryOp::CmpLt, rows, m_dim));
  b.emit_return();

  verify_and_opt(k);
  bool matched = false;
  auto res = codegen::metal::try_emit_reduce_scan_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && res.ok);
  assert(res.launch_spec.tg_x == 32);
  assert(res.launch_spec.tg_y == 8);
  assert_contains(res.msl_source, "simd_sum(");
}

}  // namespace

int main() {
  test_non_reduce_returns_unmatched();
  test_gemv_r4_float4_peeled();
  test_gemv_r4_block2048_collapsible();
  test_odd_block_k_scalar_fallback();
  test_max_min_argmax_argmin();
  test_softmax_chained_reduces_vec4();
  test_tiled_argmax_reduce_inside_forloop();
  test_scan_cumsum_and_cumprod();
  test_split_k_atomic_add_epilogue();
  test_2d_row_reduce();
  std::cout << "test_reduce_scan_emitter OK\n";
  return 0;
}
