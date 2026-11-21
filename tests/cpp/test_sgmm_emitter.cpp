#include <cassert>
#include <iostream>
#include <string>

#include "codegen/metal/sgmm_emitter.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"

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

void build_standard_gemm_ir(
    ir::Kernel& k,
    int64_t bm,
    int64_t bn,
    int64_t bk,
    ir::ScalarDType ab_dtype = ir::ScalarDType::F32,
    ir::ScalarDType c_dtype = ir::ScalarDType::F32,
    bool with_batch = false,
    bool use_atomic = false,
    bool dot_transpose_a = false,
    bool dot_transpose_b = false,
    bool permute_a = false,
    bool permute_b = false,
    bool external_acc_add = false) {
  ir::IRBuilder b(k);

  auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ab_dtype), false, 0, 16, false);
  auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ab_dtype), false, 0, 16, false);
  auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(c_dtype), false, 0, 16, false);
  auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
  auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_am = b.add_param("stride_am", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_ak = b.add_param("stride_ak", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_bk = b.add_param("stride_bk", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_bn = b.add_param("stride_bn", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_cm = b.add_param("stride_cm", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_cn = b.add_param("stride_cn", ir::Type::scalar(ir::ScalarDType::I32));

  ir::ValueId stride_ab = ir::kInvalidId;
  ir::ValueId stride_bb = ir::kInvalidId;
  ir::ValueId stride_cb = ir::kInvalidId;
  if (with_batch) {
    stride_ab = b.add_param("stride_ab", ir::Type::scalar(ir::ScalarDType::I32));
    stride_bb = b.add_param("stride_bb", ir::Type::scalar(ir::ScalarDType::I32));
    stride_cb = b.add_param("stride_cb", ir::Type::scalar(ir::ScalarDType::I32));
  }

  auto bm_c = b.add_param("BM", ir::Type::scalar(ir::ScalarDType::I32), true, bm);
  auto bn_c = b.add_param("BN", ir::Type::scalar(ir::ScalarDType::I32), true, bn);
  auto bk_c = b.add_param("BK", ir::Type::scalar(ir::ScalarDType::I32), true, bk);

  auto pid_m = b.emit_program_id(0);
  auto pid_n = b.emit_program_id(1);
  auto offs_m = b.emit_binary(
      ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_m, bm_c), {bm}), b.emit_arange(0, bm));
  auto offs_n = b.emit_binary(
      ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_n, bn_c), {bn}), b.emit_arange(0, bn));
  auto offs_k = b.emit_arange(0, bk);

  if (with_batch) {
    auto pid_b = b.emit_program_id(2);
    a_ptr = b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Mul, pid_b, stride_ab));
    b_ptr = b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Mul, pid_b, stride_bb));
    c_ptr = b.emit_ptr_add(c_ptr, b.emit_binary(ir::BinaryOp::Mul, pid_b, stride_cb));
  }

  auto zero_ab = (ab_dtype == ir::ScalarDType::F16) ? b.emit_const_f16(0.0f) : b.emit_const_f32(0.0f);
  auto zero_acc = b.emit_const_f32(0.0f);
  auto acc_init = b.emit_splat(zero_acc, {bm, bn});

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, bk_c, {acc_init});
  auto k_offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(loop.iv, {bk}), offs_k);

  bool a_swap = (dot_transpose_a != permute_a);
  ir::ValueId a_tile_raw = ir::kInvalidId;
  if (!a_swap) {
    auto am_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(stride_am, {bm})), 1), {bm, bk});
    auto ak_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_ak, {bk})), 0), {bm, bk});
    auto a_ptrs = b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, am_2d, ak_2d));
    auto a_m_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {bm})), 1), {bm, bk});
    auto a_k_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {bk})), 0), {bm, bk});
    auto a_mask = b.emit_binary(ir::BinaryOp::And, a_m_ok, a_k_ok);
    a_tile_raw = b.emit_load(a_ptrs, a_mask, zero_ab, ir::LoadStoreAttr{.has_mask = true, .has_other = true});
  } else {
    auto ak_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_ak, {bk})), 1), {bk, bm});
    auto am_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(stride_am, {bm})), 0), {bk, bm});
    auto a_ptrs = b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, ak_2d, am_2d));
    auto a_k_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {bk})), 1), {bk, bm});
    auto a_m_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {bm})), 0), {bk, bm});
    auto a_mask = b.emit_binary(ir::BinaryOp::And, a_k_ok, a_m_ok);
    a_tile_raw = b.emit_load(a_ptrs, a_mask, zero_ab, ir::LoadStoreAttr{.has_mask = true, .has_other = true});
  }
  ir::ValueId a_tile = permute_a ? b.emit_permute(a_tile_raw, {1u, 0u}) : a_tile_raw;

  bool b_swap = (dot_transpose_b != permute_b);
  ir::ValueId b_tile_raw = ir::kInvalidId;
  if (!b_swap) {
    auto bk_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_bk, {bk})), 1), {bk, bn});
    auto bn_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_n, b.emit_splat(stride_bn, {bn})), 0), {bk, bn});
    auto b_ptrs = b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Add, bk_2d, bn_2d));
    auto b_k_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {bk})), 1), {bk, bn});
    auto b_n_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {bn})), 0), {bk, bn});
    auto b_mask = b.emit_binary(ir::BinaryOp::And, b_k_ok, b_n_ok);
    b_tile_raw = b.emit_load(b_ptrs, b_mask, zero_ab, ir::LoadStoreAttr{.has_mask = true, .has_other = true});
  } else {
    auto bn_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_n, b.emit_splat(stride_bn, {bn})), 1), {bn, bk});
    auto bk_2d = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_bk, {bk})), 0), {bn, bk});
    auto b_ptrs = b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Add, bn_2d, bk_2d));
    auto b_n_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {bn})), 1), {bn, bk});
    auto b_k_ok = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {bk})), 0), {bn, bk});
    auto b_mask = b.emit_binary(ir::BinaryOp::And, b_n_ok, b_k_ok);
    b_tile_raw = b.emit_load(b_ptrs, b_mask, zero_ab, ir::LoadStoreAttr{.has_mask = true, .has_other = true});
  }
  ir::ValueId b_tile = permute_b ? b.emit_permute(b_tile_raw, {1u, 0u}) : b_tile_raw;

  ir::DotAttr dattr;
  dattr.acc_dtype = ir::ScalarDType::F32;
  dattr.transpose_a = dot_transpose_a;
  dattr.transpose_b = dot_transpose_b;

  ir::ValueId acc_next = ir::kInvalidId;
  if (external_acc_add) {
    auto d = b.emit_dot(a_tile, b_tile, ir::kInvalidId, dattr);
    acc_next = b.emit_binary(ir::BinaryOp::Add, loop.iter_args[0], d);
  } else {
    acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], dattr);
  }
  b.end_for_loop(loop, {acc_next}, entry);

  auto cm_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(stride_cm, {bm})), 1), {bm, bn});
  auto cn_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_n, b.emit_splat(stride_cn, {bn})), 0), {bm, bn});
  auto c_ptrs = b.emit_ptr_add(c_ptr, b.emit_binary(ir::BinaryOp::Add, cm_2d, cn_2d));
  auto c_m_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {bm})), 1), {bm, bn});
  auto c_n_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {bn})), 0), {bm, bn});
  auto c_mask = b.emit_binary(ir::BinaryOp::And, c_m_ok, c_n_ok);

  ir::ValueId store_val = (c_dtype == ir::ScalarDType::F16) ? b.emit_cast(loop.results[0], c_dtype) : loop.results[0];
  if (use_atomic) {
    b.emit_atomic_add(c_ptrs, store_val, c_mask);
  } else {
    b.emit_store(c_ptrs, store_val, c_mask, ir::LoadStoreAttr{.has_mask = true});
  }
  b.emit_return();
}

void test_dense_64x64x16_double_buffer() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("gemm_64x64x16");
  build_standard_gemm_ir(k, 64, 64, 16);
  k.hints.tile_spec = ir::TileSpecAttr{64, 64, 16, 4, true, 1, 8};

  bool matched = false;
  auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  assert(res.ok);
  assert(res.launch_spec.entry_name == "gemm_64x64x16");
  assert(res.launch_spec.tg_x == 128);
  assert(res.launch_spec.tg_y == 1);
  assert(res.launch_spec.tg_z == 1);
  assert(res.launch_spec.threadgroup_mem_bytes == 0);
  assert(res.launch_spec.param_slots.size() == 12);
  assert(!res.launch_spec.param_slots[0].writable);
  assert(!res.launch_spec.param_slots[1].writable);
  assert(res.launch_spec.param_slots[2].writable);

  assert_contains(res.msl_source, "simdgroup_float8x8 m1_acc[4][4]");
  assert_contains(res.msl_source, "simdgroup_multiply_accumulate");
  assert_contains(res.msl_source, "packed_float4");
  assert_contains(res.msl_source, "m1_ra[");
  assert_contains(res.msl_source, "m1_gw = 8u");
}

void test_dense_128x64x32_and_transpose_tm1() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("gemm_128x64x32_tm1");
  build_standard_gemm_ir(k, 128, 64, 32, ir::ScalarDType::F32, ir::ScalarDType::F32, false, false, true, false);
  k.hints.tile_spec = ir::TileSpecAttr{128, 64, 32, 8, false, 1, 8};

  bool matched = false;
  auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  assert(res.ok);
  assert(res.launch_spec.tg_x == 256);
  assert_contains(res.msl_source, "ulong2(0, 0), true");
  assert_contains(res.msl_source, "simdgroup_store(m1_acc[");
}

void test_permute_and_external_acc_add() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("gemm_permute_ext_acc");
  build_standard_gemm_ir(k, 64, 64, 32, ir::ScalarDType::F32, ir::ScalarDType::F32, false, false, false, false,
                         false, true, true);

  bool matched = false;
  auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  assert(res.ok);
  assert(res.launch_spec.tg_x == 128);
  assert_contains(res.msl_source, "simdgroup_multiply_accumulate");
}

void test_batched_bmm_preserves_pid2() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("bmm_kernel");
  build_standard_gemm_ir(k, 64, 64, 32, ir::ScalarDType::F32, ir::ScalarDType::F32, true);

  bool matched = false;
  auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  assert(res.ok);
  assert(res.launch_spec.param_slots.size() == 15);
  assert_contains(res.msl_source, "m1_tg_pos.z");
  assert_not_contains(res.msl_source, "m1_sk_span");
}

void test_triangular_upper_and_lower() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("triu_gemm");
    build_standard_gemm_ir(k, 64, 64, 32);
    k.hints.structure = ir::StructureAttr{ir::MatrixStructure::UpperTriangular, 0, false};

    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert_contains(res.msl_source, "if (int(m1_row0) >= int(m1_col0) + 64)");
    assert_contains(res.msl_source, "m1_tri_lo");
    assert_contains(res.msl_source, "m1_tri_hi");
    assert_contains(res.msl_source, "int(m1_gr) + 0 <= int(m1_gc)");
  }

  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("tril_gemm");
    build_standard_gemm_ir(k, 64, 64, 32);
    k.hints.structure = ir::StructureAttr{ir::MatrixStructure::LowerTriangular, 0, false};

    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert_contains(res.msl_source, "if (int(m1_col0) >= int(m1_row0) + 64)");
    assert_contains(res.msl_source, "int(m1_gc) <= int(m1_gr) + 0");
  }
}

void test_split_k_hint_and_ir_driven() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("splitk_hint_gemm");
    build_standard_gemm_ir(k, 64, 128, 32);
    k.hints.tile_spec = ir::TileSpecAttr{64, 128, 32, 8, false, 64, 8};

    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert(res.launch_spec.tg_x == 256);
    assert_contains(res.msl_source, "m1_sk_span");
    assert_contains(res.msl_source, "atomic_fetch_add_explicit");
  }

  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("splitk_ir_gemm");
    ir::IRBuilder b(k);
    constexpr int64_t kBM = 64, kBN = 64, kBK = 32;
    auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
    (void)m_dim;
    auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
    auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
    auto k_chunk = b.add_param("K_CHUNK", ir::Type::scalar(ir::ScalarDType::I32));
    auto pid_m = b.emit_program_id(0);
    auto pid_n = b.emit_program_id(1);
    auto pid_z = b.emit_program_id(2);
    auto offs_m = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_m, b.emit_const_i32(kBM)), {kBM}),
        b.emit_arange(0, kBM));
    auto offs_n = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_n, b.emit_const_i32(kBN)), {kBN}),
        b.emit_arange(0, kBN));
    auto offs_k = b.emit_arange(0, kBK);
    auto k_lo = b.emit_binary(ir::BinaryOp::Mul, pid_z, k_chunk);
    auto k_hi = b.emit_binary(ir::BinaryOp::Min, k_dim, b.emit_binary(ir::BinaryOp::Add, k_lo, k_chunk));
    auto acc_init = b.emit_splat(b.emit_const_f32(0.0f), {kBM, kBN});
    ir::BlockId entry = b.current_block();
    auto loop = b.begin_for_loop(k_lo, k_hi, b.emit_const_i32(kBK), {acc_init});
    auto k_offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(loop.iv, {kBK}), offs_k);
    auto am = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(k_dim, {kBM})), 1), {kBM, kBK});
    auto ak = b.emit_broadcast(b.emit_expand_dims(k_offs, 0), {kBM, kBK});
    auto a_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1),
                         {kBM, kBK}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_hi, {kBK})), 0),
                         {kBM, kBK}));
    auto a_tile = b.emit_load(b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, am, ak)), a_mask,
                              b.emit_const_f32(0.0f));
    auto bk = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(n_dim, {kBK})), 1), {kBK, kBN});
    auto bn = b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBK, kBN});
    auto b_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_hi, {kBK})), 1),
                         {kBK, kBN}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0),
                         {kBK, kBN}));
    auto b_tile = b.emit_load(b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Add, bk, bn)), b_mask,
                              b.emit_const_f32(0.0f));
    auto acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], ir::DotAttr{});
    b.end_for_loop(loop, {acc_next}, entry);
    auto cm = b.emit_broadcast(
        b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(n_dim, {kBM})), 1), {kBM, kBN});
    auto cn = b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBM, kBN});
    auto c_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1),
                         {kBM, kBN}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0),
                         {kBM, kBN}));
    b.emit_atomic_add(b.emit_ptr_add(c_ptr, b.emit_binary(ir::BinaryOp::Add, cm, cn)), loop.results[0], c_mask);
    b.emit_return();

    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert_not_contains(res.msl_source, "m1_sk_span");
    assert_contains(res.msl_source, "m1_tg_pos.z");
    assert_contains(res.msl_source, "atomic_fetch_add_explicit");
  }
}

void test_small_32x32_and_f16_mixed_and_ptr_stepping() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("gemm_32x32x16");
    build_standard_gemm_ir(k, 32, 32, 16);
    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert(res.launch_spec.tg_x == 32);
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("gemm_f16_acc_f32");
    build_standard_gemm_ir(k, 64, 64, 32, ir::ScalarDType::F16, ir::ScalarDType::F16);
    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert_contains(res.msl_source, "simdgroup_half8x8");
    assert_contains(res.msl_source, "simdgroup_float8x8");
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("gemm_ptr_step");
    ir::IRBuilder b(k);
    constexpr int64_t kBM = 64, kBN = 64, kBK = 16;
    auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
    (void)m_dim;
    auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
    auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
    auto pid_m = b.emit_program_id(0);
    auto pid_n = b.emit_program_id(1);
    auto offs_m = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_m, b.emit_const_i32(kBM)), {kBM}),
        b.emit_arange(0, kBM));
    auto offs_n = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_n, b.emit_const_i32(kBN)), {kBN}),
        b.emit_arange(0, kBN));
    auto offs_k = b.emit_arange(0, kBK);

    auto a_ptrs_init = b.emit_ptr_add(
        a_ptr, b.emit_binary(ir::BinaryOp::Add,
                             b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(k_dim, {kBM})), 1), {kBM, kBK}),
                             b.emit_broadcast(b.emit_expand_dims(offs_k, 0), {kBM, kBK})));
    auto b_ptrs_init = b.emit_ptr_add(
        b_ptr, b.emit_binary(ir::BinaryOp::Add,
                             b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_k, b.emit_splat(n_dim, {kBK})), 1), {kBK, kBN}),
                             b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBK, kBN})));
    auto acc_init = b.emit_splat(b.emit_const_f32(0.0f), {kBM, kBN});
    auto n_k_iters = b.emit_binary(ir::BinaryOp::Div,
                                   b.emit_binary(ir::BinaryOp::Add, k_dim, b.emit_const_i32(kBK - 1)),
                                   b.emit_const_i32(kBK));
    ir::BlockId entry = b.current_block();
    auto loop = b.begin_for_loop(b.emit_const_i32(0), n_k_iters, b.emit_const_i32(1),
                                 {acc_init, a_ptrs_init, b_ptrs_init});
    auto rem_k = b.emit_binary(ir::BinaryOp::Sub, k_dim,
                               b.emit_binary(ir::BinaryOp::Mul, loop.iv, b.emit_const_i32(kBK)));
    auto a_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1),
                         {kBM, kBK}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_k, b.emit_splat(rem_k, {kBK})), 0),
                         {kBM, kBK}));
    auto b_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_k, b.emit_splat(rem_k, {kBK})), 1),
                         {kBK, kBN}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0),
                         {kBK, kBN}));
    auto a_tile = b.emit_load(loop.iter_args[1], a_mask, b.emit_const_f32(0.0f));
    auto b_tile = b.emit_load(loop.iter_args[2], b_mask, b.emit_const_f32(0.0f));
    auto acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], ir::DotAttr{});
    auto a_ptrs_next = b.emit_ptr_add(loop.iter_args[1], b.emit_const_i32(kBK));
    auto b_ptrs_next = b.emit_ptr_add(loop.iter_args[2], b.emit_binary(ir::BinaryOp::Mul, b.emit_const_i32(kBK), n_dim));
    b.end_for_loop(loop, {acc_next, a_ptrs_next, b_ptrs_next}, entry);

    auto c_ptrs = b.emit_ptr_add(
        c_ptr, b.emit_binary(ir::BinaryOp::Add,
                             b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(n_dim, {kBM})), 1), {kBM, kBN}),
                             b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBM, kBN})));
    auto c_mask = b.emit_binary(
        ir::BinaryOp::And,
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1),
                         {kBM, kBN}),
        b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0),
                         {kBM, kBN}));
    b.emit_store(c_ptrs, loop.results[0], c_mask);
    b.emit_return();

    bool matched = false;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched);
    assert(res.ok);
    assert_contains(res.msl_source, "m1_K_bound");
  }
}

void test_fallback_non_matching_cases() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("tiny_16x16");
    build_standard_gemm_ir(k, 16, 16, 16);
    bool matched = true;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched);
    assert(!res.ok);
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("diag_gemm");
    build_standard_gemm_ir(k, 64, 64, 32);
    k.hints.structure = ir::StructureAttr{ir::MatrixStructure::Diagonal, 0, false};
    bool matched = true;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched);
    assert(!res.ok);
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("oversized_sram");
    build_standard_gemm_ir(k, 128, 128, 64);
    bool matched = true;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched);
    assert(!res.ok);
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("fused_exp_epilogue");
    ir::IRBuilder b(k);
    constexpr int64_t kBM = 64, kBN = 64, kBK = 16;
    auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
    auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
    (void)m_dim;
    auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
    auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
    auto pid_m = b.emit_program_id(0);
    auto pid_n = b.emit_program_id(1);
    auto offs_m = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_m, b.emit_const_i32(kBM)), {kBM}),
        b.emit_arange(0, kBM));
    auto offs_n = b.emit_binary(
        ir::BinaryOp::Add, b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_n, b.emit_const_i32(kBN)), {kBN}),
        b.emit_arange(0, kBN));
    auto offs_k = b.emit_arange(0, kBK);
    auto acc_init = b.emit_splat(b.emit_const_f32(0.0f), {kBM, kBN});
    ir::BlockId entry = b.current_block();
    auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, b.emit_const_i32(kBK), {acc_init});
    auto k_offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(loop.iv, {kBK}), offs_k);
    auto a_tile = b.emit_load(
        b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add,
                                            b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(k_dim, {kBM})), 1), {kBM, kBK}),
                                            b.emit_broadcast(b.emit_expand_dims(k_offs, 0), {kBM, kBK}))));
    auto b_tile = b.emit_load(
        b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Add,
                                            b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(n_dim, {kBK})), 1), {kBK, kBN}),
                                            b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBK, kBN}))));
    auto acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], ir::DotAttr{});
    b.end_for_loop(loop, {acc_next}, entry);
    auto post = b.emit_unary(ir::UnaryOp::Exp, loop.results[0]);
    auto c_ptrs = b.emit_ptr_add(
        c_ptr, b.emit_binary(ir::BinaryOp::Add,
                             b.emit_broadcast(b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(n_dim, {kBM})), 1), {kBM, kBN}),
                             b.emit_broadcast(b.emit_expand_dims(offs_n, 0), {kBM, kBN})));
    b.emit_store(c_ptrs, post);
    b.emit_return();

    bool matched = true;
    auto res = codegen::metal::try_emit_sgmm_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched);
    assert(!res.ok);
  }
}

}  // namespace

int main() {
  test_dense_64x64x16_double_buffer();
  test_dense_128x64x32_and_transpose_tm1();
  test_permute_and_external_acc_add();
  test_batched_bmm_preserves_pid2();
  test_triangular_upper_and_lower();
  test_split_k_hint_and_ir_driven();
  test_small_32x32_and_f16_mixed_and_ptr_stepping();
  test_fallback_non_matching_cases();
  std::cout << "test_sgmm_emitter OK\n";
  return 0;
}
