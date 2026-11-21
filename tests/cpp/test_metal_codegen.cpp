#include <cassert>
#include <iostream>
#include <string>

#include "codegen/metal/metal_codegen.hpp"
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

void test_vector_add_vec4() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("vector_add");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 1024;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  // Leave writable=false intentionally to verify Store-driven writability inference.
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto n_elems = b.add_param("n_elements", ir::Type::scalar(ir::ScalarDType::I32));
  auto block_c = b.add_param("BLOCK_SIZE", ir::Type::scalar(ir::ScalarDType::I32), true, kBlock);

  auto pid = b.emit_program_id(0);
  auto block_start = b.emit_binary(ir::BinaryOp::Mul, pid, block_c);
  auto arange = b.emit_arange(0, kBlock);
  auto offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(block_start, {kBlock}), arange);
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, b.emit_splat(n_elems, {kBlock}));
  auto zero = b.emit_const_f32(0.0f);

  ir::LoadStoreAttr load_attr;
  load_attr.alignment_bytes = 16;
  load_attr.vec_width = 4;
  load_attr.has_mask = true;
  load_attr.has_other = true;

  ir::LoadStoreAttr store_attr;
  store_attr.alignment_bytes = 16;
  store_attr.vec_width = 4;
  store_attr.has_mask = true;

  auto x = b.emit_load(b.emit_ptr_add(x_ptr, offs), mask, zero, load_attr);
  auto y = b.emit_load(b.emit_ptr_add(y_ptr, offs), mask, zero, load_attr);
  auto out = b.emit_binary(ir::BinaryOp::Add, x, y);
  b.emit_store(b.emit_ptr_add(out_ptr, offs), out, mask, store_attr);
  b.emit_return();

  auto res = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(res.ok);
  assert(res.launch_spec.entry_name == "vector_add");
  assert(res.launch_spec.vec_width == 4);
  assert(res.launch_spec.tg_x == 256);
  assert(res.launch_spec.tg_y == 1);
  assert(res.launch_spec.tg_z == 1);
  assert(res.launch_spec.threadgroup_mem_bytes == 0);
  assert(res.launch_spec.param_slots.size() == 4);

  assert(res.launch_spec.param_slots[0].buffer_index == 0);
  assert(res.launch_spec.param_slots[0].param_index == 0);
  assert(res.launch_spec.param_slots[0].is_pointer);
  assert(!res.launch_spec.param_slots[0].writable);

  assert(res.launch_spec.param_slots[1].buffer_index == 1);
  assert(res.launch_spec.param_slots[1].param_index == 1);
  assert(res.launch_spec.param_slots[1].is_pointer);
  assert(!res.launch_spec.param_slots[1].writable);

  assert(res.launch_spec.param_slots[2].buffer_index == 2);
  assert(res.launch_spec.param_slots[2].param_index == 2);
  assert(res.launch_spec.param_slots[2].is_pointer);
  assert(res.launch_spec.param_slots[2].writable);

  assert(res.launch_spec.param_slots[3].buffer_index == 3);
  assert(res.launch_spec.param_slots[3].param_index == 3);
  assert(!res.launch_spec.param_slots[3].is_pointer);
  assert(res.launch_spec.param_slots[3].dtype == ir::ScalarDType::I32);

  assert_contains(res.msl_source, "device const float* x_ptr [[buffer(0)]]");
  assert_contains(res.msl_source, "device const float* y_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "device float* out_ptr [[buffer(2)]]");
  assert_not_contains(res.msl_source, "device const float* out_ptr");
  assert_contains(res.msl_source, "constant int& n_elements [[buffer(3)]]");
  assert_contains(res.msl_source, "uint3 m1_tg_pos [[threadgroup_position_in_grid]]");
  assert_contains(res.msl_source, "uint3 m1_grid_dim [[threadgroups_per_grid]]");
  assert_contains(res.msl_source, "uint3 m1_tid [[thread_position_in_threadgroup]]");
  assert_contains(res.msl_source, "uint3 m1_tg_size [[threads_per_threadgroup]]");
  assert_contains(res.msl_source, "uint m1_tid_linear [[thread_index_in_threadgroup]]");
  assert_contains(res.msl_source, "*(const device float4*)");
  assert_contains(res.msl_source, "*(device float4*)");
  assert_contains(res.msl_source, "+ 3 < n_elements");
  assert_contains(res.msl_source, "} else {");
  assert_contains(res.msl_source, "for (uint m1_e = 0u; m1_e < 4u; ++m1_e)");
}

void test_fused_add_relu_vec4() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("fused_add_relu");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 1024;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto y_ptr = b.add_param("y_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, false);
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 16, true);
  auto n_elems = b.add_param("n_elements", ir::Type::scalar(ir::ScalarDType::I32));

  auto pid = b.emit_program_id(0);
  auto block_start = b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(kBlock));
  auto offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(block_start, {kBlock}), b.emit_arange(0, kBlock));
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, b.emit_splat(n_elems, {kBlock}));
  auto zero = b.emit_const_f32(0.0f);

  ir::LoadStoreAttr load_attr{.alignment_bytes = 16, .vec_width = 4, .has_mask = true, .has_other = true};
  ir::LoadStoreAttr store_attr{.alignment_bytes = 16, .vec_width = 4, .has_mask = true};

  auto x = b.emit_load(b.emit_ptr_add(x_ptr, offs), mask, zero, load_attr);
  auto y = b.emit_load(b.emit_ptr_add(y_ptr, offs), mask, zero, load_attr);
  auto sum = b.emit_binary(ir::BinaryOp::Add, x, y);
  auto zero_tile = b.emit_splat(zero, {kBlock});
  auto pos_mask = b.emit_binary(ir::BinaryOp::CmpGt, sum, zero_tile);
  auto relu = b.emit_select(pos_mask, sum, zero_tile);
  b.emit_store(b.emit_ptr_add(out_ptr, offs), relu, mask, store_attr);
  b.emit_return();

  auto res = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(res.ok);
  assert(res.launch_spec.vec_width == 4);
  assert(res.launch_spec.tg_x == 256);
  assert_contains(res.msl_source, "float4");
  assert_contains(res.msl_source, "bool4");
  assert_contains(res.msl_source, "metal::select(");
  assert_contains(res.msl_source, "*(const device float4*)");
  assert_contains(res.msl_source, "*(device float4*)");
  assert_contains(res.msl_source, "} else {");
}

void test_general_1d_loop_and_atomic() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("strided_accum");
  ir::IRBuilder b(k);

  constexpr int64_t kBlock = 128;
  auto x_ptr = b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto out_ptr = b.add_param("out_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto n_elems = b.add_param("n_elements", ir::Type::scalar(ir::ScalarDType::I32));
  auto reps = b.add_param("reps", ir::Type::scalar(ir::ScalarDType::I32));

  auto pid = b.emit_program_id(0);
  auto block_start = b.emit_binary(ir::BinaryOp::Mul, pid, b.emit_const_i32(kBlock));
  auto offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(block_start, {kBlock}), b.emit_arange(0, kBlock));
  auto mask = b.emit_binary(ir::BinaryOp::CmpLt, offs, b.emit_splat(n_elems, {kBlock}));
  auto zero = b.emit_const_f32(0.0f);

  ir::LoadStoreAttr load_attr{.alignment_bytes = 4, .vec_width = 1, .has_mask = true, .has_other = true};
  auto val = b.emit_load(b.emit_ptr_add(x_ptr, offs), mask, zero, load_attr);
  auto acc_init = b.emit_splat(zero, {kBlock});

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), reps, b.emit_const_i32(1), {acc_init});
  auto acc_next = b.emit_binary(ir::BinaryOp::Add, loop.iter_args[0], val);
  b.end_for_loop(loop, {acc_next}, entry);

  b.emit_atomic_add(b.emit_ptr_add(out_ptr, offs), loop.results[0], mask);
  b.emit_return();

  auto res = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(res.ok);
  assert(res.launch_spec.vec_width == 1);
  assert(res.launch_spec.tg_x == 128);
  assert(res.launch_spec.param_slots[1].writable);
  assert_contains(res.msl_source, "device float* out_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "for (uint m1_lane = m1_tid_linear; m1_lane < 128u; m1_lane += 128u)");
  assert_contains(res.msl_source, "atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>");
}

void test_2d_matmul_dot_forloop() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("naive_matmul");
  ir::IRBuilder b(k);

  constexpr int64_t kBM = 32;
  constexpr int64_t kBN = 32;
  constexpr int64_t kBK = 32;

  auto a_ptr = b.add_param("a_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto b_ptr = b.add_param("b_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto c_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  auto m_dim = b.add_param("M", ir::Type::scalar(ir::ScalarDType::I32));
  auto n_dim = b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32));
  auto k_dim = b.add_param("K", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_am = b.add_param("stride_am", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_ak = b.add_param("stride_ak", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_bk = b.add_param("stride_bk", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_bn = b.add_param("stride_bn", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_cm = b.add_param("stride_cm", ir::Type::scalar(ir::ScalarDType::I32));
  auto stride_cn = b.add_param("stride_cn", ir::Type::scalar(ir::ScalarDType::I32));
  auto bm_c = b.add_param("BM", ir::Type::scalar(ir::ScalarDType::I32), true, kBM);
  auto bn_c = b.add_param("BN", ir::Type::scalar(ir::ScalarDType::I32), true, kBN);
  auto bk_c = b.add_param("BK", ir::Type::scalar(ir::ScalarDType::I32), true, kBK);

  auto pid_m = b.emit_program_id(0);
  auto pid_n = b.emit_program_id(1);
  auto offs_m = b.emit_binary(ir::BinaryOp::Add,
                              b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_m, bm_c), {kBM}),
                              b.emit_arange(0, kBM));
  auto offs_n = b.emit_binary(ir::BinaryOp::Add,
                              b.emit_splat(b.emit_binary(ir::BinaryOp::Mul, pid_n, bn_c), {kBN}),
                              b.emit_arange(0, kBN));
  auto offs_k = b.emit_arange(0, kBK);

  auto zero_f = b.emit_const_f32(0.0f);
  auto acc_init = b.emit_splat(zero_f, {kBM, kBN});

  ir::BlockId entry = b.current_block();
  auto loop = b.begin_for_loop(b.emit_const_i32(0), k_dim, bk_c, {acc_init});

  auto k_offs = b.emit_binary(ir::BinaryOp::Add, b.emit_splat(loop.iv, {kBK}), offs_k);

  auto am_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(stride_am, {kBM})), 1), {kBM, kBK});
  auto ak_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_ak, {kBK})), 0), {kBM, kBK});
  auto a_ptrs = b.emit_ptr_add(a_ptr, b.emit_binary(ir::BinaryOp::Add, am_2d, ak_2d));
  auto a_m_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1), {kBM, kBK});
  auto a_k_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {kBK})), 0), {kBM, kBK});
  auto a_mask = b.emit_binary(ir::BinaryOp::And, a_m_ok, a_k_ok);
  auto a_tile = b.emit_load(a_ptrs, a_mask, zero_f, ir::LoadStoreAttr{.has_mask = true, .has_other = true});

  auto bk_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, k_offs, b.emit_splat(stride_bk, {kBK})), 1), {kBK, kBN});
  auto bn_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_n, b.emit_splat(stride_bn, {kBN})), 0), {kBK, kBN});
  auto b_ptrs = b.emit_ptr_add(b_ptr, b.emit_binary(ir::BinaryOp::Add, bk_2d, bn_2d));
  auto b_k_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, k_offs, b.emit_splat(k_dim, {kBK})), 1), {kBK, kBN});
  auto b_n_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0), {kBK, kBN});
  auto b_mask = b.emit_binary(ir::BinaryOp::And, b_k_ok, b_n_ok);
  auto b_tile = b.emit_load(b_ptrs, b_mask, zero_f, ir::LoadStoreAttr{.has_mask = true, .has_other = true});

  auto acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], ir::DotAttr{});
  b.end_for_loop(loop, {acc_next}, entry);

  auto cm_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_m, b.emit_splat(stride_cm, {kBM})), 1), {kBM, kBN});
  auto cn_2d = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::Mul, offs_n, b.emit_splat(stride_cn, {kBN})), 0), {kBM, kBN});
  auto c_ptrs = b.emit_ptr_add(c_ptr, b.emit_binary(ir::BinaryOp::Add, cm_2d, cn_2d));
  auto c_m_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_m, b.emit_splat(m_dim, {kBM})), 1), {kBM, kBN});
  auto c_n_ok = b.emit_broadcast(
      b.emit_expand_dims(b.emit_binary(ir::BinaryOp::CmpLt, offs_n, b.emit_splat(n_dim, {kBN})), 0), {kBM, kBN});
  auto c_mask = b.emit_binary(ir::BinaryOp::And, c_m_ok, c_n_ok);
  b.emit_store(c_ptrs, loop.results[0], c_mask, ir::LoadStoreAttr{.has_mask = true});
  b.emit_return();

  auto res = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(res.ok);
  assert(res.launch_spec.entry_name == "naive_matmul");
  assert(res.launch_spec.tg_x == 16);
  assert(res.launch_spec.tg_y == 16);
  assert(res.launch_spec.tg_z == 1);
  assert(res.launch_spec.threadgroup_mem_bytes == 8192);
  assert(res.launch_spec.param_slots.size() == 12);
  assert(!res.launch_spec.param_slots[0].writable);
  assert(!res.launch_spec.param_slots[1].writable);
  assert(res.launch_spec.param_slots[2].writable);

  assert_contains(res.msl_source, "device const float* a_ptr [[buffer(0)]]");
  assert_contains(res.msl_source, "device const float* b_ptr [[buffer(1)]]");
  assert_contains(res.msl_source, "device float* c_ptr [[buffer(2)]]");
  assert_contains(res.msl_source, "threadgroup float tg_a[1024];");
  assert_contains(res.msl_source, "threadgroup float tg_b[1024];");
  assert_contains(res.msl_source, "threadgroup_barrier(mem_flags::mem_threadgroup);");
  assert_contains(res.msl_source, "tg_a[m1_row * 32u + m1_k]");
  assert_contains(res.msl_source, "tg_b[m1_k * 32u + m1_col]");
}

void test_2d_dot_sram_budget_check() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("oversized_matmul");
  ir::IRBuilder b(k);

  constexpr int64_t kBM = 128;
  constexpr int64_t kBN = 128;
  constexpr int64_t kBK = 64;

  auto out_ptr = b.add_param("c_ptr", ir::Type::ptr(ir::ScalarDType::F32), false, 0, 4, true);
  auto a = b.emit_splat(b.emit_const_f32(1.0f), {kBM, kBK});
  auto bb = b.emit_splat(b.emit_const_f32(1.0f), {kBK, kBN});
  auto c = b.emit_dot(a, bb, ir::kInvalidId, ir::DotAttr{});
  auto r_offs = b.emit_broadcast(b.emit_expand_dims(b.emit_arange(0, kBM), 1), {kBM, kBN});
  auto c_offs = b.emit_broadcast(b.emit_expand_dims(b.emit_arange(0, kBN), 0), {kBM, kBN});
  auto lin = b.emit_binary(ir::BinaryOp::Add,
                           b.emit_binary(ir::BinaryOp::Mul, r_offs, b.emit_splat(b.emit_const_i32(kBN), {kBM, kBN})),
                           c_offs);
  b.emit_store(b.emit_ptr_add(out_ptr, lin), c);
  b.emit_return();

  auto res = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(!res.ok);
  assert(res.error.find("32768") != std::string::npos);
}

}  // namespace

int main() {
  test_vector_add_vec4();
  test_fused_add_relu_vec4();
  test_general_1d_loop_and_atomic();
  test_2d_matmul_dot_forloop();
  test_2d_dot_sram_budget_check();
  std::cout << "test_metal_codegen OK\n";
  return 0;
}
