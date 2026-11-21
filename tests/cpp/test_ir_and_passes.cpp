#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "ir/attrs.hpp"
#include "ir/builder.hpp"
#include "ir/ir.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/types.hpp"
#include "ir/verifier.hpp"
#include "passes/canonicalize.hpp"
#include "passes/contiguity_analysis.hpp"
#include "passes/cse.hpp"
#include "passes/dce.hpp"
#include "passes/licm.hpp"
#include "passes/pass_manager.hpp"
#include "passes/tile_layout.hpp"

using namespace m1triton::ir;
using namespace m1triton::passes;

namespace {

size_t count_active_ops(const Kernel& k, OpKind kind) {
  size_t n = 0;
  for (const auto& op : k.ops) {
    if (!op->erased && op->kind == kind) ++n;
  }
  return n;
}

size_t count_block_active_ops(const Kernel& k, BlockId blk_id, OpKind kind) {
  size_t n = 0;
  for (OpId id : k.block(blk_id).ops) {
    if (!k.op(id).erased && k.op(id).kind == kind) ++n;
  }
  return n;
}

void build_vector_add_kernel(Kernel& k, ScalarDType dt, int64_t block_size, uint32_t ptr_div, bool add_redundant = false) {
  IRBuilder b(k);
  const ValueId x_ptr = b.add_param("x_ptr", Type::ptr(dt), false, 0, ptr_div, false);
  const ValueId y_ptr = b.add_param("y_ptr", Type::ptr(dt), false, 0, ptr_div, false);
  const ValueId out_ptr = b.add_param("out_ptr", Type::ptr(dt), false, 0, ptr_div, true);
  const ValueId n = b.add_param("n", Type::scalar(ScalarDType::I32), false, 0, 1, false);
  b.add_param("BLOCK_SIZE", Type::scalar(ScalarDType::I32), true, block_size, static_cast<uint32_t>(block_size), false);

  const ValueId pid = b.emit_program_id(0);
  const ValueId bs = b.emit_const_i32(static_cast<int32_t>(block_size));
  const ValueId block_start = b.emit_binary(BinaryOp::Mul, pid, bs);
  const ValueId ar = b.emit_arange(0, block_size);
  const ValueId offs = b.emit_binary(BinaryOp::Add, block_start, ar);
  const ValueId mask1 = b.emit_binary(BinaryOp::CmpLt, offs, n);

  const ValueId px = b.emit_ptr_add(x_ptr, offs);
  const ValueId py = b.emit_ptr_add(y_ptr, offs);
  const ValueId po = b.emit_ptr_add(out_ptr, offs);

  const ValueId zero = (dt == ScalarDType::F16) ? b.emit_const_f16(0.0f) : b.emit_const_f32(0.0f);
  const ValueId vx = b.emit_load(px, mask1, zero);
  const ValueId mask2 = add_redundant ? b.emit_binary(BinaryOp::CmpLt, offs, n) : mask1;
  const ValueId vy = b.emit_load(py, mask2, zero);

  ValueId sum = b.emit_binary(BinaryOp::Add, vx, vy);
  if (add_redundant) {
    const ValueId dead = b.emit_binary(BinaryOp::Mul, vx, vy);
    (void)dead;
    sum = b.emit_binary(BinaryOp::Add, sum, zero);
  }
  b.emit_store(po, sum, mask1);
  b.emit_return();
}

void build_matmul_kernel(Kernel& k, int64_t bm, int64_t bn, int64_t bk) {
  IRBuilder b(k);
  const ValueId a_ptr = b.add_param("a_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, false);
  const ValueId b_ptr = b.add_param("b_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, false);
  const ValueId c_ptr = b.add_param("c_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, true);
  const ValueId M = b.add_param("M", Type::scalar(ScalarDType::I32), false, 0, 16, false);
  const ValueId N = b.add_param("N", Type::scalar(ScalarDType::I32), false, 0, 16, false);
  const ValueId K = b.add_param("K", Type::scalar(ScalarDType::I32), false, 0, 16, false);
  (void)M;
  (void)N;

  const ValueId pid_m = b.emit_program_id(0);
  const ValueId pid_n = b.emit_program_id(1);
  const ValueId c_bm = b.emit_const_i32(static_cast<int32_t>(bm));
  const ValueId c_bn = b.emit_const_i32(static_cast<int32_t>(bn));
  const ValueId c_bk = b.emit_const_i32(static_cast<int32_t>(bk));
  const ValueId c_zero_i = b.emit_const_i32(0);

  const ValueId offs_m = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, pid_m, c_bm), b.emit_arange(0, bm));
  const ValueId offs_n = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, pid_n, c_bn), b.emit_arange(0, bn));
  const ValueId offs_k = b.emit_arange(0, bk);

  const ValueId offs_m_2d = b.emit_expand_dims(offs_m, 1);
  const ValueId offs_n_2d = b.emit_expand_dims(offs_n, 0);
  const ValueId offs_k_row = b.emit_expand_dims(offs_k, 0);
  const ValueId offs_k_col = b.emit_expand_dims(offs_k, 1);

  const ValueId a_row = b.emit_binary(BinaryOp::Mul, offs_m_2d, K);
  const ValueId a_off = b.emit_binary(BinaryOp::Add, a_row, offs_k_row);
  const ValueId a_ptrs_init = b.emit_ptr_add(a_ptr, a_off);

  const ValueId b_row = b.emit_binary(BinaryOp::Mul, offs_k_col, N);
  const ValueId b_off = b.emit_binary(BinaryOp::Add, b_row, offs_n_2d);
  const ValueId b_ptrs_init = b.emit_ptr_add(b_ptr, b_off);

  const ValueId acc_init = b.emit_splat(b.emit_const_f32(0.0f), {bm, bn});

  const BlockId entry = b.current_block();
  const auto loop = b.begin_for_loop(c_zero_i, K, c_bk, {acc_init, a_ptrs_init, b_ptrs_init});

  // Invariant expression inside loop to also exercise LICM.
  const ValueId b_step = b.emit_binary(BinaryOp::Mul, c_bk, N);
  const ValueId a_tile = b.emit_load(loop.iter_args[1]);
  const ValueId b_tile = b.emit_load(loop.iter_args[2]);
  DotAttr dot_attr;
  dot_attr.acc_dtype = ScalarDType::F32;
  const ValueId acc_next = b.emit_dot(a_tile, b_tile, loop.iter_args[0], dot_attr);
  const ValueId a_ptrs_next = b.emit_ptr_add(loop.iter_args[1], c_bk);
  const ValueId b_ptrs_next = b.emit_ptr_add(loop.iter_args[2], b_step);
  b.end_for_loop(loop, {acc_next, a_ptrs_next, b_ptrs_next}, entry);

  const ValueId c_off = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, offs_m_2d, N), offs_n_2d);
  const ValueId c_ptrs = b.emit_ptr_add(c_ptr, c_off);
  b.emit_store(c_ptrs, loop.results[0]);
  b.emit_return();
}

void test_types_and_attrs() {
  assert(dtype_name(ScalarDType::F32) == "f32");
  assert(dtype_name(ScalarDType::F16) == "f16");
  assert(dtype_name(ScalarDType::Bool) == "bool");
  assert(parse_dtype("f32") == ScalarDType::F32);
  assert(parse_dtype("i1") == ScalarDType::Bool);
  assert(!parse_dtype("unknown").has_value());
  assert(dtype_bytes(ScalarDType::F32) == 4);
  assert(dtype_bytes(ScalarDType::F16) == 2);
  assert(dtype_bytes(ScalarDType::I64) == 8);

  const Type t1 = Type::tile({64, 64}, ScalarDType::F32);
  LayoutAttr l;
  l.kind = LayoutKind::Blocked;
  l.vec_width = 4;
  l.tg_threads[0] = 16;
  l.tg_threads[1] = 16;
  const Type t2 = Type::tile({64, 64}, ScalarDType::F32, l);
  assert(t1 != t2);
  assert(t1.same_shape_and_elem(t2));
  assert(t1.is_float_elem() && !t1.is_int_elem() && !t1.is_ptr_elem());
  assert(Type::ptr(ScalarDType::F16).is_ptr_elem());
  assert(Type::ptr(ScalarDType::F16).elem_bytes() == 8);

  assert(unary_op_name(UnaryOp::Sigmoid) == "sigmoid");
  assert(binary_op_name(BinaryOp::CmpLt) == "cmp_lt");
  assert(is_comparison_op(BinaryOp::CmpLe));
  assert(!is_comparison_op(BinaryOp::Add));
  assert(is_commutative_op(BinaryOp::Add));
  assert(is_commutative_op(BinaryOp::Mul));
  assert(!is_commutative_op(BinaryOp::Sub));
  assert(!is_pure_op(OpKind::Load));
  assert(!is_pure_op(OpKind::Store));
  assert(!is_pure_op(OpKind::Barrier));
  assert(is_pure_op(OpKind::Binary));
  assert(is_terminator_op(OpKind::Return));
  assert(is_terminator_op(OpKind::Yield));
  std::cout << "[PASS] test_types_and_attrs\n";
}

void test_builder_and_verifier_positive() {
  const auto target = TargetSpec::apple_m1_max();
  Module mod;

  // 1. vector_add
  auto& k1 = mod.add_kernel("vector_add");
  build_vector_add_kernel(k1, ScalarDType::F32, 256, 16);

  // 2. matmul
  auto& k2 = mod.add_kernel("matmul");
  build_matmul_kernel(k2, 64, 64, 16);

  // 3. Reduce, Scan, If/Else, Dot transpose/batched
  auto& k3 = mod.add_kernel("misc_ops");
  {
    IRBuilder b(k3);
    const ValueId x_ptr = b.add_param("x_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, false);
    const ValueId out_ptr = b.add_param("out_ptr", Type::ptr(ScalarDType::F32), false, 0, 16, true);
    const ValueId flag = b.add_param("flag", Type::scalar(ScalarDType::Bool));

    const ValueId r = b.emit_expand_dims(b.emit_arange(0, 8), 1);
    const ValueId c = b.emit_expand_dims(b.emit_arange(0, 16), 0);
    const ValueId offs = b.emit_binary(BinaryOp::Add, b.emit_binary(BinaryOp::Mul, r, b.emit_const_i32(16)), c);
    const ValueId x = b.emit_load(b.emit_ptr_add(x_ptr, offs));
    assert(k3.value(x).type.as_tile().shape == (std::vector<int64_t>{8, 16}));

    ReduceAttr ra;
    ra.kind = ReduceKind::Sum;
    ra.axis = -1;
    ra.keep_dims = false;
    const ValueId red = b.emit_reduce(x, ra);
    assert(k3.value(red).type.as_tile().shape == (std::vector<int64_t>{8}));

    ReduceAttr ra1d;
    ra1d.kind = ReduceKind::Max;
    ra1d.axis = 0;
    const ValueId scalar_max = b.emit_reduce(red, ra1d);
    assert(k3.value(scalar_max).type.is_scalar());

    ScanAttr sa;
    sa.kind = ScanKind::Cumsum;
    sa.axis = -1;
    const ValueId scanned = b.emit_scan(x, sa);
    assert(k3.value(scanned).type.as_tile().shape == (std::vector<int64_t>{8, 16}));

    const BlockId parent = b.current_block();
    const auto ifh = b.begin_if(flag, {Type::scalar(ScalarDType::F32)}, true);
    const ValueId tval = b.emit_unary(UnaryOp::Exp, scalar_max);
    b.emit_yield({tval});
    b.set_insertion_block(ifh.else_block);
    const ValueId fval = b.emit_unary(UnaryOp::Neg, scalar_max);
    b.emit_yield({fval});
    b.set_insertion_block(parent);

    const ValueId shifted = b.emit_binary(BinaryOp::Add, scanned, ifh.results[0]);
    b.emit_store(b.emit_ptr_add(out_ptr, offs), shifted);
    b.emit_return();
  }

  const auto vr = IRVerifier::verify(mod, target, VerifyStage::HighLevel);
  if (!vr.ok) {
    std::cerr << "Unexpected verifier error:\n" << vr.summary() << "\n";
  }
  assert(vr.ok);
  std::cout << "[PASS] test_builder_and_verifier_positive\n";
}

void test_verifier_negative_cases() {
  const auto target = TargetSpec::apple_m1_max();

  // Case 1: Missing Return terminator
  {
    Module mod;
    auto& k = mod.add_kernel("missing_ret");
    IRBuilder b(k);
    b.emit_const_i32(1);
    const auto vr = IRVerifier::verify(k, target);
    assert(!vr.ok);
  }

  // Case 2: SSA dominance violation (using value defined inside If block outside)
  {
    Module mod;
    auto& k = mod.add_kernel("bad_dom");
    IRBuilder b(k);
    const ValueId flag = b.add_param("flag", Type::scalar(ScalarDType::Bool));
    const BlockId entry = b.current_block();
    const auto ifh = b.begin_if(flag, {}, false);
    const ValueId inner = b.emit_const_f32(3.0f);
    b.emit_yield({});
    b.set_insertion_block(entry);
    b.emit_unary(UnaryOp::Neg, inner);
    b.emit_return();
    const auto vr = IRVerifier::verify(k, target);
    assert(!vr.ok);
  }

  // Case 3: Dot inner K dimension mismatch
  {
    Module mod;
    auto& k = mod.add_kernel("bad_dot");
    IRBuilder b(k);
    const ValueId a = b.emit_splat(b.emit_const_f32(1.0f), {16, 8});
    const ValueId c = b.emit_splat(b.emit_const_f32(1.0f), {12, 16});
    k.create_op(OpKind::Dot, b.current_block(), {a, c}, {Type::tile({16, 16}, ScalarDType::F32)}, DotAttr{});
    b.emit_return();
    const auto vr = IRVerifier::verify(k, target);
    assert(!vr.ok);
  }

  // Case 4: ForLoop yield type mismatch
  {
    Module mod;
    auto& k = mod.add_kernel("bad_for");
    IRBuilder b(k);
    const ValueId lo = b.emit_const_i32(0);
    const ValueId hi = b.emit_const_i32(4);
    const ValueId st = b.emit_const_i32(1);
    const ValueId init = b.emit_const_f32(0.0f);
    const BlockId entry = b.current_block();
    const auto loop = b.begin_for_loop(lo, hi, st, {init});
    const ValueId wrong = b.emit_const_i32(7);
    b.end_for_loop(loop, {wrong}, entry);
    b.emit_return();
    const auto vr = IRVerifier::verify(k, target);
    assert(!vr.ok);
  }

  // Case 5: PostTile threadgroup SRAM overflow (> 32768 bytes)
  {
    Module mod;
    auto& k = mod.add_kernel("sram_overflow");
    IRBuilder b(k);
    ThreadgroupAllocAttr alloc;
    alloc.elem = ScalarDType::F32;
    alloc.shape = {128, 64};  // 32768 bytes * 2 stages = 65536 > 32768
    alloc.stages = 2;
    k.create_op(OpKind::ThreadgroupAlloc, b.current_block(), {}, {}, alloc);
    b.emit_return();
    assert(IRVerifier::verify(k, target, VerifyStage::HighLevel).ok);
    assert(!IRVerifier::verify(k, target, VerifyStage::PostTile).ok);
  }

  // Case 6: PostTile thread count not divisible by simd_width=32
  {
    Module mod;
    auto& k = mod.add_kernel("bad_threads");
    IRBuilder b(k);
    const ValueId ar = b.emit_arange(0, 64);
    k.value(ar).type.as_tile_mut().layout.kind = LayoutKind::Blocked;
    k.value(ar).type.as_tile_mut().layout.tg_threads[0] = 48;
    b.emit_return();
    assert(!IRVerifier::verify(k, target, VerifyStage::PostTile).ok);
  }

  // Case 7: PostTile thread count > max_threads_per_threadgroup (1024)
  {
    Module mod;
    auto& k = mod.add_kernel("too_many_threads");
    IRBuilder b(k);
    const ValueId ar = b.emit_arange(0, 64);
    k.value(ar).type.as_tile_mut().layout.kind = LayoutKind::Blocked;
    k.value(ar).type.as_tile_mut().layout.tg_threads[0] = 2048;
    b.emit_return();
    assert(!IRVerifier::verify(k, target, VerifyStage::PostTile).ok);
  }

  // Case 8: PreCodegen rejects Unassigned tile layout
  {
    Module mod;
    auto& k = mod.add_kernel("unassigned_layout");
    IRBuilder b(k);
    b.emit_arange(0, 64);
    b.emit_return();
    assert(IRVerifier::verify(k, target, VerifyStage::HighLevel).ok);
    assert(!IRVerifier::verify(k, target, VerifyStage::PreCodegen).ok);
  }

  // Case 9: Binary operand type mismatch
  {
    Module mod;
    auto& k = mod.add_kernel("bad_binary");
    IRBuilder b(k);
    const ValueId a = b.emit_const_i32(1);
    const ValueId c = b.emit_const_f32(1.0f);
    k.create_op(OpKind::Binary, b.current_block(), {a, c}, {Type::scalar(ScalarDType::I32)}, BinaryAttr{BinaryOp::Add});
    b.emit_return();
    assert(!IRVerifier::verify(k, target).ok);
  }

  // Case 10: Select with non-bool condition
  {
    Module mod;
    auto& k = mod.add_kernel("bad_select");
    IRBuilder b(k);
    const ValueId cond = b.emit_const_i32(1);
    const ValueId a = b.emit_const_f32(1.0f);
    const ValueId c = b.emit_const_f32(2.0f);
    k.create_op(OpKind::Select, b.current_block(), {cond, a, c}, {Type::scalar(ScalarDType::F32)});
    b.emit_return();
    assert(!IRVerifier::verify(k, target).ok);
  }

  // Case 11: Value-producing If without else region
  {
    Module mod;
    auto& k = mod.add_kernel("if_no_else");
    IRBuilder b(k);
    const ValueId cond = b.add_param("cond", Type::scalar(ScalarDType::Bool));
    const BlockId entry = b.current_block();
    b.begin_if(cond, {Type::scalar(ScalarDType::F32)}, /*with_else=*/false);
    b.emit_yield({b.emit_const_f32(1.0f)});
    b.set_insertion_block(entry);
    b.emit_return();
    assert(!IRVerifier::verify(k, target).ok);
  }

  // Case 12: Load operand count mismatch with LoadStoreAttr::has_mask
  {
    Module mod;
    auto& k = mod.add_kernel("bad_load_mask");
    IRBuilder b(k);
    const ValueId p = b.add_param("p", Type::ptr(ScalarDType::F32));
    LoadStoreAttr bad_attr;
    bad_attr.has_mask = true;
    k.create_op(OpKind::Load, b.current_block(), {p}, {Type::scalar(ScalarDType::F32)}, bad_attr);
    b.emit_return();
    assert(!IRVerifier::verify(k, target).ok);
  }

  std::cout << "[PASS] test_verifier_negative_cases\n";
}

void test_printer_parser_roundtrip() {
  const auto target = TargetSpec::apple_m1_max();
  Module mod;
  auto& k1 = mod.add_kernel("vector_add");
  build_vector_add_kernel(k1, ScalarDType::F32, 256, 16);

  auto& k2 = mod.add_kernel("matmul_with_hints");
  k2.hints.tile_spec = TileSpecAttr{64, 64, 16, 4, true, 1};
  k2.hints.persistent_grid = PersistentGridAttr{true, 2, 8};
  build_matmul_kernel(k2, 64, 64, 16);

  auto& k3 = mod.add_kernel("branch_and_alloc");
  {
    IRBuilder b(k3);
    const ValueId cond = b.add_param("cond", Type::scalar(ScalarDType::Bool));
    const ValueId p = b.add_param("p", Type::ptr(ScalarDType::F16), false, 0, 8, true);
    ThreadgroupAllocAttr tg;
    tg.elem = ScalarDType::F16;
    tg.shape = {32, 32};
    tg.swizzle = SwizzleMode::PadBank4;
    tg.pad_elems = 4;
    tg.stages = 2;
    k3.create_op(OpKind::ThreadgroupAlloc, b.current_block(), {}, {}, tg);
    k3.create_op(OpKind::Barrier, b.current_block(), {}, {}, BarrierAttr{BarrierScope::Threadgroup});

    const BlockId entry = b.current_block();
    const auto ifh = b.begin_if(cond, {Type::scalar(ScalarDType::F16)}, true);
    b.emit_yield({b.emit_const_f16(1.25f)});
    b.set_insertion_block(ifh.else_block);
    b.emit_yield({b.emit_const_f16(-2.5f)});
    b.set_insertion_block(entry);

    b.emit_store(p, ifh.results[0]);
    b.emit_return();
  }

  // Run M1 pipeline on k1 so we also round-trip assigned LayoutAttr!
  PassManager pm = PassManager::build_m1_pipeline();
  PassContext ctx{target, true, false, nullptr};
  assert(pm.run(k1, ctx).ok);

  const std::string printed1 = IRPrinter::print(mod);
  const auto parsed = IRParser::parse(printed1);
  if (!parsed.ok) {
    std::cerr << "Parse error: " << parsed.error << "\nIR:\n" << printed1 << "\n";
  }
  assert(parsed.ok && parsed.module != nullptr);

  const auto vr = IRVerifier::verify(*parsed.module, target, VerifyStage::PostTile);
  if (!vr.ok) {
    std::cerr << "Post-parse verify error:\n" << vr.summary() << "\n";
  }
  assert(vr.ok);

  const std::string printed2 = IRPrinter::print(*parsed.module);
  if (printed1 != printed2) {
    std::cerr << "Roundtrip mismatch!\n--- PRINTED 1 ---\n" << printed1 << "\n--- PRINTED 2 ---\n" << printed2 << "\n";
  }
  assert(printed1 == printed2);

  // Negative parser check
  const auto bad = IRParser::parse("kernel @broken( { invalid");
  assert(!bad.ok && !bad.error.empty());

  std::cout << "[PASS] test_printer_parser_roundtrip\n";
}

void test_canonicalize_pass() {
  const auto target = TargetSpec::apple_m1_max();
  Module mod;
  auto& k = mod.add_kernel("canon_test");
  IRBuilder b(k);
  const ValueId ptr = b.add_param("ptr", Type::ptr(ScalarDType::F32), false, 0, 16, true);
  const ValueId x = b.add_param("x", Type::scalar(ScalarDType::F32));
  const ValueId idx = b.add_param("idx", Type::scalar(ScalarDType::I32));

  // Scalar constant folding: (2 + 3) * 4 = 20
  const ValueId c2 = b.emit_const_i32(2);
  const ValueId c3 = b.emit_const_i32(3);
  const ValueId c4 = b.emit_const_i32(4);
  const ValueId folded = b.emit_binary(BinaryOp::Mul, b.emit_binary(BinaryOp::Add, c2, c3), c4);

  // Algebraic identities: (((x + 0) - 0) * 1), select(true, val, 99.0), y * 0 -> 0
  const ValueId f0 = b.emit_const_f32(0.0f);
  const ValueId f1 = b.emit_const_f32(1.0f);
  const ValueId f99 = b.emit_const_f32(99.0f);
  const ValueId id1 = b.emit_binary(BinaryOp::Add, x, f0);
  const ValueId id2 = b.emit_binary(BinaryOp::Sub, id1, f0);
  const ValueId id3 = b.emit_binary(BinaryOp::Mul, id2, f1);
  const ValueId sel = b.emit_select(b.emit_const_bool(true), id3, f99);
  const ValueId mul0 = b.emit_binary(BinaryOp::Mul, x, f0);
  const ValueId final_val = b.emit_binary(BinaryOp::Add, sel, mul0);

  // Nested PtrAdd folding: ptr_add(ptr_add(ptr, idx), folded) -> ptr_add(ptr, add(idx, 20))
  const ValueId p1 = b.emit_ptr_add(ptr, idx);
  const ValueId p2 = b.emit_ptr_add(p1, folded);
  b.emit_store(p2, final_val);
  b.emit_return();

  PassManager pm;
  pm.add_pass<CanonicalizePass>();
  pm.add_pass<DCEPass>();
  PassContext ctx{target, true, false, nullptr};
  const auto pr = pm.run(k, ctx);
  assert(pr.ok && pr.changed);

  // Nested PtrAdd should have been folded to a single PtrAdd using original `ptr`!
  assert(count_active_ops(k, OpKind::PtrAdd) == 1);
  assert(count_active_ops(k, OpKind::Select) == 0);
  // Only 1 Binary(Add) should remain (idx + 20), and store value should be directly param `x`!
  assert(count_active_ops(k, OpKind::Binary) == 1);
  bool found_store = false;
  for (const auto& op : k.ops) {
    if (!op->erased && op->kind == OpKind::Store) {
      found_store = true;
      assert(op->operands[1] == x);
    }
    if (!op->erased && op->kind == OpKind::PtrAdd) {
      assert(op->operands[0] == ptr);
    }
  }
  assert(found_store);
  std::cout << "[PASS] test_canonicalize_pass\n";
}

void test_cse_pass() {
  const auto target = TargetSpec::apple_m1_max();
  Module mod;
  auto& k = mod.add_kernel("cse_test");
  IRBuilder b(k);
  const ValueId ptr = b.add_param("ptr", Type::ptr(ScalarDType::F32), false, 0, 16, true);
  const ValueId a = b.add_param("a", Type::scalar(ScalarDType::I32));
  const ValueId c = b.add_param("c", Type::scalar(ScalarDType::I32));

  // Commutative duplicate: a + c vs c + a
  const ValueId s1 = b.emit_binary(BinaryOp::Add, a, c);
  const ValueId p1 = b.emit_ptr_add(ptr, s1);
  const ValueId v1 = b.emit_load(p1);
  const ValueId s2 = b.emit_binary(BinaryOp::Add, c, a);
  const ValueId p2 = b.emit_ptr_add(ptr, s2);
  const ValueId v2 = b.emit_load(p2);  // Must NOT be CSE'd with v1!
  const ValueId sum = b.emit_binary(BinaryOp::Add, v1, v2);
  b.emit_store(p1, sum);
  b.emit_return();

  PassManager pm;
  pm.add_pass<CSEPass>();
  PassContext ctx{target, true, false, nullptr};
  const auto pr = pm.run(k, ctx);
  assert(pr.ok && pr.changed);

  // s1 & s2 deduplicated -> 2 Binary ops total (s1 and v1+v2); p1 & p2 deduplicated -> 1 PtrAdd; Loads untouched -> 2!
  assert(count_active_ops(k, OpKind::Binary) == 2);
  assert(count_active_ops(k, OpKind::PtrAdd) == 1);
  assert(count_active_ops(k, OpKind::Load) == 2);

  // Scoped CSE across sibling If/Else branches: outer expr reused in both, sibling expr NOT leaked across branches.
  {
    Module mod2;
    auto& k2 = mod2.add_kernel("cse_scoped");
    IRBuilder b2(k2);
    const ValueId flag = b2.add_param("flag", Type::scalar(ScalarDType::Bool));
    const ValueId x = b2.add_param("x", Type::scalar(ScalarDType::F32));
    const ValueId y = b2.add_param("y", Type::scalar(ScalarDType::F32));
    const ValueId out = b2.add_param("out", Type::ptr(ScalarDType::F32), false, 0, 4, true);

    const ValueId outer_sum = b2.emit_binary(BinaryOp::Add, x, y);
    const BlockId entry = b2.current_block();
    const auto ifh = b2.begin_if(flag, {Type::scalar(ScalarDType::F32)}, true);
    const ValueId dup_outer_then = b2.emit_binary(BinaryOp::Add, y, x);
    const ValueId branch_only_then = b2.emit_binary(BinaryOp::Mul, dup_outer_then, x);
    b2.emit_yield({branch_only_then});
    b2.set_insertion_block(ifh.else_block);
    const ValueId dup_outer_else = b2.emit_binary(BinaryOp::Add, x, y);
    const ValueId branch_only_else = b2.emit_binary(BinaryOp::Mul, dup_outer_else, x);
    b2.emit_yield({branch_only_else});
    b2.set_insertion_block(entry);
    b2.emit_store(out, b2.emit_binary(BinaryOp::Add, outer_sum, ifh.results[0]));
    b2.emit_return();

    assert(pm.run(k2, ctx).ok);
    // Both dup_outer_then and dup_outer_else replaced with outer_sum; sibling Muls MUST both remain (2 Muls)!
    assert(count_block_active_ops(k2, ifh.then_block, OpKind::Binary) == 1);
    assert(count_block_active_ops(k2, ifh.else_block, OpKind::Binary) == 1);
  }

  std::cout << "[PASS] test_cse_pass\n";
}

void test_licm_pass() {
  const auto target = TargetSpec::apple_m1_max();
  Module mod;
  auto& k = mod.add_kernel("licm_test");
  IRBuilder b(k);
  const ValueId out = b.add_param("out", Type::ptr(ScalarDType::F32), false, 0, 16, true);
  const ValueId alpha = b.add_param("alpha", Type::scalar(ScalarDType::F32));
  const ValueId beta = b.add_param("beta", Type::scalar(ScalarDType::F32));
  const ValueId n = b.add_param("n", Type::scalar(ScalarDType::I32));

  const ValueId c0 = b.emit_const_i32(0);
  const ValueId c1 = b.emit_const_i32(1);
  const ValueId init = b.emit_const_f32(0.0f);

  const BlockId entry = b.current_block();
  const auto loop = b.begin_for_loop(c0, n, c1, {init});
  // Chain of loop-invariant pure ops: inv1 = alpha * beta, inv2 = exp(inv1)
  const ValueId inv1 = b.emit_binary(BinaryOp::Mul, alpha, beta);
  const ValueId inv2 = b.emit_unary(UnaryOp::Exp, inv1);
  // Loop-variant op depending on iv & iter_arg
  const ValueId iv_f = b.emit_cast(loop.iv, ScalarDType::F32);
  const ValueId next = b.emit_binary(BinaryOp::Add, loop.iter_args[0], b.emit_binary(BinaryOp::Add, inv2, iv_f));
  b.end_for_loop(loop, {next}, entry);

  b.emit_store(out, loop.results[0]);
  b.emit_return();

  assert(count_block_active_ops(k, loop.body_block, OpKind::Unary) == 1);
  assert(count_block_active_ops(k, entry, OpKind::Unary) == 0);

  PassManager pm;
  pm.add_pass<LICMPass>();
  PassContext ctx{target, true, false, nullptr};
  const auto pr = pm.run(k, ctx);
  assert(pr.ok && pr.changed);

  // inv1 (Mul) and inv2 (Exp) must be hoisted out of loop.body_block into entry!
  assert(count_block_active_ops(k, loop.body_block, OpKind::Unary) == 0);
  assert(count_block_active_ops(k, entry, OpKind::Unary) == 1);
  assert(count_block_active_ops(k, loop.body_block, OpKind::Binary) == 2);
  std::cout << "[PASS] test_licm_pass\n";
}

void test_contiguity_analysis() {
  const auto target = TargetSpec::apple_m1_max();
  PassContext ctx{target, true, false, nullptr};

  // Positive 1: F32 contiguous aligned -> vec_width=4, alignment_bytes=16
  {
    Module mod;
    auto& k = mod.add_kernel("vec4_f32");
    build_vector_add_kernel(k, ScalarDType::F32, 256, 16);
    PassManager pm;
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
    const auto pr = pm.run(k, ctx);
    assert(pr.ok && pr.changed);
    for (const auto& op : k.ops) {
      if (!op->erased && (op->kind == OpKind::Load || op->kind == OpKind::Store)) {
        const auto& ls = std::get<LoadStoreAttr>(op->attr);
        assert(ls.vec_width == 4);
        assert(ls.alignment_bytes == 16);
      }
    }
  }

  // Positive 2: F16 contiguous aligned -> vec_width=4, alignment_bytes=8
  {
    Module mod;
    auto& k = mod.add_kernel("vec4_f16");
    build_vector_add_kernel(k, ScalarDType::F16, 256, 16);
    PassManager pm;
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
    const auto pr = pm.run(k, ctx);
    assert(pr.ok && pr.changed);
    for (const auto& op : k.ops) {
      if (!op->erased && (op->kind == OpKind::Load || op->kind == OpKind::Store)) {
        const auto& ls = std::get<LoadStoreAttr>(op->attr);
        assert(ls.vec_width == 4);
        assert(ls.alignment_bytes == 8);
      }
    }
  }

  // Negative 1: unaligned base pointer (divisibility=1) -> stays vec_width=1
  {
    Module mod;
    auto& k = mod.add_kernel("unaligned_ptr");
    build_vector_add_kernel(k, ScalarDType::F32, 256, 1);
    PassManager pm;
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
    const auto pr = pm.run(k, ctx);
    assert(pr.ok && !pr.changed);
    for (const auto& op : k.ops) {
      if (!op->erased && (op->kind == OpKind::Load || op->kind == OpKind::Store)) {
        assert(std::get<LoadStoreAttr>(op->attr).vec_width == 1);
      }
    }
  }

  // Negative 2: strided access (arange * 2) -> stays vec_width=1
  {
    Module mod;
    auto& k = mod.add_kernel("strided");
    IRBuilder b(k);
    const ValueId ptr = b.add_param("p", Type::ptr(ScalarDType::F32), false, 0, 16, true);
    const ValueId offs = b.emit_binary(BinaryOp::Mul, b.emit_arange(0, 64), b.emit_const_i32(2));
    const ValueId v = b.emit_load(b.emit_ptr_add(ptr, offs));
    b.emit_store(b.emit_ptr_add(ptr, offs), v);
    b.emit_return();
    PassManager pm;
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
    const auto pr = pm.run(k, ctx);
    assert(pr.ok && !pr.changed);
  }

  // Negative 3: tile fastest dim not divisible by 4 (BLOCK=6) -> stays vec_width=1
  {
    Module mod;
    auto& k = mod.add_kernel("non_mult4");
    build_vector_add_kernel(k, ScalarDType::F32, 6, 16);
    PassManager pm;
    pm.add_pass<ContiguityAndAlignmentAnalysis>();
    const auto pr = pm.run(k, ctx);
    assert(pr.ok && !pr.changed);
  }

  std::cout << "[PASS] test_contiguity_analysis\n";
}

void test_tile_layout_and_dce_and_m1_pipeline() {
  const auto target = TargetSpec::apple_m1_max();
  std::ostringstream dump;
  PassContext ctx{target, true, true, &dump};
  PassManager pm = PassManager::build_m1_pipeline();

  // 1. 1D vector_add with redundant ops: B=256, vec_width=4 -> tg={64, 1, 1}, ept={4, 1}
  {
    Module mod;
    auto& k = mod.add_kernel("vector_add_256");
    build_vector_add_kernel(k, ScalarDType::F32, 256, 16, /*add_redundant=*/true);
    const size_t before_ops = k.block(k.entry_block).ops.size();
    const auto pr = pm.run(k, ctx);
    if (!pr.ok) std::cerr << pr.diagnostic << "\n";
    assert(pr.ok && pr.changed);
    assert(k.block(k.entry_block).ops.size() < before_ops);
    assert(IRVerifier::verify(k, target, VerifyStage::PreCodegen).ok);

    bool checked_load = false;
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Load) {
        checked_load = true;
        assert(std::get<LoadStoreAttr>(op->attr).vec_width == 4);
        const auto& layout = k.value(op->results[0]).type.as_tile().layout;
        assert(layout.kind == LayoutKind::Blocked);
        assert(layout.vec_width == 4);
        assert(layout.tg_threads[0] == 64 && layout.tg_threads[1] == 1 && layout.tg_threads[2] == 1);
        assert(layout.elems_per_thread[0] == 4 && layout.elems_per_thread[1] == 1);
      }
    }
    assert(checked_load);
  }

  // 1b. 1D vector_add with B=1024, vec_width=4 -> tg={256, 1, 1}, ept={4, 1}
  {
    Module mod;
    auto& k = mod.add_kernel("vector_add_1024");
    build_vector_add_kernel(k, ScalarDType::F32, 1024, 16);
    PassContext qctx{target, true, false, nullptr};
    assert(pm.run(k, qctx).ok);
    assert(IRVerifier::verify(k, target, VerifyStage::PreCodegen).ok);
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Load) {
        const auto& layout = k.value(op->results[0]).type.as_tile().layout;
        assert(layout.tg_threads[0] == 256 && layout.tg_threads[1] == 1 && layout.tg_threads[2] == 1);
        assert(layout.elems_per_thread[0] == 4 && layout.elems_per_thread[1] == 1);
        assert(layout.vec_width == 4);
      }
    }
  }

  // 2. 2D matmul [64, 64] -> tg={16, 16, 1}, ept={4, 4}, loop-carried A/B loads vec4, LICM hoisted b_step
  {
    Module mod;
    auto& k = mod.add_kernel("matmul_64x64");
    build_matmul_kernel(k, 64, 64, 16);
    PassContext qctx{target, true, false, nullptr};
    const auto pr = pm.run(k, qctx);
    if (!pr.ok) std::cerr << pr.diagnostic << "\n";
    assert(pr.ok && pr.changed);
    assert(IRVerifier::verify(k, target, VerifyStage::PreCodegen).ok);

    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Dot) {
        const auto& layout = k.value(op->results[0]).type.as_tile().layout;
        assert(layout.kind == LayoutKind::Blocked);
        assert(layout.tg_threads[0] == 16 && layout.tg_threads[1] == 16 && layout.tg_threads[2] == 1);
        assert(layout.elems_per_thread[0] == 4 && layout.elems_per_thread[1] == 4);
      }
      if (!op->erased && (op->kind == OpKind::Load || op->kind == OpKind::Store)) {
        assert(std::get<LoadStoreAttr>(op->attr).vec_width == 4);
      }
    }
  }

  // 3. Small 2D matmul [16, 16] (BM*BN == 256 <= 256) -> tg={8, 8, 1}, ept={2, 2}
  {
    Module mod;
    auto& k = mod.add_kernel("matmul_16x16");
    build_matmul_kernel(k, 16, 16, 16);
    PassContext qctx{target, true, false, nullptr};
    assert(pm.run(k, qctx).ok);
    for (const auto& op : k.ops) {
      if (!op->erased && op->kind == OpKind::Dot) {
        const auto& layout = k.value(op->results[0]).type.as_tile().layout;
        assert(layout.tg_threads[0] == 8 && layout.tg_threads[1] == 8 && layout.tg_threads[2] == 1);
        assert(layout.elems_per_thread[0] == 2 && layout.elems_per_thread[1] == 2);
      }
    }
  }

  // 4. Standalone DCE preserves side effects & removes dead pure loop + dead pure ops
  {
    Module mod;
    auto& k = mod.add_kernel("dce_test");
    IRBuilder b(k);
    const ValueId out = b.add_param("out", Type::ptr(ScalarDType::F32), false, 0, 16, true);
    const ValueId dead1 = b.emit_const_f32(42.0f);
    const ValueId dead2 = b.emit_unary(UnaryOp::Exp, dead1);
    (void)dead2;

    // Pure dead loop whose result is never used
    const BlockId entry = b.current_block();
    const auto dead_loop =
        b.begin_for_loop(b.emit_const_i32(0), b.emit_const_i32(4), b.emit_const_i32(1), {b.emit_const_f32(1.0f)});
    b.end_for_loop(dead_loop, {b.emit_unary(UnaryOp::Neg, dead_loop.iter_args[0])}, entry);

    // Side-effecting loop must be preserved
    const auto live_loop = b.begin_for_loop(b.emit_const_i32(0), b.emit_const_i32(4), b.emit_const_i32(1), {});
    b.emit_store(out, b.emit_const_f32(1.0f));
    b.end_for_loop(live_loop, {}, entry);

    k.create_op(OpKind::Barrier, b.current_block(), {}, {}, BarrierAttr{BarrierScope::Threadgroup});
    b.emit_return();

    PassManager dce_pm;
    dce_pm.add_pass<DCEPass>();
    PassContext qctx{target, true, false, nullptr};
    assert(dce_pm.run(k, qctx).ok);
    assert(count_active_ops(k, OpKind::Unary) == 0);
    assert(count_active_ops(k, OpKind::ForLoop) == 1);
    assert(count_active_ops(k, OpKind::Store) == 1);
    assert(count_active_ops(k, OpKind::Barrier) == 1);
  }

  assert(!dump.str().empty());
  std::cout << "[PASS] test_tile_layout_and_dce_and_m1_pipeline\n";
}

}  // namespace

int main() {
  test_types_and_attrs();
  test_builder_and_verifier_positive();
  test_verifier_negative_cases();
  test_printer_parser_roundtrip();
  test_canonicalize_pass();
  test_cse_pass();
  test_licm_pass();
  test_contiguity_analysis();
  test_tile_layout_and_dce_and_m1_pipeline();
  std::cout << "ALL IR & PASS TESTS PASSED!\n";
  return 0;
}
