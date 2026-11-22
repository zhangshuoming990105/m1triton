#include "codegen/metal/l3_gemm_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/builder.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/verifier.hpp"

#include <cassert>
#include <iostream>
#include <string>

namespace {

using namespace m1triton::ir;
using namespace m1triton::codegen::metal;

void assert_contains(const std::string& haystack, const std::string& needle) {
  if (haystack.find(needle) == std::string::npos) {
    std::cerr << "Expected MSL to contain: " << needle << "\nActual MSL:\n" << haystack << "\n";
    assert(false);
  }
}

void test_grouped_gemm_persistent_fp32() {
  Module mod;
  Kernel& k = mod.add_kernel("grouped_gemm_persistent_f32");
  IRBuilder b(k);
  auto ptr_f32 = Type::ptr(ScalarDType::F32);
  auto ptr_i32 = Type::ptr(ScalarDType::I32);
  auto i32 = Type::scalar(ScalarDType::I32);
  ValueId a = b.add_param("a", ptr_f32);
  ValueId w = b.add_param("w", ptr_f32);
  ValueId off = b.add_param("offsets", ptr_i32);
  ValueId out = b.add_param("out", ptr_f32, false, 0, 1, true);
  ValueId e = b.add_param("E", i32);
  ValueId n = b.add_param("N", i32);
  ValueId kk = b.add_param("K", i32);
  b.add_param("BLOCK_M", i32, true, 64);
  b.add_param("BLOCK_N", i32, true, 64);
  b.add_param("BLOCK_K", i32, true, 32);
  b.add_param("NUM_SG", i32, true, 4);

  GroupedGemmAttr gg;
  gg.bm = 64;
  gg.bn = 64;
  gg.bk = 16;
  gg.num_simdgroups = 4;
  gg.transpose_b = true;
  gg.persistent = true;
  b.emit_grouped_gemm({a, w, off, out, e, n, kk}, gg);
  b.emit_return();

  auto target = TargetSpec::apple_m1_max();
  auto vr = IRVerifier::verify(mod, target);
  assert(vr.ok);
  std::string txt = IRPrinter::print(mod);
  auto parsed = IRParser::parse(txt);
  assert(parsed.ok && parsed.module != nullptr);
  assert(IRVerifier::verify(*parsed.module, target).ok);

  auto msl = MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 128u);
  assert(msl.launch_spec.tg_x <= target.max_threads_per_threadgroup);
  assert(msl.launch_spec.param_slots.size() == 7u);
  assert(msl.launch_spec.param_slots[3].writable);
  assert_contains(msl.msl_source, "simdgroup_float8x8");
  assert_contains(msl.msl_source, "simdgroup_multiply_accumulate");
  assert_contains(msl.msl_source, "m1_mma2");
  assert_contains(msl.msl_source, "m1_tile_idx += m1_grid_dim.x");
  assert_contains(msl.msl_source, "m1_sa[2304]");
  assert_contains(msl.msl_source, "m1_sb[2304]");
}

void test_grouped_gemm_non_persistent_nt_fp16() {
  Module mod;
  Kernel& k = mod.add_kernel("grouped_gemm_non_persistent_f16");
  IRBuilder b(k);
  auto ptr_f16 = Type::ptr(ScalarDType::F16);
  auto ptr_i32 = Type::ptr(ScalarDType::I32);
  auto i32 = Type::scalar(ScalarDType::I32);
  ValueId a = b.add_param("a", ptr_f16);
  ValueId w = b.add_param("w", ptr_f16);
  ValueId off = b.add_param("offsets", ptr_i32);
  ValueId out = b.add_param("out", ptr_f16, false, 0, 1, true);
  ValueId e = b.add_param("E", i32);
  ValueId n = b.add_param("N", i32);
  ValueId kk = b.add_param("K", i32);

  GroupedGemmAttr gg;
  gg.bm = 64;
  gg.bn = 128;
  gg.bk = 32;
  gg.num_simdgroups = 8;
  gg.transpose_b = false;
  gg.persistent = false;
  b.emit_grouped_gemm({a, w, off, out, e, n, kk}, gg);
  b.emit_return();

  auto target = TargetSpec::apple_m1_max();
  auto msl = MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 256u);
  assert_contains(msl.msl_source, "simdgroup_half8x8");
  assert_contains(msl.msl_source, "m1_mma2_hf");
  assert_contains(msl.msl_source, "m1_grid_dim.z > 1u");
}

void test_quant_dot_w4a16_asymmetric() {
  Module mod;
  Kernel& k = mod.add_kernel("quant_dot_w4a16");
  IRBuilder b(k);
  auto ptr_f16 = Type::ptr(ScalarDType::F16);
  auto ptr_i32 = Type::ptr(ScalarDType::I32);
  auto i32 = Type::scalar(ScalarDType::I32);
  ValueId a = b.add_param("a", ptr_f16);
  ValueId qw = b.add_param("qweight", ptr_i32);
  ValueId sc = b.add_param("scales", ptr_f16);
  ValueId zp = b.add_param("zeros", ptr_f16);
  ValueId out = b.add_param("out", ptr_f16, false, 0, 1, true);
  ValueId m = b.add_param("M", i32);
  ValueId n = b.add_param("N", i32);
  ValueId kk = b.add_param("K", i32);
  b.add_param("BLOCK_M", i32, true, 64);
  b.add_param("BLOCK_N", i32, true, 128);
  b.add_param("BLOCK_K", i32, true, 32);
  b.add_param("NUM_SG", i32, true, 8);

  QuantDotAttr qd;
  qd.bits = 4;
  qd.group_size = 128;
  qd.bm = 64;
  qd.bn = 64;
  qd.bk = 32;
  qd.num_simdgroups = 4;
  qd.symmetric = false;
  b.emit_quant_dot({a, qw, sc, zp, out, m, n, kk}, qd);
  b.emit_return();

  auto target = TargetSpec::apple_m1_max();
  auto vr = IRVerifier::verify(mod, target);
  assert(vr.ok);
  std::string txt = IRPrinter::print(mod);
  auto parsed = IRParser::parse(txt);
  assert(parsed.ok && parsed.module != nullptr);
  assert(IRVerifier::verify(*parsed.module, target).ok);

  auto msl = MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 256u);
  assert(msl.launch_spec.param_slots.size() == 8u);
  assert(msl.launch_spec.param_slots[4].writable);
  assert_contains(msl.msl_source, "simdgroup_half8x8");
  assert_contains(msl.msl_source, "simdgroup_float8x8");
  assert_contains(msl.msl_source, "m1_mma2_hf");
  assert_contains(msl.msl_source, "& 0xFu");
  assert_contains(msl.msl_source, "packed_uint2");
}

void test_quant_dot_w4a32_symmetric_and_limits() {
  Module mod;
  Kernel& k = mod.add_kernel("quant_dot_w4a32_sym");
  IRBuilder b(k);
  auto ptr_f32 = Type::ptr(ScalarDType::F32);
  auto ptr_u32 = Type::ptr(ScalarDType::U32);
  auto i32 = Type::scalar(ScalarDType::I32);
  ValueId a = b.add_param("a", ptr_f32);
  ValueId qw = b.add_param("qweight", ptr_u32);
  ValueId sc = b.add_param("scales", ptr_f32);
  ValueId zp = b.add_param("zeros", ptr_f32);
  ValueId out = b.add_param("out", ptr_f32, false, 0, 1, true);
  ValueId m = b.add_param("M", i32);
  ValueId n = b.add_param("N", i32);
  ValueId kk = b.add_param("K", i32);

  QuantDotAttr qd;
  qd.bits = 4;
  qd.group_size = 64;
  qd.bm = 64;
  qd.bn = 64;
  qd.bk = 32;
  qd.num_simdgroups = 4;
  qd.symmetric = true;
  b.emit_quant_dot({a, qw, sc, zp, out, m, n, kk}, qd);
  b.emit_return();

  auto target = TargetSpec::apple_m1_max();
  auto msl = MetalCodegen::emit(k, target);
  assert(msl.ok);
  assert(msl.launch_spec.tg_x == 128u);
  assert_contains(msl.msl_source, "float2(8.0f)");
  assert_contains(msl.msl_source, "m1_mma2(");

  // Verify SRAM budget rejection (> 32 KB)
  Module bad_sram_mod;
  Kernel& k_bad = bad_sram_mod.add_kernel("quant_dot_over_sram");
  IRBuilder b2(k_bad);
  ValueId a2 = b2.add_param("a", ptr_f32);
  ValueId qw2 = b2.add_param("qweight", ptr_u32);
  ValueId sc2 = b2.add_param("scales", ptr_f32);
  ValueId zp2 = b2.add_param("zeros", ptr_f32);
  ValueId out2 = b2.add_param("out", ptr_f32, false, 0, 1, true);
  ValueId m2 = b2.add_param("M", i32);
  ValueId n2 = b2.add_param("N", i32);
  ValueId kk2 = b2.add_param("K", i32);
  QuantDotAttr qd_big;
  qd_big.bm = 128;
  qd_big.bn = 128;
  qd_big.bk = 64;
  qd_big.num_simdgroups = 8;
  b2.emit_quant_dot({a2, qw2, sc2, zp2, out2, m2, n2, kk2}, qd_big);
  b2.emit_return();

  bool matched = false;
  auto bad_res = try_emit_l3_gemm_kernel(k_bad, target, &matched);
  assert(matched && !bad_res.ok);

  // Verify max_threads_per_threadgroup rejection (> 1024 threads)
  Module bad_tg_mod;
  Kernel& k_tg = bad_tg_mod.add_kernel("grouped_gemm_over_tg");
  IRBuilder b3(k_tg);
  ValueId a3 = b3.add_param("a", ptr_f32);
  ValueId w3 = b3.add_param("w", ptr_f32);
  ValueId off3 = b3.add_param("offsets", Type::ptr(ScalarDType::I32));
  ValueId out3 = b3.add_param("out", ptr_f32, false, 0, 1, true);
  ValueId e3 = b3.add_param("E", i32);
  ValueId n3 = b3.add_param("N", i32);
  ValueId kk3 = b3.add_param("K", i32);
  GroupedGemmAttr gg_tg;
  gg_tg.bm = 64;
  gg_tg.bn = 64;
  gg_tg.bk = 16;
  gg_tg.num_simdgroups = 64;  // 2048 threads > 1024
  b3.emit_grouped_gemm({a3, w3, off3, out3, e3, n3, kk3}, gg_tg);
  b3.emit_return();

  matched = false;
  auto bad_tg = try_emit_l3_gemm_kernel(k_tg, target, &matched);
  assert(matched && !bad_tg.ok);

  // Verify QuantDot with group_size < BK (multiple groups per K tile) and unsupported bits rejection
  Module multi_grp_mod;
  Kernel& k_mg = multi_grp_mod.add_kernel("quant_dot_multi_group");
  IRBuilder b4(k_mg);
  ValueId a4 = b4.add_param("a", Type::ptr(ScalarDType::F16));
  ValueId qw4 = b4.add_param("qweight", ptr_u32);
  ValueId sc4 = b4.add_param("scales", Type::ptr(ScalarDType::F16));
  ValueId zp4 = b4.add_param("zeros", Type::ptr(ScalarDType::F16));
  ValueId out4 = b4.add_param("out", Type::ptr(ScalarDType::F16), false, 0, 1, true);
  ValueId m4 = b4.add_param("M", i32);
  ValueId n4 = b4.add_param("N", i32);
  ValueId kk4 = b4.add_param("K", i32);
  QuantDotAttr qd_mg;
  qd_mg.bits = 4;
  qd_mg.group_size = 32;
  qd_mg.bm = 64;
  qd_mg.bn = 64;
  qd_mg.bk = 64;
  qd_mg.num_simdgroups = 4;
  b4.emit_quant_dot({a4, qw4, sc4, zp4, out4, m4, n4, kk4}, qd_mg);
  b4.emit_return();
  auto mg_res = MetalCodegen::emit(k_mg, target);
  assert(mg_res.ok);
  assert_contains(mg_res.msl_source, "* 8u) / 32u");

  Module bad_bits_mod;
  Kernel& k_bits = bad_bits_mod.add_kernel("quant_dot_bad_bits");
  IRBuilder b5(k_bits);
  ValueId a5 = b5.add_param("a", Type::ptr(ScalarDType::F16));
  ValueId qw5 = b5.add_param("qweight", ptr_u32);
  ValueId sc5 = b5.add_param("scales", Type::ptr(ScalarDType::F16));
  ValueId zp5 = b5.add_param("zeros", Type::ptr(ScalarDType::F16));
  ValueId out5 = b5.add_param("out", Type::ptr(ScalarDType::F16), false, 0, 1, true);
  ValueId m5 = b5.add_param("M", i32);
  ValueId n5 = b5.add_param("N", i32);
  ValueId kk5 = b5.add_param("K", i32);
  QuantDotAttr qd_bad;
  qd_bad.bits = 8;
  b5.emit_quant_dot({a5, qw5, sc5, zp5, out5, m5, n5, kk5}, qd_bad);
  b5.emit_return();
  matched = false;
  auto bad_bits = try_emit_l3_gemm_kernel(k_bits, target, &matched);
  assert(matched && !bad_bits.ok);
}

}  // namespace

int main() {
  test_grouped_gemm_persistent_fp32();
  test_grouped_gemm_non_persistent_nt_fp16();
  test_quant_dot_w4a16_asymmetric();
  test_quant_dot_w4a32_symmetric_and_limits();
  std::cout << "test_l3_gemm_emitter ALL PASS\n";
  return 0;
}
