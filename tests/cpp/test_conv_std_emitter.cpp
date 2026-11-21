#include <cassert>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "codegen/metal/conv_std_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"
#include "passes/conv_lowering.hpp"
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
    std::cerr << "Expected MSL to NOT contain:\n  " << needle << "\nActual MSL:\n" << haystack << "\n";
    assert(false);
  }
}

struct ConvSpec {
  uint32_t ndim = 2;
  std::vector<int32_t> k;
  std::vector<int32_t> s;
  std::vector<int32_t> p;
  std::vector<int32_t> d;
  int32_t groups = 1;
  bool has_bias = false;
  bool transposed = false;
  bool is_depthwise = false;
  int64_t cin_constexpr = 0;
  int64_t cout_constexpr = 0;
  int64_t block_m = 0;
  int64_t block_n = 0;
  int64_t block_k = 0;
  int64_t num_sg = 0;
  ir::ScalarDType elem_dtype = ir::ScalarDType::F32;
  std::optional<ir::TileSpecAttr> hint = std::nullopt;
};

ir::OpId build_conv_kernel(ir::Kernel& k, const ConvSpec& spec) {
  ir::IRBuilder b(k);
  const auto ptr_ty = ir::Type::ptr(spec.elem_dtype);
  const auto i32 = ir::Type::scalar(ir::ScalarDType::I32);
  std::vector<ir::ValueId> operands{b.add_param("x_ptr", ptr_ty, false, 0, 16, false),
                                    b.add_param("w_ptr", ptr_ty, false, 0, 16, false),
                                    b.add_param("out_ptr", ptr_ty, false, 0, 16, true)};
  if (spec.has_bias) operands.push_back(b.add_param("b_ptr", ptr_ty, false, 0, 16, false));
  operands.push_back(b.add_param("N", i32));
  if (spec.cin_constexpr > 0) {
    operands.push_back(b.add_param("C_in", i32, true, spec.cin_constexpr));
  } else {
    operands.push_back(b.add_param("C_in", i32));
  }
  if (spec.cout_constexpr > 0) {
    operands.push_back(b.add_param("C_out", i32, true, spec.cout_constexpr));
  } else {
    operands.push_back(b.add_param("C_out", i32));
  }
  const char* in_names[3] = {"D_in", "H_in", "W_in"};
  const char* out_names[3] = {"D_out", "H_out", "W_out"};
  for (uint32_t i = 0; i < spec.ndim; ++i) operands.push_back(b.add_param(in_names[3 - spec.ndim + i], i32));
  for (uint32_t i = 0; i < spec.ndim; ++i) operands.push_back(b.add_param(out_names[3 - spec.ndim + i], i32));
  if (spec.block_m > 0) b.add_param("BLOCK_M", i32, true, spec.block_m);
  if (spec.block_n > 0) b.add_param("BLOCK_N", i32, true, spec.block_n);
  if (spec.block_k > 0) b.add_param("BLOCK_K", i32, true, spec.block_k);
  if (spec.num_sg > 0) b.add_param("NUM_SG", i32, true, spec.num_sg);
  ir::ConvAttr attr;
  attr.ndim = spec.ndim;
  attr.kernel_size = spec.k;
  attr.stride = spec.s;
  attr.padding = spec.p;
  attr.dilation = spec.d;
  attr.groups = spec.groups;
  attr.has_bias = spec.has_bias;
  attr.transposed = spec.transposed;
  attr.is_depthwise = spec.is_depthwise;
  ir::OpHints hints;
  hints.tile_spec = spec.hint;
  const ir::OpId conv = b.emit_implicit_gemm_conv(operands, attr, hints);
  b.emit_return();
  return conv;
}

const ir::ConvAttr& lower(ir::Kernel& k, ir::OpId conv) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  passes::ConvLoweringPass pass;
  passes::PassContext ctx{target};
  auto pr = pass.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "ConvLoweringPass error (" << k.name << "): " << pr.diagnostic << "\n";
    assert(false);
  }
  return std::get<ir::ConvAttr>(k.op(conv).attr);
}

std::string emit(const ir::Kernel& k, codegen::metal::CompiledMSL* out = nullptr) {
  bool matched = false;
  auto res = codegen::metal::try_emit_conv_std_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error: " << res.error << "\n";
    assert(false);
  }
  if (out != nullptr) *out = res;
  return res.msl_source;
}

uint32_t gemm_smem(const ir::ConvAttr& ca) { return (ca.bm * (ca.bk + 4) + ca.bk * (ca.bn + 4)) * 4; }

void test_pass_default_schedule_and_normalization() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv2d_default");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3};
  spec.s = {1};
  auto conv = build_conv_kernel(k, spec);
  const auto& ca = lower(k, conv);
  assert((ca.kernel_size == std::vector<int32_t>{3, 3}));
  assert((ca.stride == std::vector<int32_t>{1, 1}));
  assert((ca.padding == std::vector<int32_t>{0, 0}));
  assert((ca.dilation == std::vector<int32_t>{1, 1}));
  assert((ca.output_padding == std::vector<int32_t>{0, 0}));
  assert(ca.bm == 64 && ca.bn == 64 && ca.bk == 16 && ca.num_simdgroups == 4);
  assert(!ca.is_pointwise && !ca.use_sram_halo && !ca.has_bias);
  assert(gemm_smem(ca) <= 32768);
}

void test_pass_constexpr_channel_schedules() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv2d_c96");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {11, 11};
  spec.s = {4, 4};
  spec.p = {2, 2};
  spec.d = {1, 1};
  spec.has_bias = true;
  spec.cin_constexpr = 3;
  spec.cout_constexpr = 96;
  const auto& ca = lower(k, build_conv_kernel(k, spec));
  assert(ca.bm == 32 && ca.bn == 64 && ca.bk == 16 && ca.num_simdgroups == 4 && ca.has_bias);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("conv2d_large_8sg");
  ConvSpec big;
  big.ndim = 2;
  big.k = {3, 3};
  big.s = {1, 1};
  big.p = {0, 0};
  big.d = {1, 1};
  big.cin_constexpr = 64;
  big.cout_constexpr = 128;
  const auto& ca2 = lower(k2, build_conv_kernel(k2, big));
  assert(ca2.bm == 128 && ca2.bn == 64 && ca2.bk == 32 && ca2.num_simdgroups == 8);
  assert(gemm_smem(ca2) <= 32768);

  ir::Module mod3;
  ir::Kernel& k3 = mod3.add_kernel("conv2d_pw128");
  ConvSpec pw;
  pw.ndim = 2;
  pw.k = {1, 1};
  pw.s = {1, 1};
  pw.p = {0, 0};
  pw.d = {1, 1};
  pw.cin_constexpr = 64;
  pw.cout_constexpr = 128;
  const auto& ca3 = lower(k3, build_conv_kernel(k3, pw));
  assert(ca3.is_pointwise);
  assert(ca3.bm == 64 && ca3.bn == 64 && ca3.bk == 32 && ca3.num_simdgroups == 4);
  assert(gemm_smem(ca3) <= 32768);

  ir::Module mod4;
  ir::Kernel& k4 = mod4.add_kernel("conv1d_strided_1x1");
  ConvSpec strided;
  strided.ndim = 1;
  strided.k = {1};
  strided.s = {2};
  strided.p = {0};
  strided.d = {1};
  assert(!lower(k4, build_conv_kernel(k4, strided)).is_pointwise);
}

void test_pass_hint_and_block_params() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv2d_hint");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {1, 1};
  spec.p = {1, 1};
  spec.d = {1, 1};
  ir::TileSpecAttr ts;
  ts.bm = 32;
  ts.bn = 64;
  ts.bk = 8;
  ts.num_simdgroups = 4;
  spec.hint = ts;
  const auto& ca = lower(k, build_conv_kernel(k, spec));
  assert(ca.bm == 32 && ca.bn == 64 && ca.bk == 8 && ca.num_simdgroups == 4);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("conv2d_block_params");
  ConvSpec spec2 = spec;
  spec2.block_m = 16;
  spec2.block_n = 64;
  spec2.block_k = 16;
  spec2.num_sg = 4;
  const auto& ca2 = lower(k2, build_conv_kernel(k2, spec2));
  assert(ca2.bm == 16 && ca2.bn == 64 && ca2.bk == 16 && ca2.num_simdgroups == 4);

  ir::Module mod3;
  ir::Kernel& k3 = mod3.add_kernel("conv2d_oversized_hint");
  ConvSpec spec3 = spec;
  ir::TileSpecAttr big;
  big.bm = 128;
  big.bn = 128;
  big.bk = 64;
  big.num_simdgroups = 4;
  spec3.hint = big;
  const auto& ca3 = lower(k3, build_conv_kernel(k3, spec3));
  assert(gemm_smem(ca3) <= 32768);
  assert(ca3.num_simdgroups == 4 || ca3.num_simdgroups == 8);
}

void test_pass_preserves_transposed_and_depthwise_flags() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("convT");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {2, 2};
  spec.p = {1, 1};
  spec.d = {1, 1};
  spec.transposed = true;
  spec.has_bias = true;
  const auto& ca = lower(k, build_conv_kernel(k, spec));
  assert(ca.transposed && ca.has_bias && !ca.is_depthwise && !ca.is_pointwise);
  assert(ca.bm == 64 && ca.bn == 64 && ca.bk == 16 && ca.num_simdgroups == 4);
  assert(gemm_smem(ca) <= 32768);

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("convdw");
  ConvSpec dw;
  dw.ndim = 2;
  dw.k = {3, 3};
  dw.s = {1, 1};
  dw.p = {1, 1};
  dw.d = {1, 1};
  dw.is_depthwise = true;
  dw.groups = 8;
  const auto& ca2 = lower(k2, build_conv_kernel(k2, dw));
  assert(ca2.is_depthwise && !ca2.transposed && !ca2.has_bias);
  assert(ca2.use_sram_halo);
  assert(ca2.bm * ca2.bn >= 128);

  ir::Module mod3;
  ir::Kernel& k3 = mod3.add_kernel("convdw_huge_dilation");
  ConvSpec dw3;
  dw3.ndim = 2;
  dw3.k = {9, 9};
  dw3.s = {4, 4};
  dw3.p = {0, 0};
  dw3.d = {8, 8};
  dw3.is_depthwise = true;
  assert(!lower(k3, build_conv_kernel(k3, dw3)).use_sram_halo);
}

void test_pass_rejects_invalid_stride() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv_bad");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {0, 1};
  spec.p = {0, 0};
  spec.d = {1, 1};
  (void)build_conv_kernel(k, spec);
  passes::ConvLoweringPass pass;
  const auto target = ir::TargetSpec::apple_m1_max();
  passes::PassContext ctx{target};
  auto pr = pass.run(k, ctx);
  assert(!pr.ok);
  assert(pr.diagnostic.find("stride") != std::string::npos);
}

void test_emit_2d_bias_strided() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv2d_bias");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {11, 11};
  spec.s = {4, 4};
  spec.p = {2, 2};
  spec.d = {1, 1};
  spec.has_bias = true;
  spec.block_m = 32;
  spec.block_n = 64;
  spec.block_k = 16;
  lower(k, build_conv_kernel(k, spec));
  codegen::metal::CompiledMSL res;
  const std::string msl = emit(k, &res);
  assert_contains(msl, "kernel void conv2d_bias(");
  assert_contains(msl, "device const float* x_ptr [[buffer(0)]]");
  assert_contains(msl, "device const float* w_ptr [[buffer(1)]]");
  assert_contains(msl, "device float* out_ptr [[buffer(2)]]");
  assert_contains(msl, "device const float* b_ptr [[buffer(3)]]");
  assert_contains(msl, "constant int& W_out [[buffer(10)]]");
  assert_contains(msl, "constexpr int m1_BM = 32;");
  assert_contains(msl, "constexpr int m1_BN = 64;");
  assert_contains(msl, "constexpr int m1_KD = 1, m1_KH = 11, m1_KW = 11;");
  assert_contains(msl, "simdgroup_float8x8 m1_acc[2][4];");
  assert_contains(msl, "simdgroup_multiply_accumulate(m1_acc[1][3], m1_fa[1], m1_fb[3], m1_acc[1][3]);");
  assert_contains(msl, "const int m1_ih_lo = m1_oh0 * 4 - 2;");
  assert_contains(msl, "m1_ov += float4((float)b_ptr[m1_g * m1_Cog + m1_co]);");
  assert_contains(msl, "simdgroup_store(m1_acc[0][0]");
  assert_contains(msl, "packed_float4");
  assert_contains(msl, "threadgroup_barrier(mem_flags::mem_threadgroup);");
  assert(res.launch_spec.tg_x == 128 && res.launch_spec.tg_y == 1 && res.launch_spec.tg_z == 1);
  assert(res.launch_spec.vec_width == 4);
  assert(res.launch_spec.param_slots.size() == 11);
  assert(res.launch_spec.param_slots[2].writable);
  assert(!res.launch_spec.param_slots[0].writable && !res.launch_spec.param_slots[3].writable);
  assert(res.launch_spec.param_slots[3].name == "b_ptr" && res.launch_spec.param_slots[3].is_pointer);
}

void test_emit_8sg_128x64x32() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv2d_8sg");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {1, 1};
  spec.p = {0, 0};
  spec.d = {1, 1};
  spec.cin_constexpr = 64;
  spec.cout_constexpr = 128;
  lower(k, build_conv_kernel(k, spec));
  codegen::metal::CompiledMSL res;
  const std::string msl = emit(k, &res);
  assert_contains(msl, "constexpr int m1_BM = 128;");
  assert_contains(msl, "constexpr int m1_BN = 64;");
  assert_contains(msl, "constexpr int m1_BK = 32;");
  assert_contains(msl, "simdgroup_float8x8 m1_acc[4][4];");
  assert_contains(msl, "for (uint m1_sb = 0u; m1_sb < 8u;");
  assert_not_contains(msl, "if (m1_KFull < m1_K)");
  assert(res.launch_spec.tg_x == 256);
}

void test_emit_1d_dilated_strided_groups() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv1d_dil");
  ConvSpec spec;
  spec.ndim = 1;
  spec.k = {3};
  spec.s = {3};
  spec.p = {0};
  spec.d = {4};
  spec.groups = 2;
  lower(k, build_conv_kernel(k, spec));
  codegen::metal::CompiledMSL res;
  const std::string msl = emit(k, &res);
  assert_contains(msl, "constexpr int m1_G = 2;");
  assert_contains(msl, "const int m1_iw_lo = m1_ow0 * 3 - 0;");
  assert_contains(msl, "m1_xs[3], (float)m1_xs[6], (float)m1_xs[9]");
  assert_contains(msl, "const int m1_Din = (int)1L, m1_Hin = (int)1L, m1_Win = (int)((long)W_in);");
  assert_not_contains(msl, "b_ptr");
  assert(res.launch_spec.param_slots.size() == 8);
}

void test_emit_3d_asymmetric_kernel_and_f16() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv3d_k331");
  ConvSpec spec;
  spec.ndim = 3;
  spec.k = {3, 3, 1};
  spec.s = {1, 1, 1};
  spec.p = {0, 0, 0};
  spec.d = {1, 1, 1};
  lower(k, build_conv_kernel(k, spec));
  const std::string msl = emit(k);
  assert_contains(msl, "constexpr int m1_KD = 3, m1_KH = 3, m1_KW = 1;");
  assert_contains(msl, "typedef long m1_idx;");
  assert_contains(msl, "simdgroup_float8x8 m1_acc[4][4];");

  ir::Module mod2;
  ir::Kernel& k2 = mod2.add_kernel("conv2d_f16");
  ConvSpec f16;
  f16.ndim = 2;
  f16.k = {3, 3};
  f16.s = {1, 1};
  f16.p = {1, 1};
  f16.d = {1, 1};
  f16.elem_dtype = ir::ScalarDType::F16;
  lower(k2, build_conv_kernel(k2, f16));
  const std::string msl2 = emit(k2);
  assert_contains(msl2, "device const half* x_ptr [[buffer(0)]]");
  assert_contains(msl2, "packed_half4");
}

void test_emit_pointwise_fast_path() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv_pw");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {1, 1};
  spec.s = {1, 1};
  spec.p = {0, 0};
  spec.d = {1, 1};
  spec.cin_constexpr = 64;
  spec.cout_constexpr = 128;
  const auto& ca = lower(k, build_conv_kernel(k, spec));
  assert(ca.is_pointwise);
  const std::string msl = emit(k);
  assert_contains(msl, "m1_xp + (m1_idx)m1_k * m1_PinI");
  assert_contains(msl, "const long m1_Cout = 128L;");
  assert_contains(msl, "constexpr int m1_BM = 64;");
  assert_contains(msl, "constexpr int m1_BK = 32;");
  assert_not_contains(msl, "m1_tg_fast");
}

void test_emit_rejects_unsupported_tile() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv_bad_tile");
  ConvSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {1, 1};
  spec.p = {0, 0};
  spec.d = {1, 1};
  auto conv = build_conv_kernel(k, spec);
  std::get<ir::ConvAttr>(k.op(conv).attr).bm = 48;
  bool matched = false;
  auto res = codegen::metal::try_emit_conv_std_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(matched && !res.ok);
  assert(res.error.find("tile") != std::string::npos);
}

void test_transposed_and_depthwise_unmatched() {
  for (int variant = 0; variant < 2; ++variant) {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("conv_other");
    ConvSpec spec;
    spec.ndim = 2;
    spec.k = {3, 3};
    spec.s = {1, 1};
    spec.p = {0, 0};
    spec.d = {1, 1};
    spec.transposed = variant == 0;
    spec.is_depthwise = variant == 1;
    lower(k, build_conv_kernel(k, spec));
    bool matched = true;
    auto res = codegen::metal::try_emit_conv_std_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched && !res.ok && res.error.empty());
  }
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("noop");
  ir::IRBuilder b(k);
  b.add_param("x_ptr", ir::Type::ptr(ir::ScalarDType::F32));
  b.emit_return();
  bool matched = true;
  auto res = codegen::metal::try_emit_conv_std_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
  assert(!matched && !res.ok && res.error.empty());
}

void test_full_pipeline_routes_conv_std() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("conv3d_pipeline");
  ConvSpec spec;
  spec.ndim = 3;
  spec.k = {3, 5, 7};
  spec.s = {1, 2, 1};
  spec.p = {1, 0, 3};
  spec.d = {1, 1, 2};
  spec.has_bias = true;
  build_conv_kernel(k, spec);
  const auto target = ir::TargetSpec::apple_m1_max();
  auto pm = passes::PassManager::build_m1_pipeline();
  passes::PassContext ctx{target};
  auto pr = pm.run(k, ctx);
  if (!pr.ok) {
    std::cerr << "Pipeline error: " << pr.diagnostic << "\n";
    assert(false);
  }
  codegen::metal::MetalCodegen cg;
  auto res = cg.emit(k, target);
  if (!res.ok) {
    std::cerr << "Codegen error: " << res.error << "\n";
    assert(false);
  }
  assert_contains(res.msl_source, "kernel void conv3d_pipeline(");
  assert_contains(res.msl_source, "constexpr int m1_KD = 3, m1_KH = 5, m1_KW = 7;");
  assert_contains(res.msl_source, "const int m1_ih_lo = m1_oh0 * 2 - 0;");
  assert_contains(res.msl_source, "m1_ov += float4((float)b_ptr[m1_g * m1_Cog + m1_co]);");
  assert(res.launch_spec.tg_x == 128);
}

}  // namespace

int main() {
  test_pass_default_schedule_and_normalization();
  test_pass_constexpr_channel_schedules();
  test_pass_hint_and_block_params();
  test_pass_preserves_transposed_and_depthwise_flags();
  test_pass_rejects_invalid_stride();
  test_emit_2d_bias_strided();
  test_emit_8sg_128x64x32();
  test_emit_1d_dilated_strided_groups();
  test_emit_3d_asymmetric_kernel_and_f16();
  test_emit_pointwise_fast_path();
  test_emit_rejects_unsupported_tile();
  test_transposed_and_depthwise_unmatched();
  test_full_pipeline_routes_conv_std();
  std::cout << "test_conv_std_emitter: PASS\n";
  return 0;
}
