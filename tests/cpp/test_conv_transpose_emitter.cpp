#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "codegen/metal/conv_transpose_emitter.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/builder.hpp"
#include "ir/target_spec.hpp"
#include "ir/verifier.hpp"

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

struct ConvTransSpec {
  uint32_t ndim = 2;
  std::vector<int32_t> k = {3, 3};
  std::vector<int32_t> s = {1, 1};
  std::vector<int32_t> p = {0, 0};
  std::vector<int32_t> d = {1, 1};
  std::vector<int32_t> op = {0, 0};
  int32_t groups = 1;
  bool has_bias = false;
  ir::ScalarDType dtype = ir::ScalarDType::F32;
  int64_t c_in = 0;
  int64_t c_out = 0;
  int64_t bm = 0;
  int64_t bn = 0;
  int64_t bk = 0;
  int64_t num_sg = 0;
  int64_t w_blk8 = 0;
  int64_t halo = 0;
  std::vector<int64_t> tiles;
  bool transposed = true;
};

void build_conv_transpose_kernel(ir::Kernel& k, const ConvTransSpec& spec) {
  ir::IRBuilder b(k);
  auto x = b.add_param("x_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto w = b.add_param("w_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false);
  auto o = b.add_param("out_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, true);
  std::vector<ir::ValueId> operands{x, w, o};
  if (spec.has_bias) {
    operands.push_back(b.add_param("b_ptr", ir::Type::ptr(spec.dtype), false, 0, 16, false));
  }
  operands.push_back(b.add_param("N", ir::Type::scalar(ir::ScalarDType::I32)));
  if (spec.c_in > 0) {
    operands.push_back(b.add_param("C_IN", ir::Type::scalar(ir::ScalarDType::I32), true, spec.c_in));
  } else {
    operands.push_back(b.add_param("C_in", ir::Type::scalar(ir::ScalarDType::I32)));
  }
  if (spec.c_out > 0) {
    operands.push_back(b.add_param("C_OUT", ir::Type::scalar(ir::ScalarDType::I32), true, spec.c_out));
  } else {
    operands.push_back(b.add_param("C_out", ir::Type::scalar(ir::ScalarDType::I32)));
  }
  const char* in_names[3] = {"D_in", "H_in", "W_in"};
  const char* out_names[3] = {"D_out", "H_out", "W_out"};
  for (uint32_t i = 0; i < spec.ndim; ++i) {
    operands.push_back(b.add_param(in_names[3 - spec.ndim + i], ir::Type::scalar(ir::ScalarDType::I32)));
  }
  for (uint32_t i = 0; i < spec.ndim; ++i) {
    operands.push_back(b.add_param(out_names[3 - spec.ndim + i], ir::Type::scalar(ir::ScalarDType::I32)));
  }
  if (spec.bm > 0) b.add_param("BLOCK_M", ir::Type::scalar(ir::ScalarDType::I32), true, spec.bm);
  if (spec.bn > 0) b.add_param("BLOCK_N", ir::Type::scalar(ir::ScalarDType::I32), true, spec.bn);
  if (spec.bk > 0) b.add_param("BLOCK_K", ir::Type::scalar(ir::ScalarDType::I32), true, spec.bk);
  if (spec.num_sg > 0) b.add_param("NUM_SG", ir::Type::scalar(ir::ScalarDType::I32), true, spec.num_sg);
  if (spec.w_blk8 > 0) b.add_param("W_BLK8", ir::Type::scalar(ir::ScalarDType::I32), true, spec.w_blk8);
  if (spec.halo > 0) b.add_param("HALO", ir::Type::scalar(ir::ScalarDType::I32), true, spec.halo);
  const char* tile_names[3] = {"TILE_D", "TILE_H", "TILE_W"};
  for (uint32_t i = 0; i < spec.tiles.size() && i < spec.ndim; ++i) {
    b.add_param(tile_names[3 - spec.ndim + i], ir::Type::scalar(ir::ScalarDType::I32), true, spec.tiles[i]);
  }

  ir::ConvAttr attr;
  attr.ndim = spec.ndim;
  attr.kernel_size = spec.k;
  attr.stride = spec.s;
  attr.padding = spec.p;
  attr.dilation = spec.d;
  attr.output_padding = spec.op;
  attr.groups = spec.groups;
  attr.transposed = spec.transposed;
  attr.has_bias = spec.has_bias;
  b.emit_implicit_gemm_conv(operands, attr);
  b.emit_return();
}

codegen::metal::CompiledMSL emit_checked(const ir::Kernel& k) {
  const auto target = ir::TargetSpec::apple_m1_max();
  auto vr = ir::IRVerifier::verify(k, target);
  if (!vr.ok) {
    std::cerr << "Verify error: " << vr.summary() << "\n";
    assert(false);
  }
  bool matched = false;
  auto res = codegen::metal::try_emit_conv_transpose_kernel(k, target, &matched);
  assert(matched);
  if (!res.ok) {
    std::cerr << "Emit error (" << k.name << "): " << res.error << "\n";
    assert(false);
  }
  return res;
}

void test_2d_halo_and_router() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("ct2d_halo");
  ConvTransSpec spec;
  spec.ndim = 2;
  spec.k = {3, 3};
  spec.s = {1, 1};
  spec.p = {0, 0};
  spec.d = {1, 1};
  spec.c_in = 64;
  spec.c_out = 64;
  spec.bm = 64;
  spec.bn = 64;
  spec.bk = 16;
  spec.num_sg = 4;
  spec.w_blk8 = 1;
  spec.halo = 1;
  spec.tiles = {8, 8};
  build_conv_transpose_kernel(k, spec);

  auto res = emit_checked(k);
  assert(res.launch_spec.tg_x == 128 && res.launch_spec.tg_y == 1 && res.launch_spec.tg_z == 1);
  // Runtime params: x_ptr, w_ptr, out_ptr, N, H_in, W_in, H_out, W_out (8 slots; constexprs omitted).
  assert(res.launch_spec.param_slots.size() == 8);
  assert(!res.launch_spec.param_slots[0].writable);
  assert(!res.launch_spec.param_slots[1].writable);
  assert(res.launch_spec.param_slots[2].writable);
  assert_contains(res.msl_source, "#include <metal_simdgroup_matrix>");
  assert_contains(res.msl_source, "simdgroup_multiply_accumulate");
  assert_contains(res.msl_source, "constexpr int m1_TD = 1, m1_TH = 8, m1_TW = 8;");
  assert_contains(res.msl_source, "constexpr int m1_XP = 100, m1_XPL = 100;");
  assert_not_contains(res.msl_source, "atomic_fetch_add");

  // Also confirm top-level metal_codegen routes transposed ImplicitGemmConv here.
  auto routed = codegen::metal::MetalCodegen::emit(k, ir::TargetSpec::apple_m1_max());
  assert(routed.ok);
  assert(routed.msl_source == res.msl_source);
}

void test_1d_8sg_and_bias_3d_strided_grouped() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("ct1d_8sg");
    ConvTransSpec spec;
    spec.ndim = 1;
    spec.k = {3};
    spec.s = {2};
    spec.p = {1};
    spec.d = {2};
    spec.op = {0};
    spec.c_in = 128;
    spec.c_out = 128;
    spec.bm = 128;
    spec.bn = 64;
    spec.bk = 32;
    spec.num_sg = 8;
    spec.w_blk8 = 1;
    spec.halo = 1;
    spec.tiles = {64};
    build_conv_transpose_kernel(k, spec);

    auto res = emit_checked(k);
    assert(res.launch_spec.tg_x == 256);
    assert_contains(res.msl_source, "m1_SW = 2;");
    assert_contains(res.msl_source, "m1_DW = 2;");
    assert_contains(res.msl_source, "const long m1_n =");
  }

  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("ct3d_strided_grouped_bias");
    ConvTransSpec spec;
    spec.ndim = 3;
    spec.k = {3, 5, 7};
    spec.s = {2, 2, 2};
    spec.p = {1, 2, 3};
    spec.d = {1, 1, 1};
    spec.op = {1, 1, 1};
    spec.groups = 4;
    spec.has_bias = true;
    spec.c_in = 32;
    spec.c_out = 32;
    spec.bm = 8;
    spec.bn = 64;
    spec.bk = 8;
    spec.num_sg = 4;
    spec.w_blk8 = 1;
    spec.halo = 1;
    spec.tiles = {2, 4, 8};
    build_conv_transpose_kernel(k, spec);

    auto res = emit_checked(k);
    // x, w, out, b, N, D/H/W_in, D/H/W_out -> 11 runtime buffer slots.
    assert(res.launch_spec.param_slots.size() == 11);
    assert_contains(res.msl_source, "constexpr long m1_G = 4;");
    assert_contains(res.msl_source, "b_ptr[m1_g * m1_Cog + m1_co]");
    assert_not_contains(res.msl_source, "atomic_fetch_add");
  }
}

void test_general_fallback_and_f16() {
  ir::Module mod;
  ir::Kernel& k = mod.add_kernel("ct2d_general_f16");
  ConvTransSpec spec;
  spec.ndim = 2;
  spec.k = {3, 5};
  spec.s = {2, 3};
  spec.p = {1, 2};
  spec.d = {2, 1};
  spec.groups = 2;
  spec.has_bias = true;
  spec.dtype = ir::ScalarDType::F16;
  // Runtime C_in/C_out, default attr bm/bn/bk/num_simdgroups (64/64/16/4), raw PyTorch weight layout.
  build_conv_transpose_kernel(k, spec);

  auto res = emit_checked(k);
  assert(res.launch_spec.tg_x == 128);
  assert_contains(res.msl_source, "device const half* x_ptr");
  assert_contains(res.msl_source, "device half* out_ptr");
  assert_contains(res.msl_source, "m1_wg[(m1_ci * m1_Cog + m1_co) * m1_KT + m1_tap]");
  assert_contains(res.msl_source, "constexpr uint m1_ALD = 68u, m1_BLD = 68u;");
}

void test_ignores_forward_conv_and_rejects_bad_halo() {
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("fwd_conv");
    ConvTransSpec spec;
    spec.transposed = false;
    build_conv_transpose_kernel(k, spec);
    bool matched = true;
    auto res = codegen::metal::try_emit_conv_transpose_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(!matched && !res.ok);
  }
  {
    ir::Module mod;
    ir::Kernel& k = mod.add_kernel("bad_halo_mismatch");
    ConvTransSpec spec;
    spec.ndim = 2;
    spec.c_in = 64;
    spec.c_out = 64;
    spec.bm = 64;
    spec.bn = 64;
    spec.bk = 16;
    spec.num_sg = 4;
    spec.w_blk8 = 1;
    spec.halo = 1;
    spec.tiles = {4, 8};  // 4*8 = 32 != BLOCK_N (64).
    build_conv_transpose_kernel(k, spec);
    bool matched = false;
    auto res = codegen::metal::try_emit_conv_transpose_kernel(k, ir::TargetSpec::apple_m1_max(), &matched);
    assert(matched && !res.ok);
    assert(res.error.find("TILE_D*TILE_H*TILE_W == BLOCK_N") != std::string::npos);
  }
}

}  // namespace

int main() {
  test_2d_halo_and_router();
  test_1d_8sg_and_bias_3d_strided_grouped();
  test_general_fallback_and_f16();
  test_ignores_forward_conv_and_rejects_bad_halo();
  std::cout << "test_conv_transpose_emitter: PASS\n";
  return 0;
}
