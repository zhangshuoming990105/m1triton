#include "codegen/metal/l3_gemm_emitter.hpp"
#include "ir/builder.hpp"
#include "ir/parser.hpp"
#include "ir/printer.hpp"
#include "ir/verifier.hpp"

#include <cassert>
#include <iostream>

int main() {
  using namespace m1triton::ir;
  Module mod;
  Kernel& k = mod.add_kernel("grouped_gemm_roundtrip");
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
  GroupedGemmAttr gg;
  gg.bm = 64;
  gg.bn = 64;
  gg.bk = 16;
  b.emit_grouped_gemm({a, w, off, out, e, n, kk}, gg);
  b.emit_return();

  auto vr = IRVerifier::verify(mod, TargetSpec::apple_m1_max());
  assert(vr.ok);
  std::string txt = IRPrinter::print(mod);
  auto parsed = IRParser::parse(txt);
  assert(parsed.ok && parsed.module != nullptr);
  assert(IRVerifier::verify(*parsed.module, TargetSpec::apple_m1_max()).ok);
  std::cout << "test_l3_gemm_emitter contract PASS\n";
  return 0;
}
