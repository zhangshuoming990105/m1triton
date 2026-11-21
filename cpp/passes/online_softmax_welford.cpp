#include "passes/online_softmax_welford.hpp"

#include <algorithm>
#include <variant>

namespace m1triton::passes {

namespace {

constexpr uint32_t kPreferredSimdgroupsPerThreadgroup = 8;
constexpr uint32_t kRmsTileSpatial = 64;
constexpr uint32_t kRmsTileChannels = 64;

uint32_t select_tg_threads(const ir::TargetSpec& target) {
  const uint32_t simd = std::max<uint32_t>(target.simd_width, 1);
  uint32_t threads = simd * kPreferredSimdgroupsPerThreadgroup;
  threads = std::min(threads, target.max_threads_per_threadgroup);
  threads = std::max(simd, (threads / simd) * simd);
  return threads;
}

uint32_t select_vec_width(const ir::TargetSpec& target) {
  const uint32_t lanes = target.vector_bytes / 4u;
  return lanes >= 4u ? 4u : std::max<uint32_t>(lanes, 1u);
}

bool rms_tile_fits(const ir::TargetSpec& target, uint32_t tg_threads) {
  const uint32_t tile_bytes = kRmsTileSpatial * kRmsTileChannels * 4u;
  const uint32_t reduce_bytes = tg_threads * 4u + kRmsTileSpatial * 4u;
  return tg_threads % kRmsTileSpatial == 0 && target.fits_threadgroup_memory(tile_bytes + reduce_bytes);
}

}  // namespace

PassResult OnlineSoftmaxWelfordPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  const uint32_t tg_threads = select_tg_threads(ctx.target);
  const uint32_t vec_width = select_vec_width(ctx.target);

  for (auto& op_ptr : kernel.ops) {
    ir::Op& op = *op_ptr;
    if (op.erased) continue;
    if (op.kind == ir::OpKind::OnlineSoftmax) {
      auto* attr = std::get_if<ir::OnlineSoftmaxAttr>(&op.attr);
      if (attr == nullptr) {
        res.ok = false;
        res.diagnostic = "online_softmax op is missing OnlineSoftmaxAttr";
        return res;
      }
      ir::OnlineSoftmaxAttr next = *attr;
      next.tg_threads = tg_threads;
      next.vec_width = vec_width;
      if (!(next == *attr)) {
        *attr = next;
        res.changed = true;
      }
    } else if (op.kind == ir::OpKind::WelfordReduce) {
      auto* attr = std::get_if<ir::WelfordNormAttr>(&op.attr);
      if (attr == nullptr) {
        res.ok = false;
        res.diagnostic = "welford_reduce op is missing WelfordNormAttr";
        return res;
      }
      ir::WelfordNormAttr next = *attr;
      next.tg_threads = tg_threads;
      next.vec_width = vec_width;
      next.rms_sram_cache = next.kind == ir::NormKind::RMSNorm && rms_tile_fits(ctx.target, tg_threads);
      if (!(next == *attr)) {
        *attr = next;
        res.changed = true;
      }
    }
  }
  return res;
}

}  // namespace m1triton::passes
