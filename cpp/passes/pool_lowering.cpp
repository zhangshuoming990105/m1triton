#include "passes/pool_lowering.hpp"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace m1triton::passes {

namespace {

bool normalize_vec(std::vector<int32_t>& v, uint32_t ndim, int32_t fill, const std::vector<int32_t>* fallback) {
  if (v.size() == ndim) return false;
  if (v.empty() && fallback != nullptr && fallback->size() == ndim) {
    v = *fallback;
  } else if (v.size() == 1) {
    const int32_t scalar = v[0];
    v.assign(ndim, scalar);
  } else {
    v.assign(ndim, fill);
  }
  return true;
}

int64_t halo_extent(int64_t tile, int32_t stride, int32_t dilation, int32_t kernel) {
  return (tile - 1) * stride + static_cast<int64_t>(dilation) * (kernel - 1) + 1;
}

std::string validate(const ir::PoolAttr& pa) {
  for (uint32_t i = 0; i < pa.ndim; ++i) {
    if (pa.kernel_size[i] <= 0) return "kernel_size must be positive";
    if (pa.stride[i] <= 0) return "stride must be positive";
    if (pa.dilation[i] <= 0) return "dilation must be positive";
    if (pa.padding[i] < 0) return "padding must be non-negative";
    if (pa.padding[i] * 2 > pa.kernel_size[i]) return "padding must be at most half of the kernel size";
  }
  return {};
}

}  // namespace

PassResult PoolLoweringPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  const uint32_t smem_budget = ctx.target.max_threadgroup_memory_bytes / 2;

  for (auto& op_ptr : kernel.ops) {
    ir::Op& op = *op_ptr;
    if (op.erased || op.kind != ir::OpKind::Pool) continue;
    auto* pa = std::get_if<ir::PoolAttr>(&op.attr);
    if (pa == nullptr || pa->ndim < 1 || pa->ndim > 3) {
      res.ok = false;
      res.diagnostic = "pool op requires PoolAttr with ndim in [1, 3]";
      return res;
    }
    const ir::PoolAttr before = *pa;
    const uint32_t nd = pa->ndim;

    normalize_vec(pa->kernel_size, nd, 1, nullptr);
    normalize_vec(pa->stride, nd, 1, &pa->kernel_size);
    normalize_vec(pa->padding, nd, 0, nullptr);
    normalize_vec(pa->dilation, nd, 1, nullptr);
    if (pa->kind == ir::PoolKind::Avg) pa->dilation.assign(nd, 1);

    if (const std::string err = validate(*pa); !err.empty()) {
      res.ok = false;
      res.diagnostic = "pool op: " + err;
      return res;
    }

    const auto& k = pa->kernel_size;
    const auto& s = pa->stride;
    const auto& d = pa->dilation;
    if (nd == 1) {
      pa->tg_x = 256;
      pa->tg_y = 1;
      const int64_t tile_bytes = halo_extent(pa->tg_x, s[0], d[0], k[0]) * 4;
      pa->use_sram_halo = s[0] == 1 && tile_bytes <= smem_budget;
    } else if (nd == 2) {
      pa->tg_x = 16;
      pa->tg_y = 16;
      const int64_t tile_bytes =
          halo_extent(pa->tg_y, s[0], d[0], k[0]) * halo_extent(pa->tg_x, s[1], d[1], k[1]) * 4;
      pa->use_sram_halo = s[0] == 1 && s[1] == 1 && tile_bytes <= smem_budget;
    } else {
      pa->tg_x = 16;
      pa->tg_y = 16;
      pa->use_sram_halo = false;
    }

    if (!(*pa == before)) res.changed = true;
  }
  return res;
}

}  // namespace m1triton::passes
