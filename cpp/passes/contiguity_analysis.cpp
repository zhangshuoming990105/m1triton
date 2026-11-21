#include "passes/contiguity_analysis.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

constexpr uint32_t kMaxAlignment = 1u << 20;

uint32_t pow2_divisibility(int64_t v) {
  if (v == 0) return kMaxAlignment;
  const uint64_t u = (v < 0) ? (0ULL - static_cast<uint64_t>(v)) : static_cast<uint64_t>(v);
  const uint64_t lowbit = u & (0ULL - u);
  return static_cast<uint32_t>(std::min<uint64_t>(lowbit, kMaxAlignment));
}

uint32_t combine_align_add(uint32_t a, uint32_t b) {
  if (a == 0) return b;
  if (b == 0) return a;
  return std::gcd(a, b);
}

uint32_t combine_align_mul(uint32_t a, uint32_t b) {
  const uint64_t p = static_cast<uint64_t>(std::max<uint32_t>(1, a)) * std::max<uint32_t>(1, b);
  return static_cast<uint32_t>(std::min<uint64_t>(p, kMaxAlignment));
}

IndexFact meet_facts(const IndexFact& a, const IndexFact& b) {
  IndexFact out;
  out.uniform = a.uniform && b.uniform;
  out.step = (a.step == b.step) ? a.step : -1;
  out.contiguity = std::min(a.contiguity, b.contiguity);
  out.alignment = combine_align_add(a.alignment, b.alignment);
  out.base_ptr_divisibility = combine_align_add(a.base_ptr_divisibility, b.base_ptr_divisibility);
  return out;
}

IndexFact combine_add(const IndexFact& fa, const IndexFact& fb) {
  IndexFact out;
  out.alignment = combine_align_add(fa.alignment, fb.alignment);
  out.base_ptr_divisibility = std::max(fa.base_ptr_divisibility, fb.base_ptr_divisibility);
  if (fa.uniform && fb.uniform) {
    out.uniform = true;
    out.step = 0;
    out.contiguity = 1;
  } else if (fa.uniform && fb.step >= 0) {
    out.uniform = false;
    out.step = fb.step;
    out.contiguity = fb.contiguity;
  } else if (fb.uniform && fa.step >= 0) {
    out.uniform = false;
    out.step = fa.step;
    out.contiguity = fa.contiguity;
  } else if (fa.step >= 0 && fb.step >= 0) {
    out.uniform = false;
    out.step = fa.step + fb.step;
    out.contiguity = std::min(fa.contiguity, fb.contiguity);
  } else {
    out.uniform = false;
    out.step = -1;
    out.contiguity = 1;
  }
  return out;
}

class Analyzer {
 public:
  explicit Analyzer(const Kernel& k) : kernel_(k) {}

  std::unordered_map<ValueId, IndexFact> run() {
    if (kernel_.entry_block == kInvalidId) return facts_;
    const Block& entry = kernel_.block(kernel_.entry_block);
    for (size_t i = 0; i < entry.args.size() && i < kernel_.params.size(); ++i) {
      const ValueId vid = entry.args[i];
      const KernelParam& p = kernel_.params[i];
      uint32_t div = std::max<uint32_t>(1, p.divisibility);
      if (p.is_constexpr) {
        const_ints_[vid] = p.constexpr_val;
        div = std::max(div, pow2_divisibility(p.constexpr_val));
      }
      IndexFact f;
      f.step = 0;
      f.contiguity = 1;
      if (p.type.is_ptr()) {
        const uint32_t eb = std::max<uint32_t>(1, dtype_bytes(p.type.as_ptr().pointee));
        const uint32_t elem_div = (div >= eb && (div % eb) == 0) ? (div / eb) : 1;
        f.alignment = elem_div;
        f.base_ptr_divisibility = elem_div;
      } else {
        f.alignment = div;
        f.base_ptr_divisibility = 1;
      }
      f.uniform = true;
      facts_[vid] = f;
    }
    analyze_block(kernel_.entry_block);
    return facts_;
  }

 private:
  IndexFact get_fact(ValueId vid) const {
    auto it = facts_.find(vid);
    if (it != facts_.end()) return it->second;
    const Type& ty = kernel_.value(vid).type;
    IndexFact unknown;
    unknown.uniform = !ty.is_tile();
    unknown.step = unknown.uniform ? 0 : -1;
    unknown.contiguity = 1;
    unknown.alignment = 1;
    unknown.base_ptr_divisibility = 1;
    return unknown;
  }

  void analyze_block(BlockId blk_id) {
    const Block& blk = kernel_.block(blk_id);
    for (OpId op_id : blk.ops) {
      const Op& node = kernel_.op(op_id);
      if (node.erased) continue;
      analyze_op(node);
    }
  }

  void analyze_op(const Op& node) {
    switch (node.kind) {
      case OpKind::Constant: {
        const ValueId rid = node.results[0];
        const auto& ca = std::get<ConstAttr>(node.attr);
        IndexFact f;
        f.step = 0;
        f.contiguity = 1;
        f.uniform = true;
        if (const auto* i = std::get_if<int64_t>(&ca.val)) {
          f.alignment = pow2_divisibility(*i);
          const_ints_[rid] = *i;
        } else if (const auto* u = std::get_if<uint64_t>(&ca.val)) {
          f.alignment = pow2_divisibility(static_cast<int64_t>(*u));
          const_ints_[rid] = static_cast<int64_t>(*u);
        } else {
          f.alignment = 1;
        }
        facts_[rid] = f;
        break;
      }

      case OpKind::ProgramId:
      case OpKind::NumPrograms: {
        IndexFact f;
        f.step = 0;
        f.contiguity = 1;
        f.alignment = 1;
        f.uniform = true;
        facts_[node.results[0]] = f;
        break;
      }

      case OpKind::Arange: {
        const auto& ar = std::get<ArangeAttr>(node.attr);
        const uint32_t len = static_cast<uint32_t>(std::max<int64_t>(1, ar.end - ar.start));
        IndexFact f;
        f.step = 1;
        f.contiguity = len;
        f.alignment = pow2_divisibility(ar.start);
        f.uniform = false;
        facts_[node.results[0]] = f;
        break;
      }

      case OpKind::Splat: {
        const ValueId src = node.operands[0];
        const ValueId rid = node.results[0];
        IndexFact f = get_fact(src);
        f.step = 0;
        f.contiguity = 1;
        f.uniform = true;
        facts_[rid] = f;
        if (auto it = const_ints_.find(src); it != const_ints_.end()) {
          const_ints_[rid] = it->second;
        }
        break;
      }

      case OpKind::ExpandDims: {
        const ValueId src = node.operands[0];
        const ValueId rid = node.results[0];
        const Type& sty = kernel_.value(src).type;
        const uint32_t axis = std::get<AxisAttr>(node.attr).axis;
        IndexFact sf = get_fact(src);
        if (!sty.is_tile()) {
          sf.step = 0;
          sf.contiguity = 1;
          sf.uniform = true;
          facts_[rid] = sf;
        } else if (axis == sty.as_tile().shape.size()) {
          IndexFact f;
          f.step = 0;
          f.contiguity = 1;
          f.uniform = true;
          f.base_ptr_divisibility = sf.base_ptr_divisibility;
          if (sf.uniform) {
            f.alignment = sf.alignment;
          } else if (sf.step > 0) {
            f.alignment = combine_align_add(sf.alignment, pow2_divisibility(sf.step));
          } else {
            f.alignment = 1;
          }
          facts_[rid] = f;
        } else {
          facts_[rid] = sf;
        }
        if (auto it = const_ints_.find(src); it != const_ints_.end()) {
          const_ints_[rid] = it->second;
        }
        break;
      }

      case OpKind::Broadcast: {
        const ValueId src = node.operands[0];
        const ValueId rid = node.results[0];
        const auto& sshape = kernel_.value(src).type.as_tile().shape;
        const auto& dshape = kernel_.value(rid).type.as_tile().shape;
        IndexFact sf = get_fact(src);
        if (sshape.back() == 1 && dshape.back() > 1) {
          sf.step = 0;
          sf.contiguity = 1;
          sf.uniform = true;
        }
        facts_[rid] = sf;
        if (auto it = const_ints_.find(src); it != const_ints_.end()) {
          const_ints_[rid] = it->second;
        }
        break;
      }

      case OpKind::Cast: {
        const ValueId src = node.operands[0];
        const ValueId rid = node.results[0];
        if (kernel_.value(src).type.is_int_elem() && kernel_.value(rid).type.is_int_elem()) {
          facts_[rid] = get_fact(src);
          if (auto it = const_ints_.find(src); it != const_ints_.end()) {
            const_ints_[rid] = it->second;
          }
        }
        break;
      }

      case OpKind::Binary: {
        const BinaryOp bop = std::get<BinaryAttr>(node.attr).op;
        const ValueId lhs = node.operands[0];
        const ValueId rhs = node.operands[1];
        const ValueId rid = node.results[0];
        const IndexFact fa = get_fact(lhs);
        const IndexFact fb = get_fact(rhs);

        if (bop == BinaryOp::Add) {
          facts_[rid] = combine_add(fa, fb);
          if (const_ints_.contains(lhs) && const_ints_.contains(rhs)) {
            const_ints_[rid] = const_ints_[lhs] + const_ints_[rhs];
          }
        } else if (bop == BinaryOp::Sub) {
          IndexFact f;
          f.alignment = combine_align_add(fa.alignment, fb.alignment);
          if (fa.uniform && fb.uniform) {
            f.uniform = true;
            f.step = 0;
            f.contiguity = 1;
          } else if (fb.uniform && fa.step >= 0) {
            f.uniform = false;
            f.step = fa.step;
            f.contiguity = fa.contiguity;
          } else {
            f.uniform = false;
            f.step = -1;
            f.contiguity = 1;
          }
          facts_[rid] = f;
          if (const_ints_.contains(lhs) && const_ints_.contains(rhs)) {
            const_ints_[rid] = const_ints_[lhs] - const_ints_[rhs];
          }
        } else if (bop == BinaryOp::Mul) {
          IndexFact f;
          f.alignment = combine_align_mul(fa.alignment, fb.alignment);
          if (fa.uniform && fb.uniform) {
            f.uniform = true;
            f.step = 0;
            f.contiguity = 1;
          } else if (fa.uniform && fb.step >= 0 && const_ints_.contains(lhs) && const_ints_[lhs] >= 0) {
            const int64_t c = const_ints_[lhs];
            f.uniform = (c == 0);
            f.step = fb.step * c;
            f.contiguity = (c == 0) ? 1 : fb.contiguity;
          } else if (fb.uniform && fa.step >= 0 && const_ints_.contains(rhs) && const_ints_[rhs] >= 0) {
            const int64_t c = const_ints_[rhs];
            f.uniform = (c == 0);
            f.step = fa.step * c;
            f.contiguity = (c == 0) ? 1 : fa.contiguity;
          } else {
            f.uniform = false;
            f.step = -1;
            f.contiguity = 1;
          }
          facts_[rid] = f;
          if (const_ints_.contains(lhs) && const_ints_.contains(rhs)) {
            const_ints_[rid] = const_ints_[lhs] * const_ints_[rhs];
          }
        }
        break;
      }

      case OpKind::PtrAdd: {
        const ValueId ptr = node.operands[0];
        const ValueId off = node.operands[1];
        const ValueId rid = node.results[0];
        const IndexFact fp = get_fact(ptr);
        const IndexFact fo = get_fact(off);
        IndexFact f = combine_add(fp, fo);
        f.base_ptr_divisibility = fp.base_ptr_divisibility;
        facts_[rid] = f;
        break;
      }

      case OpKind::ForLoop: {
        const BlockId body_id = node.regions[0];
        const Block& body = kernel_.block(body_id);
        const IndexFact lo_f = get_fact(node.operands[0]);
        const IndexFact step_f = get_fact(node.operands[2]);

        IndexFact iv_fact;
        iv_fact.step = 0;
        iv_fact.contiguity = 1;
        iv_fact.uniform = true;
        iv_fact.alignment = combine_align_add(lo_f.alignment, step_f.alignment);
        facts_[body.args[0]] = iv_fact;

        const size_t num_iter = node.operands.size() - 3;
        for (size_t i = 0; i < num_iter; ++i) {
          facts_[body.args[1 + i]] = get_fact(node.operands[3 + i]);
        }

        constexpr int kMaxLoopIters = 6;
        for (int iter = 0; iter < kMaxLoopIters; ++iter) {
          analyze_block(body_id);
          bool updated = false;
          if (!body.ops.empty()) {
            const Op& term = kernel_.op(body.ops.back());
            if (term.kind == OpKind::Yield && term.operands.size() == num_iter) {
              for (size_t i = 0; i < num_iter; ++i) {
                const IndexFact cur = get_fact(body.args[1 + i]);
                const IndexFact next = meet_facts(cur, get_fact(term.operands[i]));
                if (next != cur) {
                  facts_[body.args[1 + i]] = next;
                  updated = true;
                }
              }
            }
          }
          if (!updated) break;
        }

        for (size_t i = 0; i < num_iter && i < node.results.size(); ++i) {
          facts_[node.results[i]] = get_fact(body.args[1 + i]);
        }
        break;
      }

      case OpKind::If: {
        for (BlockId reg_id : node.regions) {
          analyze_block(reg_id);
        }
        if (node.regions.size() == 2 && !node.results.empty()) {
          const Block& tb = kernel_.block(node.regions[0]);
          const Block& eb = kernel_.block(node.regions[1]);
          if (!tb.ops.empty() && !eb.ops.empty()) {
            const Op& ty = kernel_.op(tb.ops.back());
            const Op& ey = kernel_.op(eb.ops.back());
            if (ty.kind == OpKind::Yield && ey.kind == OpKind::Yield && ty.operands.size() == node.results.size() &&
                ey.operands.size() == node.results.size()) {
              for (size_t i = 0; i < node.results.size(); ++i) {
                facts_[node.results[i]] = meet_facts(get_fact(ty.operands[i]), get_fact(ey.operands[i]));
              }
            }
          }
        }
        break;
      }

      default:
        break;
    }
  }

  const Kernel& kernel_;
  std::unordered_map<ValueId, IndexFact> facts_;
  std::unordered_map<ValueId, int64_t> const_ints_;
};

bool upgrade_block_loads_stores(Kernel& k, BlockId blk_id, const std::unordered_map<ValueId, IndexFact>& facts,
                                const TargetSpec& target) {
  bool changed = false;
  const auto ops = k.block(blk_id).ops;
  for (OpId op_id : ops) {
    Op& node = k.op(op_id);
    if (node.erased) continue;

    for (BlockId reg_id : node.regions) {
      if (upgrade_block_loads_stores(k, reg_id, facts, target)) {
        changed = true;
      }
    }

    if (node.kind != OpKind::Load && node.kind != OpKind::Store) {
      continue;
    }

    const ValueId ptrs = node.operands[0];
    const Type& pty = k.value(ptrs).type;
    if (!pty.is_tile() || !pty.is_ptr_elem()) {
      continue;
    }
    const int64_t fastest_dim = pty.as_tile().shape.back();
    const ScalarDType pointee = pty.ptr_elem().pointee;
    const uint32_t elem_bytes = dtype_bytes(pointee);
    if (elem_bytes * 4 > target.vector_bytes) {
      continue;
    }

    auto it = facts.find(ptrs);
    if (it == facts.end()) {
      continue;
    }
    const IndexFact& f = it->second;

    uint32_t target_vec = 4;
    if (node.hints.vec_width.has_value()) {
      target_vec = std::min(target_vec, node.hints.vec_width->width);
    }
    if (target_vec < 4) {
      continue;
    }

    const bool can_vec4 = (f.step == 1) && (f.contiguity >= 4) && (fastest_dim >= 4) && (fastest_dim % 4 == 0) &&
                          (f.base_ptr_divisibility % 4 == 0) && (f.alignment % 4 == 0);
    if (!can_vec4) {
      continue;
    }

    auto& ls = std::get<LoadStoreAttr>(node.attr);
    const uint32_t new_align = 4 * elem_bytes;
    if (ls.vec_width != 4 || ls.alignment_bytes != new_align) {
      ls.vec_width = 4;
      ls.alignment_bytes = new_align;
      changed = true;
    }
  }
  return changed;
}

}  // namespace

std::unordered_map<ir::ValueId, IndexFact> ContiguityAndAlignmentAnalysis::analyze(const ir::Kernel& kernel) {
  Analyzer analyzer(kernel);
  return analyzer.run();
}

PassResult ContiguityAndAlignmentAnalysis::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  if (kernel.entry_block == ir::kInvalidId) {
    return res;
  }
  const auto facts = analyze(kernel);
  res.changed = upgrade_block_loads_stores(kernel, kernel.entry_block, facts, ctx.target);
  return res;
}

}  // namespace m1triton::passes
