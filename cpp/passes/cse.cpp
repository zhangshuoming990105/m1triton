#include "passes/cse.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

struct ExprKey {
  OpKind kind = OpKind::Constant;
  std::vector<ValueId> operands;
  std::vector<Type> result_types;
  OpPayload attr = EmptyAttr{};
  OpHints hints;
  bool has_const_double = false;
  uint64_t const_double_bits = 0;

  bool operator==(const ExprKey& o) const {
    if (kind != o.kind || operands != o.operands || result_types != o.result_types || hints != o.hints) {
      return false;
    }
    if (has_const_double || o.has_const_double) {
      return has_const_double == o.has_const_double && const_double_bits == o.const_double_bits;
    }
    return attr == o.attr;
  }
};

struct ExprKeyHash {
  size_t operator()(const ExprKey& k) const {
    size_t h = std::hash<uint16_t>{}(static_cast<uint16_t>(k.kind));
    auto combine = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (ValueId vid : k.operands) combine(std::hash<ValueId>{}(vid));
    for (const Type& t : k.result_types) combine(std::hash<std::string>{}(type_to_string(t)));
    combine(std::hash<size_t>{}(k.attr.index()));
    if (k.has_const_double) {
      combine(std::hash<uint64_t>{}(k.const_double_bits));
    } else if (const auto* c = std::get_if<ConstAttr>(&k.attr)) {
      if (const auto* i = std::get_if<int64_t>(&c->val)) combine(std::hash<int64_t>{}(*i));
      else if (const auto* u = std::get_if<uint64_t>(&c->val)) combine(std::hash<uint64_t>{}(*u));
      else if (const auto* b = std::get_if<bool>(&c->val)) combine(std::hash<bool>{}(*b));
    } else if (const auto* ax = std::get_if<AxisAttr>(&k.attr)) {
      combine(std::hash<uint32_t>{}(ax->axis));
    } else if (const auto* ar = std::get_if<ArangeAttr>(&k.attr)) {
      combine(std::hash<int64_t>{}(ar->start));
      combine(std::hash<int64_t>{}(ar->end));
    } else if (const auto* u = std::get_if<UnaryAttr>(&k.attr)) {
      combine(std::hash<uint8_t>{}(static_cast<uint8_t>(u->op)));
    } else if (const auto* b = std::get_if<BinaryAttr>(&k.attr)) {
      combine(std::hash<uint8_t>{}(static_cast<uint8_t>(b->op)));
    }
    return h;
  }
};

using ScopeTable = std::unordered_map<ExprKey, std::vector<ValueId>, ExprKeyHash>;

ExprKey build_key(const Kernel& k, const Op& node) {
  ExprKey key;
  key.kind = node.kind;
  key.operands = node.operands;
  if (node.kind == OpKind::Binary && key.operands.size() == 2 && std::holds_alternative<BinaryAttr>(node.attr)) {
    if (is_commutative_op(std::get<BinaryAttr>(node.attr).op) && key.operands[0] > key.operands[1]) {
      std::swap(key.operands[0], key.operands[1]);
    }
  }
  key.result_types.reserve(node.results.size());
  for (ValueId rid : node.results) {
    key.result_types.push_back(k.value(rid).type);
  }
  key.attr = node.attr;
  key.hints = node.hints;
  if (const auto* c = std::get_if<ConstAttr>(&node.attr)) {
    if (const auto* d = std::get_if<double>(&c->val)) {
      key.has_const_double = true;
      key.const_double_bits = std::bit_cast<uint64_t>(*d);
    }
  }
  return key;
}

const std::vector<ValueId>* lookup_scoped(const std::vector<ScopeTable>& scopes, const ExprKey& key) {
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
    auto found = it->find(key);
    if (found != it->end()) {
      return &found->second;
    }
  }
  return nullptr;
}

bool cse_block(Kernel& k, BlockId blk_id, std::vector<ScopeTable>& scopes) {
  bool changed = false;
  scopes.emplace_back();

  const auto op_snapshot = k.block(blk_id).ops;
  for (OpId op_id : op_snapshot) {
    if (k.op(op_id).erased) continue;
    const Op& node = k.op(op_id);

    if (!node.regions.empty()) {
      const auto regions = node.regions;
      for (BlockId reg_id : regions) {
        if (cse_block(k, reg_id, scopes)) {
          changed = true;
        }
      }
      continue;
    }

    // Never CSE memory, synchronization, control-flow, or side-effecting ops.
    if (!is_pure_op(node.kind) || node.results.empty()) {
      continue;
    }

    const ExprKey key = build_key(k, node);
    if (const auto* existing = lookup_scoped(scopes, key)) {
      const auto old_results = node.results;
      for (size_t i = 0; i < old_results.size(); ++i) {
        k.replace_all_uses(old_results[i], (*existing)[i]);
      }
      k.erase_op(op_id);
      changed = true;
    } else {
      scopes.back().emplace(key, node.results);
    }
  }

  scopes.pop_back();
  return changed;
}

}  // namespace

PassResult CSEPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  if (kernel.entry_block == ir::kInvalidId) {
    return res;
  }
  std::vector<ScopeTable> scopes;
  res.changed = cse_block(kernel, kernel.entry_block, scopes);
  return res;
}

}  // namespace m1triton::passes
