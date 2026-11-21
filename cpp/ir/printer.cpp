#include "ir/printer.hpp"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace m1triton::ir {

namespace {

bool is_valid_ident(std::string_view s) {
  if (s.empty()) return false;
  if (!(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
  for (char c : s) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
  }
  return true;
}

std::string format_double(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return std::signbit(v) ? "-inf" : "inf";
  std::ostringstream os;
  os << std::setprecision(17) << v;
  std::string s = os.str();
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
    s += ".0";
  }
  return s;
}

std::string_view reduce_kind_name(ReduceKind k) {
  switch (k) {
    case ReduceKind::Sum:
      return "sum";
    case ReduceKind::Max:
      return "max";
    case ReduceKind::Min:
      return "min";
    case ReduceKind::Argmax:
      return "argmax";
    case ReduceKind::Argmin:
      return "argmin";
  }
  return "sum";
}

std::string_view reduce_mode_name(ReduceResultMode m) {
  switch (m) {
    case ReduceResultMode::ValueOnly:
      return "value";
    case ReduceResultMode::IndexOnly:
      return "index";
    case ReduceResultMode::ValueAndIndex:
      return "value_and_index";
  }
  return "value";
}

std::string_view scan_kind_name(ScanKind k) {
  switch (k) {
    case ScanKind::Cumsum:
      return "cumsum";
    case ScanKind::Cumprod:
      return "cumprod";
  }
  return "cumsum";
}

std::string_view barrier_scope_name(BarrierScope s) {
  switch (s) {
    case BarrierScope::Simdgroup:
      return "simdgroup";
    case BarrierScope::Threadgroup:
      return "threadgroup";
    case BarrierScope::DeviceMem:
      return "device_mem";
  }
  return "threadgroup";
}

std::string_view tensor_layout_hint_name(TensorLayoutHint h) {
  switch (h) {
    case TensorLayoutHint::NCHW:
      return "nchw";
    case TensorLayoutHint::NHWC:
      return "nhwc";
    case TensorLayoutHint::NCDHW:
      return "ncdhw";
    case TensorLayoutHint::RowMajor:
      return "row_major";
    case TensorLayoutHint::ColMajor:
      return "col_major";
  }
  return "nchw";
}

std::string_view swizzle_name(SwizzleMode m) {
  switch (m) {
    case SwizzleMode::None:
      return "none";
    case SwizzleMode::PadBank4:
      return "pad4";
    case SwizzleMode::PadBank8:
      return "pad8";
    case SwizzleMode::Xor8:
      return "xor8";
  }
  return "none";
}

void print_structure_inline(std::ostream& os, const StructureAttr& s) {
  os << structure_name(s.kind);
  if (s.diagonal_offset != 0 || s.is_unit_diagonal) {
    os << "(off=" << s.diagonal_offset << ", unit=" << (s.is_unit_diagonal ? "true" : "false") << ")";
  }
}

bool has_any_hints(const OpHints& h) {
  return h.tile_spec.has_value() || h.structure.has_value() || h.vec_width.has_value() || h.layout_hint.has_value() ||
         h.persistent_grid.has_value();
}

void print_hints(std::ostream& os, const OpHints& h) {
  if (!has_any_hints(h)) return;
  os << " hints {";
  bool first = true;
  auto sep = [&]() {
    if (!first) os << ", ";
    first = false;
  };
  if (h.tile_spec) {
    sep();
    const auto& t = *h.tile_spec;
    os << "tile_spec(bm=" << t.bm << ", bn=" << t.bn << ", bk=" << t.bk << ", simds=" << t.num_simdgroups
       << ", dbl=" << (t.double_buffer ? "true" : "false") << ", split_k=" << t.split_k << ")";
  }
  if (h.structure) {
    sep();
    os << "structure=";
    print_structure_inline(os, *h.structure);
  }
  if (h.vec_width) {
    sep();
    os << "vec_width=" << h.vec_width->width;
  }
  if (h.layout_hint) {
    sep();
    os << "layout_hint(layout=" << tensor_layout_hint_name(h.layout_hint->layout)
       << ", prefer_vec4=" << (h.layout_hint->prefer_vec4 ? "true" : "false") << ")";
  }
  if (h.persistent_grid) {
    sep();
    os << "persistent_grid(enabled=" << (h.persistent_grid->enabled ? "true" : "false")
       << ", max_tgs=" << h.persistent_grid->max_tgs_per_core << ", swizzle_w=" << h.persistent_grid->swizzle_w << ")";
  }
  os << "}";
}

class KernelPrinter {
 public:
  KernelPrinter(const Kernel& kernel, std::ostream& os) : kernel_(kernel), os_(os) { build_value_names(); }

  void print() {
    os_ << "kernel @" << kernel_.name << "(";
    const auto& entry_args =
        (kernel_.entry_block != kInvalidId && kernel_.entry_block < kernel_.blocks.size())
            ? kernel_.block(kernel_.entry_block).args
            : std::vector<ValueId>{};

    for (size_t i = 0; i < kernel_.params.size(); ++i) {
      if (i > 0) os_ << ", ";
      const auto& p = kernel_.params[i];
      const std::string pname =
          (i < entry_args.size()) ? val_name(entry_args[i]) : ("%" + (p.name.empty() ? ("arg" + std::to_string(i)) : p.name));
      os_ << pname << ": " << type_to_string(p.type);
      if (p.is_constexpr || p.divisibility != 1 || p.writable) {
        os_ << " {";
        bool first = true;
        auto sep = [&]() {
          if (!first) os_ << ", ";
          first = false;
        };
        if (p.is_constexpr) {
          sep();
          os_ << "constexpr=" << p.constexpr_val;
        }
        if (p.divisibility != 1) {
          sep();
          os_ << "divisibility=" << p.divisibility;
        }
        if (p.writable) {
          sep();
          os_ << "writable=true";
        }
        os_ << "}";
      }
    }
    os_ << ")";
    print_hints(os_, kernel_.hints);
    os_ << " {\n";
    if (kernel_.entry_block != kInvalidId && kernel_.entry_block < kernel_.blocks.size()) {
      print_block(kernel_.entry_block, 1);
    }
    os_ << "}\n";
  }

 private:
  void assign_block_names(BlockId block_id, uint32_t& next_id) {
    const Block& blk = kernel_.block(block_id);
    if (block_id != kernel_.entry_block) {
      for (ValueId arg : blk.args) {
        if (!names_.contains(arg)) {
          names_[arg] = "%" + std::to_string(next_id++);
        }
      }
    }
    for (OpId op_id : blk.ops) {
      const Op& node = kernel_.op(op_id);
      if (node.erased) continue;
      for (ValueId rid : node.results) {
        if (!names_.contains(rid)) {
          names_[rid] = "%" + std::to_string(next_id++);
        }
      }
      for (BlockId reg_id : node.regions) {
        assign_block_names(reg_id, next_id);
      }
    }
  }

  void build_value_names() {
    std::unordered_set<std::string> used;
    uint32_t next_id = 0;
    if (kernel_.entry_block != kInvalidId && kernel_.entry_block < kernel_.blocks.size()) {
      const auto& entry_args = kernel_.block(kernel_.entry_block).args;
      for (size_t i = 0; i < entry_args.size(); ++i) {
        std::string candidate;
        if (i < kernel_.params.size() && is_valid_ident(kernel_.params[i].name)) {
          candidate = "%" + kernel_.params[i].name;
        } else {
          candidate = "%arg" + std::to_string(i);
        }
        if (!used.insert(candidate).second) {
          candidate = "%" + std::to_string(next_id++);
          used.insert(candidate);
        }
        names_[entry_args[i]] = std::move(candidate);
      }
      assign_block_names(kernel_.entry_block, next_id);
    }
    for (const auto& v : kernel_.values) {
      if (!names_.contains(v->id)) {
        names_[v->id] = "%" + std::to_string(next_id++);
      }
    }
  }

  std::string val_name(ValueId id) const {
    auto it = names_.find(id);
    if (it != names_.end()) return it->second;
    return "%" + std::to_string(id);
  }

  void indent(int level) {
    for (int i = 0; i < level; ++i) os_ << "  ";
  }

  void print_result_types(const std::vector<ValueId>& results) {
    if (results.empty()) return;
    os_ << " : ";
    if (results.size() == 1) {
      os_ << type_to_string(kernel_.value(results[0]).type);
    } else {
      os_ << "(";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) os_ << ", ";
        os_ << type_to_string(kernel_.value(results[i]).type);
      }
      os_ << ")";
    }
  }

  void print_attr(const Op& node) {
    if (std::holds_alternative<EmptyAttr>(node.attr)) return;
    if (std::holds_alternative<UnaryAttr>(node.attr) || std::holds_alternative<BinaryAttr>(node.attr)) return;

    os_ << " {";
    if (const auto* c = std::get_if<ConstAttr>(&node.attr)) {
      os_ << "val=";
      if (std::holds_alternative<int64_t>(c->val)) {
        os_ << std::get<int64_t>(c->val);
      } else if (std::holds_alternative<uint64_t>(c->val)) {
        os_ << std::get<uint64_t>(c->val) << "u";
      } else if (std::holds_alternative<double>(c->val)) {
        os_ << format_double(std::get<double>(c->val));
      } else {
        os_ << (std::get<bool>(c->val) ? "true" : "false");
      }
    } else if (const auto* ax = std::get_if<AxisAttr>(&node.attr)) {
      os_ << "axis=" << ax->axis;
    } else if (const auto* ar = std::get_if<ArangeAttr>(&node.attr)) {
      os_ << "start=" << ar->start << ", end=" << ar->end;
    } else if (const auto* pm = std::get_if<PermuteAttr>(&node.attr)) {
      os_ << "perm=[";
      for (size_t i = 0; i < pm->perm.size(); ++i) {
        if (i > 0) os_ << ", ";
        os_ << pm->perm[i];
      }
      os_ << "]";
    } else if (const auto* ls = std::get_if<LoadStoreAttr>(&node.attr)) {
      os_ << "align=" << ls->alignment_bytes << ", vec=" << ls->vec_width;
      if (ls->has_mask) os_ << ", has_mask=true";
      if (ls->has_other) os_ << ", has_other=true";
      if (ls->is_volatile) os_ << ", volatile=true";
      if (ls->non_temporal) os_ << ", nt=true";
    } else if (const auto* da = std::get_if<DotAttr>(&node.attr)) {
      os_ << "acc=" << dtype_name(da->acc_dtype);
      if (da->allow_tf32) os_ << ", tf32=true";
      if (da->transpose_a) os_ << ", ta=true";
      if (da->transpose_b) os_ << ", tb=true";
      if (da->lhs_struct != StructureAttr{}) {
        os_ << ", lhs_struct=";
        print_structure_inline(os_, da->lhs_struct);
      }
      if (da->rhs_struct != StructureAttr{}) {
        os_ << ", rhs_struct=";
        print_structure_inline(os_, da->rhs_struct);
      }
      if (da->out_struct != StructureAttr{}) {
        os_ << ", out_struct=";
        print_structure_inline(os_, da->out_struct);
      }
    } else if (const auto* ra = std::get_if<ReduceAttr>(&node.attr)) {
      os_ << "kind=" << reduce_kind_name(ra->kind) << ", mode=" << reduce_mode_name(ra->result_mode)
          << ", axis=" << ra->axis;
      if (ra->keep_dims) os_ << ", keep_dims=true";
    } else if (const auto* sa = std::get_if<ScanAttr>(&node.attr)) {
      os_ << "kind=" << scan_kind_name(sa->kind) << ", axis=" << sa->axis;
      if (sa->exclusive) os_ << ", exclusive=true";
      if (sa->reverse) os_ << ", reverse=true";
      if (sa->has_mask) os_ << ", has_mask=true";
    } else if (const auto* ca = std::get_if<ConvAttr>(&node.attr)) {
      auto print_vec = [&](const char* k, const std::vector<int32_t>& v) {
        os_ << k << "=[";
        for (size_t i = 0; i < v.size(); ++i) {
          if (i > 0) os_ << ", ";
          os_ << v[i];
        }
        os_ << "]";
      };
      os_ << "ndim=" << ca->ndim << ", ";
      print_vec("stride", ca->stride);
      os_ << ", ";
      print_vec("padding", ca->padding);
      os_ << ", ";
      print_vec("dilation", ca->dilation);
      os_ << ", groups=" << ca->groups << ", transposed=" << (ca->transposed ? "true" : "false")
          << ", layout=" << tensor_layout_hint_name(ca->layout);
    } else if (const auto* fa = std::get_if<FlashAttnAttr>(&node.attr)) {
      os_ << "scale=" << format_double(fa->scale) << ", causal=" << (fa->causal ? "true" : "false")
          << ", causal_offset=" << fa->causal_offset;
    } else if (const auto* ta = std::get_if<ThreadgroupAllocAttr>(&node.attr)) {
      os_ << "elem=" << dtype_name(ta->elem) << ", shape=[";
      for (size_t i = 0; i < ta->shape.size(); ++i) {
        if (i > 0) os_ << ", ";
        os_ << ta->shape[i];
      }
      os_ << "], swizzle=" << swizzle_name(ta->swizzle) << ", pad=" << ta->pad_elems << ", stages=" << ta->stages;
    } else if (const auto* ts = std::get_if<ThreadgroupStageAttr>(&node.attr)) {
      os_ << "stage=" << ts->stage_idx << ", zero_pad=" << (ts->zero_pad_oob ? "true" : "false");
    } else if (const auto* ba = std::get_if<BarrierAttr>(&node.attr)) {
      os_ << "scope=" << barrier_scope_name(ba->scope);
    } else if (const auto* sm = std::get_if<SimdgroupMatmulAttr>(&node.attr)) {
      os_ << "m_tiles=" << sm->m_tiles << ", n_tiles=" << sm->n_tiles << ", k_step=" << sm->k_step
          << ", ldm_a=" << sm->ldm_a << ", ldm_b=" << sm->ldm_b << ", acc=" << dtype_name(sm->acc_dtype);
    }
    os_ << "}";
  }

  void print_block(BlockId block_id, int level) {
    const Block& blk = kernel_.block(block_id);
    for (OpId op_id : blk.ops) {
      const Op& node = kernel_.op(op_id);
      if (node.erased) continue;
      print_op(node, level);
    }
  }

  void print_op(const Op& node, int level) {
    indent(level);
    if (!node.results.empty()) {
      for (size_t i = 0; i < node.results.size(); ++i) {
        if (i > 0) os_ << ", ";
        os_ << val_name(node.results[i]);
      }
      os_ << " = ";
    }

    if (node.kind == OpKind::ForLoop) {
      os_ << "for_loop (";
      const Block& body = kernel_.block(node.regions[0]);
      if (!body.args.empty()) {
        os_ << val_name(body.args[0]) << ": " << type_to_string(kernel_.value(body.args[0]).type);
        for (size_t i = 1; i < body.args.size(); ++i) {
          os_ << ", " << val_name(body.args[i]) << ": " << type_to_string(kernel_.value(body.args[i]).type);
          if (2 + i < node.operands.size()) {
            os_ << " = " << val_name(node.operands[2 + i]);
          }
        }
      }
      os_ << ") in (" << val_name(node.operands[0]) << ", " << val_name(node.operands[1]) << ", "
          << val_name(node.operands[2]) << ")";
      print_hints(os_, node.hints);
      if (!node.results.empty()) {
        os_ << " : (";
        for (size_t i = 0; i < node.results.size(); ++i) {
          if (i > 0) os_ << ", ";
          os_ << type_to_string(kernel_.value(node.results[i]).type);
        }
        os_ << ")";
      }
      os_ << " {\n";
      print_block(node.regions[0], level + 1);
      indent(level);
      os_ << "}\n";
      return;
    }

    if (node.kind == OpKind::If) {
      os_ << "if " << val_name(node.operands[0]);
      print_hints(os_, node.hints);
      if (!node.results.empty()) {
        os_ << " : (";
        for (size_t i = 0; i < node.results.size(); ++i) {
          if (i > 0) os_ << ", ";
          os_ << type_to_string(kernel_.value(node.results[i]).type);
        }
        os_ << ")";
      }
      os_ << " {\n";
      print_block(node.regions[0], level + 1);
      indent(level);
      os_ << "}";
      if (node.regions.size() == 2) {
        os_ << " else {\n";
        print_block(node.regions[1], level + 1);
        indent(level);
        os_ << "}";
      }
      os_ << "\n";
      return;
    }

    os_ << op_kind_name(node.kind);
    if (node.kind == OpKind::Unary && std::holds_alternative<UnaryAttr>(node.attr)) {
      os_ << "." << unary_op_name(std::get<UnaryAttr>(node.attr).op);
    } else if (node.kind == OpKind::Binary && std::holds_alternative<BinaryAttr>(node.attr)) {
      os_ << "." << binary_op_name(std::get<BinaryAttr>(node.attr).op);
    }

    if (!node.operands.empty()) {
      os_ << " ";
      for (size_t i = 0; i < node.operands.size(); ++i) {
        if (i > 0) os_ << ", ";
        os_ << val_name(node.operands[i]);
      }
    }

    print_attr(node);
    print_hints(os_, node.hints);
    print_result_types(node.results);
    os_ << "\n";
  }

  const Kernel& kernel_;
  std::ostream& os_;
  std::unordered_map<ValueId, std::string> names_;
};

}  // namespace

void IRPrinter::print(const Kernel& kernel, std::ostream& os) {
  KernelPrinter printer(kernel, os);
  printer.print();
}

void IRPrinter::print(const Module& mod, std::ostream& os) {
  for (size_t i = 0; i < mod.kernels.size(); ++i) {
    if (i > 0) os << "\n";
    print(*mod.kernels[i], os);
  }
}

std::string IRPrinter::print(const Kernel& kernel) {
  std::ostringstream os;
  print(kernel, os);
  return os.str();
}

std::string IRPrinter::print(const Module& mod) {
  std::ostringstream os;
  print(mod, os);
  return os.str();
}

}  // namespace m1triton::ir
