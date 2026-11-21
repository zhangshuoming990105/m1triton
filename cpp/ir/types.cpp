#include "ir/types.hpp"

#include <sstream>

namespace m1triton::ir {

namespace {

std::string_view layout_kind_name(LayoutKind kind) {
  switch (kind) {
    case LayoutKind::Unassigned:
      return "unassigned";
    case LayoutKind::Blocked:
      return "blocked";
    case LayoutKind::ThreadgroupStaged:
      return "tg_staged";
    case LayoutKind::SimdgroupFragment:
      return "simd_frag";
    case LayoutKind::Replicated:
      return "replicated";
  }
  return "unassigned";
}

std::string_view swizzle_mode_name(SwizzleMode mode) {
  switch (mode) {
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

std::string_view frag_role_name(FragRole role) {
  switch (role) {
    case FragRole::MatrixA:
      return "a";
    case FragRole::MatrixB:
      return "b";
    case FragRole::Accumulator:
      return "acc";
  }
  return "acc";
}

void format_elem_type(std::ostringstream& os, const ElemType& elem) {
  if (std::holds_alternative<ScalarDType>(elem)) {
    os << dtype_name(std::get<ScalarDType>(elem));
  } else {
    const auto& ptr = std::get<PtrType>(elem);
    os << "ptr<" << dtype_name(ptr.pointee) << ", " << addr_space_name(ptr.space) << ">";
  }
}

void format_layout_attr(std::ostringstream& os, const LayoutAttr& layout) {
  os << layout_kind_name(layout.kind) << "(";
  os << "vec=" << layout.vec_width;
  os << ", align=" << layout.alignment_bytes;
  os << ", contig=" << layout.contiguity_elems;
  os << ", tg=" << layout.tg_threads[0] << "x" << layout.tg_threads[1] << "x" << layout.tg_threads[2];
  os << ", ept=" << layout.elems_per_thread[0] << "x" << layout.elems_per_thread[1];
  if (layout.swizzle != SwizzleMode::None) {
    os << ", swizzle=" << swizzle_mode_name(layout.swizzle);
  }
  if (layout.pad_elems != 0) {
    os << ", pad=" << layout.pad_elems;
  }
  if (layout.stages != 1) {
    os << ", stages=" << layout.stages;
  }
  if (layout.frag_role != FragRole::Accumulator) {
    os << ", role=" << frag_role_name(layout.frag_role);
  }
  if (layout.frag_tiles_m != 1 || layout.frag_tiles_n != 1) {
    os << ", ftiles=" << layout.frag_tiles_m << "x" << layout.frag_tiles_n;
  }
  if (layout.simd_grid[0] != 1 || layout.simd_grid[1] != 1) {
    os << ", sgrid=" << layout.simd_grid[0] << "x" << layout.simd_grid[1];
  }
  os << ")";
}

void format_structure_attr(std::ostringstream& os, const StructureAttr& s) {
  os << structure_name(s.kind);
  if (s.diagonal_offset != 0 || s.is_unit_diagonal) {
    os << "(off=" << s.diagonal_offset << ", unit=" << (s.is_unit_diagonal ? "true" : "false") << ")";
  }
}

}  // namespace

ElemType Type::elem_type() const {
  if (is_scalar()) {
    return ElemType{as_scalar()};
  }
  if (is_ptr()) {
    return ElemType{as_ptr()};
  }
  return as_tile().elem;
}

bool Type::is_float_elem() const {
  const ElemType elem = elem_type();
  if (!std::holds_alternative<ScalarDType>(elem)) {
    return false;
  }
  const ScalarDType dt = std::get<ScalarDType>(elem);
  return dt == ScalarDType::F32 || dt == ScalarDType::F16;
}

bool Type::is_int_elem() const {
  const ElemType elem = elem_type();
  if (!std::holds_alternative<ScalarDType>(elem)) {
    return false;
  }
  const ScalarDType dt = std::get<ScalarDType>(elem);
  return dt == ScalarDType::I32 || dt == ScalarDType::U32 || dt == ScalarDType::I64;
}

bool Type::is_bool_elem() const {
  const ElemType elem = elem_type();
  return std::holds_alternative<ScalarDType>(elem) && std::get<ScalarDType>(elem) == ScalarDType::Bool;
}

bool Type::is_ptr_elem() const {
  return std::holds_alternative<PtrType>(elem_type());
}

ScalarDType Type::scalar_elem() const {
  return std::get<ScalarDType>(elem_type());
}

PtrType Type::ptr_elem() const {
  return std::get<PtrType>(elem_type());
}

uint32_t Type::elem_bytes() const {
  const ElemType elem = elem_type();
  if (std::holds_alternative<ScalarDType>(elem)) {
    return dtype_bytes(std::get<ScalarDType>(elem));
  }
  return 8;
}

bool Type::same_shape_and_elem(const Type& other) const {
  if (is_scalar() && other.is_scalar()) {
    return as_scalar() == other.as_scalar();
  }
  if (is_ptr() && other.is_ptr()) {
    return as_ptr() == other.as_ptr();
  }
  if (is_tile() && other.is_tile()) {
    return as_tile().shape == other.as_tile().shape && as_tile().elem == other.as_tile().elem;
  }
  return false;
}

std::string_view dtype_name(ScalarDType dt) {
  switch (dt) {
    case ScalarDType::F32:
      return "f32";
    case ScalarDType::F16:
      return "f16";
    case ScalarDType::I32:
      return "i32";
    case ScalarDType::U32:
      return "u32";
    case ScalarDType::I64:
      return "i64";
    case ScalarDType::Bool:
      return "bool";
  }
  return "f32";
}

std::optional<ScalarDType> parse_dtype(std::string_view s) {
  if (s == "f32" || s == "float32") return ScalarDType::F32;
  if (s == "f16" || s == "float16") return ScalarDType::F16;
  if (s == "i32" || s == "int32") return ScalarDType::I32;
  if (s == "u32" || s == "uint32") return ScalarDType::U32;
  if (s == "i64" || s == "int64") return ScalarDType::I64;
  if (s == "bool" || s == "i1") return ScalarDType::Bool;
  return std::nullopt;
}

uint32_t dtype_bytes(ScalarDType dt) {
  switch (dt) {
    case ScalarDType::F32:
      return 4;
    case ScalarDType::F16:
      return 2;
    case ScalarDType::I32:
      return 4;
    case ScalarDType::U32:
      return 4;
    case ScalarDType::I64:
      return 8;
    case ScalarDType::Bool:
      return 1;
  }
  return 4;
}

std::string_view addr_space_name(AddrSpace space) {
  switch (space) {
    case AddrSpace::Device:
      return "device";
    case AddrSpace::Constant:
      return "constant";
    case AddrSpace::Threadgroup:
      return "threadgroup";
  }
  return "device";
}

std::string_view structure_name(MatrixStructure kind) {
  switch (kind) {
    case MatrixStructure::Dense:
      return "dense";
    case MatrixStructure::UpperTriangular:
      return "upper_triangular";
    case MatrixStructure::LowerTriangular:
      return "lower_triangular";
    case MatrixStructure::Symmetric:
      return "symmetric";
    case MatrixStructure::Diagonal:
      return "diagonal";
    case MatrixStructure::CausalMask:
      return "causal_mask";
  }
  return "dense";
}

std::string type_to_string(const Type& ty) {
  std::ostringstream os;
  if (ty.is_scalar()) {
    os << dtype_name(ty.as_scalar());
  } else if (ty.is_ptr()) {
    const auto& ptr = ty.as_ptr();
    os << "ptr<" << dtype_name(ptr.pointee) << ", " << addr_space_name(ptr.space) << ">";
  } else {
    const auto& tile = ty.as_tile();
    os << "tile<";
    if (tile.shape.empty()) {
      os << "0d";
    } else {
      for (size_t i = 0; i < tile.shape.size(); ++i) {
        if (i > 0) os << "x";
        os << tile.shape[i];
      }
    }
    os << ", ";
    format_elem_type(os, tile.elem);
    if (tile.layout != LayoutAttr{}) {
      os << ", layout=";
      format_layout_attr(os, tile.layout);
    }
    if (tile.structure != StructureAttr{}) {
      os << ", struct=";
      format_structure_attr(os, tile.structure);
    }
    os << ">";
  }
  return os.str();
}

}  // namespace m1triton::ir
