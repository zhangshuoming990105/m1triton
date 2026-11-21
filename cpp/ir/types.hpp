#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace m1triton::ir {

enum class ScalarDType : uint8_t { F32, F16, I32, U32, I64, Bool };

enum class AddrSpace : uint8_t { Device, Constant, Threadgroup };

struct PtrType {
  ScalarDType pointee = ScalarDType::F32;
  AddrSpace space = AddrSpace::Device;
  bool operator==(const PtrType&) const = default;
};

using ElemType = std::variant<ScalarDType, PtrType>;

enum class MatrixStructure : uint8_t {
  Dense,
  UpperTriangular,
  LowerTriangular,
  Symmetric,
  Diagonal,
  CausalMask,
};

struct StructureAttr {
  MatrixStructure kind = MatrixStructure::Dense;
  int32_t diagonal_offset = 0;
  bool is_unit_diagonal = false;
  bool operator==(const StructureAttr&) const = default;
};

enum class LayoutKind : uint8_t {
  Unassigned,
  Blocked,
  ThreadgroupStaged,
  SimdgroupFragment,
  Replicated,
};

enum class SwizzleMode : uint8_t { None, PadBank4, PadBank8, Xor8 };

enum class FragRole : uint8_t { MatrixA, MatrixB, Accumulator };

struct LayoutAttr {
  LayoutKind kind = LayoutKind::Unassigned;
  uint32_t vec_width = 1;
  uint32_t alignment_bytes = 4;
  uint32_t contiguity_elems = 1;
  uint32_t tg_threads[3] = {1, 1, 1};
  uint32_t elems_per_thread[2] = {1, 1};
  SwizzleMode swizzle = SwizzleMode::None;
  uint32_t pad_elems = 0;
  uint32_t stages = 1;
  FragRole frag_role = FragRole::Accumulator;
  uint32_t frag_tiles_m = 1;
  uint32_t frag_tiles_n = 1;
  uint32_t simd_grid[2] = {1, 1};
  bool operator==(const LayoutAttr&) const = default;
};

struct TileType {
  std::vector<int64_t> shape;
  ElemType elem = ScalarDType::F32;
  LayoutAttr layout;
  StructureAttr structure;
  bool operator==(const TileType&) const = default;
};

struct Type {
  std::variant<ScalarDType, PtrType, TileType> value = ScalarDType::F32;

  static Type scalar(ScalarDType dt) { return Type{dt}; }
  static Type ptr(ScalarDType pointee, AddrSpace space = AddrSpace::Device) {
    return Type{PtrType{pointee, space}};
  }
  static Type tile(std::vector<int64_t> shape, ElemType elem, LayoutAttr layout = {}, StructureAttr structure = {}) {
    return Type{TileType{std::move(shape), elem, layout, structure}};
  }

  bool is_scalar() const { return std::holds_alternative<ScalarDType>(value); }
  bool is_ptr() const { return std::holds_alternative<PtrType>(value); }
  bool is_tile() const { return std::holds_alternative<TileType>(value); }

  ScalarDType as_scalar() const { return std::get<ScalarDType>(value); }
  const PtrType& as_ptr() const { return std::get<PtrType>(value); }
  const TileType& as_tile() const { return std::get<TileType>(value); }
  TileType& as_tile_mut() { return std::get<TileType>(value); }

  ElemType elem_type() const;
  bool is_float_elem() const;
  bool is_int_elem() const;
  bool is_bool_elem() const;
  bool is_ptr_elem() const;
  ScalarDType scalar_elem() const;
  PtrType ptr_elem() const;
  uint32_t elem_bytes() const;
  bool same_shape_and_elem(const Type& other) const;

  bool operator==(const Type&) const = default;
};

std::string_view dtype_name(ScalarDType dt);
std::optional<ScalarDType> parse_dtype(std::string_view s);
uint32_t dtype_bytes(ScalarDType dt);
std::string_view addr_space_name(AddrSpace space);
std::string_view structure_name(MatrixStructure kind);
std::string type_to_string(const Type& ty);

}  // namespace m1triton::ir
