#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "ir/types.hpp"

namespace m1triton::ir {

using ValueId = uint32_t;
using OpId = uint32_t;
using BlockId = uint32_t;
inline constexpr uint32_t kInvalidId = UINT32_MAX;

enum class UnaryOp : uint8_t {
  Exp,
  Log,
  Sqrt,
  Rsqrt,
  Abs,
  Neg,
  Tanh,
  Erf,
  Floor,
  Ceil,
  Sin,
  Cos,
  Sigmoid,
  Not,
};

enum class BinaryOp : uint8_t {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Max,
  Min,
  Pow,
  And,
  Or,
  Xor,
  Shl,
  Shr,
  CmpEq,
  CmpNe,
  CmpLt,
  CmpLe,
  CmpGt,
  CmpGe,
};

enum class ReduceKind : uint8_t { Sum, Max, Min, Argmax, Argmin };
enum class ReduceResultMode : uint8_t { ValueOnly, IndexOnly, ValueAndIndex };
enum class ScanKind : uint8_t { Cumsum, Cumprod };
enum class BarrierScope : uint8_t { Simdgroup, Threadgroup, DeviceMem };
enum class TensorLayoutHint : uint8_t { NCHW, NHWC, NCDHW, RowMajor, ColMajor };

struct TileSpecAttr {
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t num_simdgroups = 4;
  bool double_buffer = false;
  uint32_t split_k = 1;
  uint32_t swizzle_w = 8;
  bool operator==(const TileSpecAttr&) const = default;
};

struct VecWidthAttr {
  uint32_t width = 1;
  bool operator==(const VecWidthAttr&) const = default;
};

struct LayoutHintAttr {
  TensorLayoutHint layout = TensorLayoutHint::NCHW;
  bool prefer_vec4 = true;
  bool operator==(const LayoutHintAttr&) const = default;
};

struct PersistentGridAttr {
  bool enabled = false;
  uint32_t max_tgs_per_core = 2;
  uint32_t swizzle_w = 8;
  bool operator==(const PersistentGridAttr&) const = default;
};

struct OpHints {
  std::optional<TileSpecAttr> tile_spec;
  std::optional<StructureAttr> structure;
  std::optional<VecWidthAttr> vec_width;
  std::optional<LayoutHintAttr> layout_hint;
  std::optional<PersistentGridAttr> persistent_grid;
  bool operator==(const OpHints&) const = default;
};

struct EmptyAttr {
  bool operator==(const EmptyAttr&) const = default;
};

struct ConstAttr {
  std::variant<int64_t, uint64_t, double, bool> val = int64_t{0};
  bool operator==(const ConstAttr&) const = default;
};

struct AxisAttr {
  uint32_t axis = 0;
  bool operator==(const AxisAttr&) const = default;
};

struct ArangeAttr {
  int64_t start = 0;
  int64_t end = 0;
  bool operator==(const ArangeAttr&) const = default;
};

struct PermuteAttr {
  std::vector<uint32_t> perm;
  bool operator==(const PermuteAttr&) const = default;
};

struct UnaryAttr {
  UnaryOp op = UnaryOp::Neg;
  bool operator==(const UnaryAttr&) const = default;
};

struct BinaryAttr {
  BinaryOp op = BinaryOp::Add;
  bool operator==(const BinaryAttr&) const = default;
};

struct LoadStoreAttr {
  uint32_t alignment_bytes = 4;
  uint32_t vec_width = 1;
  bool has_mask = false;
  bool has_other = false;
  bool is_volatile = false;
  bool non_temporal = false;
  bool operator==(const LoadStoreAttr&) const = default;
};

struct DotAttr {
  ScalarDType acc_dtype = ScalarDType::F32;
  bool allow_tf32 = false;
  bool transpose_a = false;
  bool transpose_b = false;
  StructureAttr lhs_struct;
  StructureAttr rhs_struct;
  StructureAttr out_struct;
  bool operator==(const DotAttr&) const = default;
};

struct ReduceAttr {
  ReduceKind kind = ReduceKind::Sum;
  ReduceResultMode result_mode = ReduceResultMode::ValueOnly;
  int32_t axis = -1;
  bool keep_dims = false;
  bool operator==(const ReduceAttr&) const = default;
};

struct ScanAttr {
  ScanKind kind = ScanKind::Cumsum;
  int32_t axis = -1;
  bool exclusive = false;
  bool reverse = false;
  bool has_mask = false;
  bool operator==(const ScanAttr&) const = default;
};

struct ConvAttr {
  uint32_t ndim = 2;
  std::vector<int32_t> stride;
  std::vector<int32_t> padding;
  std::vector<int32_t> dilation;
  int32_t groups = 1;
  bool transposed = false;
  TensorLayoutHint layout = TensorLayoutHint::NCHW;
  bool operator==(const ConvAttr&) const = default;
};

struct FlashAttnAttr {
  float scale = 1.0f;
  bool causal = false;
  int32_t causal_offset = 0;
  bool operator==(const FlashAttnAttr&) const = default;
};

struct ThreadgroupAllocAttr {
  ScalarDType elem = ScalarDType::F32;
  std::vector<int64_t> shape;
  SwizzleMode swizzle = SwizzleMode::None;
  uint32_t pad_elems = 0;
  uint32_t stages = 1;
  bool operator==(const ThreadgroupAllocAttr&) const = default;
};

struct ThreadgroupStageAttr {
  uint32_t stage_idx = 0;
  bool zero_pad_oob = true;
  bool operator==(const ThreadgroupStageAttr&) const = default;
};

struct BarrierAttr {
  BarrierScope scope = BarrierScope::Threadgroup;
  bool operator==(const BarrierAttr&) const = default;
};

struct SimdgroupMatmulAttr {
  uint32_t m_tiles = 4;
  uint32_t n_tiles = 4;
  uint32_t k_step = 8;
  uint32_t ldm_a = 16;
  uint32_t ldm_b = 64;
  ScalarDType acc_dtype = ScalarDType::F32;
  bool operator==(const SimdgroupMatmulAttr&) const = default;
};

using OpPayload = std::variant<
    EmptyAttr,
    ConstAttr,
    AxisAttr,
    ArangeAttr,
    PermuteAttr,
    UnaryAttr,
    BinaryAttr,
    LoadStoreAttr,
    DotAttr,
    ReduceAttr,
    ScanAttr,
    ConvAttr,
    FlashAttnAttr,
    ThreadgroupAllocAttr,
    ThreadgroupStageAttr,
    BarrierAttr,
    SimdgroupMatmulAttr>;

std::string_view unary_op_name(UnaryOp op);
std::string_view binary_op_name(BinaryOp op);
bool is_comparison_op(BinaryOp op);
bool is_commutative_op(BinaryOp op);

}  // namespace m1triton::ir
