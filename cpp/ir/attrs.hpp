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
enum class PoolKind : uint8_t { Max, Avg };
enum class NormKind : uint8_t {
  LayerNorm,
  BatchNorm,
  InstanceNorm,
  GroupNorm,
  RMSNorm,
  FrobeniusNorm,
  L1Norm,
  L2Norm,
};
enum class LossKind : uint8_t {
  MSE,
  CrossEntropy,
  Huber,
  KLDiv,
  TripletMargin,
  Hinge,
};
enum class BarrierScope : uint8_t { Simdgroup, Threadgroup, DeviceMem };
enum class TensorLayoutHint : uint8_t { NCHW, NHWC, NCDHW, RowMajor, ColMajor };

enum class EpilogueOpKind : uint8_t {
  AddConst,
  SubConst,
  RSubConst,
  MulConst,
  DivConst,
  MinConst,
  MaxConst,
  Clamp,
  AddBias,
  SubBias,
  MulBias,
  DivBias,
  MinBias,
  MaxBias,
  AddScalar,
  SubScalar,
  MulScalar,
  DivScalar,
  AddResidual,
  SubResidual,
  MulResidual,
  SaveReg,
  AddSaved,
  SubSaved,
  MulSaved,
  ReLU,
  LeakyReLU,
  Sigmoid,
  Tanh,
  GELU,
  GELUTanh,
  Swish,
  Mish,
  HardSwish,
  HardSigmoid,
  HardTanh,
  ELU,
  SELU,
  Softplus,
  Softsign,
  Log,
  Exp,
  Abs,
  Neg,
  Sqrt,
};

struct EpilogueStep {
  EpilogueOpKind op = EpilogueOpKind::ReLU;
  float param0 = 0.0f;
  float param1 = 0.0f;
  int32_t arg_idx = -1;
  bool operator==(const EpilogueStep&) const = default;
};

struct EpilogueDAGAttr {
  std::vector<EpilogueStep> steps;
  bool operator==(const EpilogueDAGAttr&) const = default;
};

enum class TailKind : uint8_t {
  None,
  RowSum,
  RowMean,
  RowMax,
  RowMin,
  RowLogSumExp,
  SpatialGlobalAvgPool,
  SpatialGlobalSum,
  SpatialGlobalMax,
  Pool1DMax,
  Pool1DAvg,
};

struct FusedTailAttr {
  TailKind kind = TailKind::None;
  int32_t kernel_size = 1;
  int32_t stride = 1;
  float post_scale = 1.0f;
  bool keep_dims = true;
  bool operator==(const FusedTailAttr&) const = default;
};

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
  std::optional<EpilogueDAGAttr> epilogue_dag;
  std::optional<FusedTailAttr> fused_tail;
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
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> padding;
  std::vector<int32_t> dilation;
  std::vector<int32_t> output_padding;
  int32_t groups = 1;
  bool transposed = false;
  bool has_bias = false;
  bool is_depthwise = false;
  bool is_pointwise = false;
  bool use_sram_halo = false;
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t num_simdgroups = 4;
  TensorLayoutHint layout = TensorLayoutHint::NCHW;
  bool operator==(const ConvAttr&) const = default;
};

struct FlashAttnAttr {
  float scale = 1.0f;
  bool causal = false;
  int32_t causal_offset = 0;
  uint32_t bm = 16;
  uint32_t bn = 64;
  uint32_t bd = 32;
  uint32_t num_simdgroups = 16;
  bool operator==(const FlashAttnAttr&) const = default;
};

struct LossAttr {
  LossKind kind = LossKind::MSE;
  float margin = 1.0f;
  float beta = 1.0f;
  float eps = 1e-6f;
  int32_t stage = 1;
  uint32_t tg_threads = 256;
  bool operator==(const LossAttr&) const = default;
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

struct OnlineSoftmaxAttr {
  int32_t axis = -1;
  bool log_softmax = false;
  uint32_t tg_threads = 256;
  uint32_t vec_width = 4;
  bool operator==(const OnlineSoftmaxAttr&) const = default;
};

struct WelfordNormAttr {
  NormKind kind = NormKind::LayerNorm;
  float eps = 1e-5f;
  int32_t axis = -1;
  int32_t num_groups = 1;
  int32_t stage = 0;
  bool has_weight = false;
  bool has_bias = false;
  bool rms_sram_cache = false;
  uint32_t tg_threads = 256;
  uint32_t vec_width = 4;
  bool operator==(const WelfordNormAttr&) const = default;
};

struct PoolAttr {
  PoolKind kind = PoolKind::Max;
  uint32_t ndim = 2;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> padding;
  std::vector<int32_t> dilation;
  bool count_include_pad = true;
  bool ceil_mode = false;
  bool use_sram_halo = false;
  uint32_t tg_x = 0;
  uint32_t tg_y = 0;
  bool operator==(const PoolAttr&) const = default;
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
    SimdgroupMatmulAttr,
    OnlineSoftmaxAttr,
    WelfordNormAttr,
    PoolAttr,
    LossAttr>;

std::string_view unary_op_name(UnaryOp op);
std::string_view binary_op_name(BinaryOp op);
bool is_comparison_op(BinaryOp op);
bool is_commutative_op(BinaryOp op);

std::string_view epilogue_op_name(EpilogueOpKind op);
std::optional<EpilogueOpKind> parse_epilogue_op_name(std::string_view name);
bool epilogue_op_needs_ptr_arg(EpilogueOpKind op);
bool epilogue_op_needs_scalar_arg(EpilogueOpKind op);
bool epilogue_op_uses_reg_slot(EpilogueOpKind op);
std::string_view tail_kind_name(TailKind kind);
std::optional<TailKind> parse_tail_kind_name(std::string_view name);

}  // namespace m1triton::ir
