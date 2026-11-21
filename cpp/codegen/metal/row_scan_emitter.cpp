#include "codegen/metal/row_scan_emitter.hpp"

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::Kernel;
using ir::OpKind;
using ir::ScalarDType;
using ir::ScanAttr;
using ir::ScanKind;
using ir::ValueId;

constexpr uint32_t kRowsPerThreadgroup = 8;
constexpr uint32_t kBlocksPerGroup = 4;

// Device helpers shared by every scan variant. M1_ID / M1_OP / M1_REV / M1_MASK are defined per kernel.
// Logical element l runs along the scan direction; the physical index is l (forward) or n-1-l (reverse),
// so a thread's four logical elements always map to one aligned, coalesced float4 / uchar4 access.
constexpr const char* kHelpers = R"MSL(
inline float4 m1_load4(device const float* src, device const uchar* msk, long n, long l0, bool vec4) {
  float4 v = float4(M1_ID);
  if (vec4) {
    if (l0 < n) {
      const long p = M1_REV ? n - 4 - l0 : l0;
      v = *(device const float4*)(src + p);
#if M1_MASK
      v *= float4(*(device const uchar4*)(msk + p));
#endif
      if (M1_REV) v = v.wzyx;
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      const long l = l0 + k;
      if (l < n) {
        const long p = M1_REV ? n - 1 - l : l;
        float e = src[p];
#if M1_MASK
        e *= float(msk[p]);
#endif
        v[k] = e;
      }
    }
  }
  return v;
}

inline void m1_store4(device float* dst, long n, long l0, bool vec4, float4 o) {
  if (vec4) {
    if (l0 < n) {
      if (M1_REV) o = o.wzyx;
      *(device float4*)(dst + (M1_REV ? n - 4 - l0 : l0)) = o;
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      const long l = l0 + k;
      if (l < n) dst[M1_REV ? n - 1 - l : l] = o[k];
    }
  }
}
)MSL";

constexpr const char* kSumSimdScan = R"MSL(
    const float sg_exc = simd_prefix_exclusive_sum(s3);
    const float sg_tot = simd_broadcast(sg_exc + s3, 31);
)MSL";

constexpr const char* kProdSimdScan = R"MSL(
    float pp = s3;
    for (ushort d = 1; d < 32; d <<= 1) {
      const float up = simd_shuffle_up(pp, d);
      if (lane >= d) pp *= up;
    }
    const float sg_exc = lane == 0 ? 1.0f : simd_shuffle_up(pp, 1);
    const float sg_tot = simd_broadcast(pp, 31);
)MSL";

// One SIMD group scans one row with no threadgroup barriers: each lane owns a float4 per block, the
// 32 lanes scan 128 floats with simd_prefix_*, and a uniform carry chains blocks. Loads for the next
// group of kBlocks are issued before the current group is scanned so DRAM latency stays hidden.
constexpr const char* kKernelBody = R"MSL(
  const long row = long(tg_pos) * M1_ROWS_PER_TG + long(sg);
  if (row >= long($ROWS)) return;
  const long n = long($LEN);
  if (n <= 0) return;
  const long base = row * n;
  device const float* src = $X + base;
  device float* dst = $OUT + base;
  device const uchar* msk = (device const uchar*)$MASK + base;
  const bool vec4 = (n & 3) == 0;
  const long ngroups = (n + M1_GROUP - 1) / M1_GROUP;
  const long lane_off = long(lane) * 4;

  float carry = M1_ID;
  float4 cur[M1_BLOCKS];
  for (int j = 0; j < M1_BLOCKS; ++j) cur[j] = m1_load4(src, msk, n, j * 128 + lane_off, vec4);
  for (long g = 0; g < ngroups; ++g) {
    const long g0 = g * M1_GROUP + lane_off;
    float4 nxt[M1_BLOCKS];
    for (int j = 0; j < M1_BLOCKS; ++j) {
      nxt[j] = (g + 1 < ngroups) ? m1_load4(src, msk, n, g0 + M1_GROUP + j * 128, vec4) : float4(M1_ID);
    }
    for (int j = 0; j < M1_BLOCKS; ++j) {
      const float s0 = cur[j].x;
      const float s1 = M1_OP(s0, cur[j].y);
      const float s2 = M1_OP(s1, cur[j].z);
      const float s3 = M1_OP(s2, cur[j].w);
$SIMD_SCAN
      const float prefix = M1_OP(carry, sg_exc);
      carry = M1_OP(carry, sg_tot);
#if M1_EXCL
      const float4 o = float4(prefix, M1_OP(prefix, s0), M1_OP(prefix, s1), M1_OP(prefix, s2));
#else
      const float4 o = float4(M1_OP(prefix, s0), M1_OP(prefix, s1), M1_OP(prefix, s2), M1_OP(prefix, s3));
#endif
      m1_store4(dst, n, g0 + j * 128, vec4, o);
    }
    for (int j = 0; j < M1_BLOCKS; ++j) cur[j] = nxt[j];
  }
)MSL";

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  for (size_t pos = s.find(from); pos != std::string::npos; pos = s.find(from, pos + to.size())) {
    s.replace(pos, from.size(), to);
  }
}

// Maps each entry-block argument value to the kernel parameter index it carries.
std::unordered_map<ValueId, uint32_t> map_entry_args(const Kernel& kernel) {
  std::unordered_map<ValueId, uint32_t> out;
  const auto& entry = kernel.block(kernel.entry_block);
  std::vector<uint32_t> runtime_params;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    if (!kernel.params[i].is_constexpr) runtime_params.push_back(i);
  }
  for (size_t k = 0; k < entry.args.size(); ++k) {
    if (entry.args.size() == kernel.params.size()) {
      out[entry.args[k]] = static_cast<uint32_t>(k);
    } else if (entry.args.size() == runtime_params.size()) {
      out[entry.args[k]] = runtime_params[k];
    } else {
      out[entry.args[k]] = kernel.value(entry.args[k]).arg_idx;
    }
  }
  return out;
}

const char* msl_elem_type(ScalarDType dt) {
  switch (dt) {
    case ScalarDType::F32:
      return "float";
    case ScalarDType::F16:
      return "half";
    case ScalarDType::I32:
      return "int";
    case ScalarDType::U32:
      return "uint";
    case ScalarDType::I64:
      return "long";
    case ScalarDType::Bool:
      return "bool";
  }
  return "float";
}

}  // namespace

CompiledMSL try_emit_row_scan_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  (void)target;
  if (matched_out != nullptr) *matched_out = false;
  const CompiledMSL unmatched{false, "", {}, ""};

  const ir::Op* scan_op = nullptr;
  for (const auto& op : kernel.ops) {
    if (op->erased || op->kind != OpKind::SimdScan) continue;
    if (scan_op != nullptr) return unmatched;
    scan_op = op.get();
  }
  if (scan_op == nullptr || !std::holds_alternative<ScanAttr>(scan_op->attr)) return unmatched;
  const auto& sa = std::get<ScanAttr>(scan_op->attr);
  if (scan_op->operands.size() != (sa.has_mask ? 5u : 4u)) return unmatched;

  const auto arg_to_param = map_entry_args(kernel);
  std::vector<uint32_t> operand_params;
  for (ValueId vid : scan_op->operands) {
    const auto it = arg_to_param.find(vid);
    if (it == arg_to_param.end() || it->second >= kernel.params.size()) return unmatched;
    operand_params.push_back(it->second);
  }

  const size_t shift = sa.has_mask ? 1 : 0;
  const uint32_t x_idx = operand_params[0];
  const uint32_t out_idx = operand_params[1];
  const uint32_t rows_idx = operand_params[2 + shift];
  const uint32_t len_idx = operand_params[3 + shift];
  for (uint32_t idx : {x_idx, out_idx}) {
    const auto& ty = kernel.params[idx].type;
    if (!ty.is_ptr_elem() || ty.ptr_elem().pointee != ScalarDType::F32) return unmatched;
  }
  if (kernel.params[rows_idx].type.is_ptr_elem() || kernel.params[len_idx].type.is_ptr_elem()) return unmatched;
  if (matched_out != nullptr) *matched_out = true;

  CompiledMSL res;
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = 32 * kRowsPerThreadgroup;
  res.launch_spec.vec_width = 4;

  const bool is_sum = sa.kind == ScanKind::Cumsum;
  std::ostringstream os;
  os << "#include <metal_stdlib>\nusing namespace metal;\n\n";
  os << "#define M1_ID " << (is_sum ? "0.0f" : "1.0f") << "\n";
  os << "#define M1_OP(a, b) ((a) " << (is_sum ? "+" : "*") << " (b))\n";
  os << "#define M1_REV " << (sa.reverse ? 1 : 0) << "\n";
  os << "#define M1_EXCL " << (sa.exclusive ? 1 : 0) << "\n";
  os << "#define M1_MASK " << (sa.has_mask ? 1 : 0) << "\n";
  os << "#define M1_ROWS_PER_TG " << kRowsPerThreadgroup << "\n";
  os << "#define M1_BLOCKS " << kBlocksPerGroup << "\n";
  os << "#define M1_GROUP " << kBlocksPerGroup * 128 << "\n";
  os << kHelpers << "\nkernel void " << kernel.name << "(\n";

  std::vector<std::string> names(kernel.params.size());
  uint32_t next_slot = 0;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const auto& kp = kernel.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot slot;
    slot.param_index = i;
    slot.buffer_index = next_slot++;
    slot.name = kp.name.empty() ? "m1_arg" + std::to_string(slot.buffer_index) : kp.name;
    slot.is_pointer = kp.type.is_ptr_elem();
    slot.dtype = slot.is_pointer ? kp.type.ptr_elem().pointee : kp.type.scalar_elem();
    slot.writable = slot.is_pointer && (kp.writable || i == out_idx);
    names[i] = slot.name;
    if (slot.is_pointer) {
      os << "    device " << (slot.writable ? "" : "const ") << msl_elem_type(slot.dtype) << "* " << slot.name;
    } else {
      os << "    constant " << msl_elem_type(slot.dtype) << "& " << slot.name;
    }
    os << " [[buffer(" << slot.buffer_index << ")]],\n";
    res.launch_spec.param_slots.push_back(std::move(slot));
  }
  os << "    uint tg_pos [[threadgroup_position_in_grid]],\n"
     << "    ushort lane [[thread_index_in_simdgroup]],\n"
     << "    ushort sg [[simdgroup_index_in_threadgroup]]) {";

  std::string body = kKernelBody;
  replace_all(body, "$SIMD_SCAN", is_sum ? kSumSimdScan : kProdSimdScan);
  replace_all(body, "$ROWS", names[rows_idx]);
  replace_all(body, "$LEN", names[len_idx]);
  replace_all(body, "$OUT", names[out_idx]);
  replace_all(body, "$MASK", sa.has_mask ? names[operand_params[2]] : names[x_idx]);
  replace_all(body, "$X", names[x_idx]);
  os << body << "}\n";

  res.msl_source = os.str();
  res.ok = true;
  return res;
}

}  // namespace m1triton::codegen::metal
