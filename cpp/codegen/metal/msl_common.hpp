#pragma once

#include "codegen/metal/launch_spec.hpp"
#include "ir/ir.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace m1triton::codegen::metal {

struct HelperFlags {
  bool needs_erf = false;
  bool needs_tanh = false;
  bool needs_nan_minmax = false;
};

struct LoadOperands {
  ir::ValueId ptrs = ir::kInvalidId;
  ir::ValueId mask = ir::kInvalidId;
  ir::ValueId other = ir::kInvalidId;
  ir::LoadStoreAttr attr;
};

struct StoreOperands {
  ir::ValueId ptrs = ir::kInvalidId;
  ir::ValueId val = ir::kInvalidId;
  ir::ValueId mask = ir::kInvalidId;
  ir::LoadStoreAttr attr;
};

bool is_ptr_type(const ir::Type& ty);
ir::ScalarDType scalar_dtype_of(const ir::Type& ty);
bool is_float_dtype(ir::ScalarDType dt);
bool is_int_dtype(ir::ScalarDType dt);
bool can_vec4_mem(ir::ScalarDType dt);

std::string msl_scalar_type(ir::ScalarDType dt);
std::string msl_vec4_type(ir::ScalarDType dt);
std::string msl_value_type(ir::ScalarDType dt, bool vec4);
ir::ScalarDType common_cmp_dtype(ir::ScalarDType a, ir::ScalarDType b);

std::string format_float_literal(double d);
std::string format_scalar_literal_from_i64(int64_t v, ir::ScalarDType dt);
std::string msl_const_literal(const ir::ConstAttr& attr, ir::ScalarDType dt);
std::optional<int64_t> const_as_i64(const ir::ConstAttr& attr);
std::optional<double> const_as_f64(const ir::ConstAttr& attr);
std::string msl_zero_literal(ir::ScalarDType dt, bool vec4);
std::string msl_int_literal(int64_t v);
std::string msl_exact_float_literal(double v);

std::string msl_cast_expr(const std::string& expr, ir::ScalarDType src_dt, ir::ScalarDType dst_dt, bool vec4);
std::string promote_to_vec4(const std::string& expr, ir::ScalarDType dt, bool is_already_vec4);
std::string msl_unary_expr(ir::UnaryOp op, const std::string& x, ir::ScalarDType dt, bool vec4, HelperFlags& helpers);
std::string msl_binary_expr(ir::BinaryOp op, const std::string& lhs_in, const std::string& rhs_in,
                            ir::ScalarDType lhs_dt, ir::ScalarDType rhs_dt, ir::ScalarDType res_dt, bool vec4,
                            HelperFlags* helpers = nullptr);
std::string msl_select_expr(const std::string& cond, const std::string& t_val, const std::string& f_val,
                            ir::ScalarDType t_dt, ir::ScalarDType f_dt, ir::ScalarDType res_dt, bool vec4);
std::string combine_add_expr(const std::string& a, bool a_zero, const std::string& b, bool b_zero, ir::ScalarDType dt);
std::string combine_sub_expr(const std::string& a, bool a_zero, const std::string& b, bool b_zero, ir::ScalarDType dt);

std::string indent_str(int depth);
std::string ssa_var(ir::ValueId id);
std::string axis_component(uint32_t axis);
uint32_t round_up_32(uint32_t v);
uint32_t fit_divisor(uint32_t threads, uint32_t block);

void emit_helper_functions(std::ostringstream& os, const HelperFlags& helpers);
void emit_param_slot_decls(std::ostringstream& os, const std::vector<ParamSlot>& param_slots);
std::string build_msl_preamble_and_sig(const ir::Kernel& kernel, const std::vector<ParamSlot>& param_slots,
                                       const HelperFlags& helpers, bool include_simd_ids = false);
void emit_kernel_param_decls(std::ostringstream& os, const ir::Kernel& kernel, std::string_view out_name);
std::vector<ParamSlot> build_single_out_param_slots(const ir::Kernel& kernel, std::string_view out_name);
std::string msl_prefixed_param_name(const ir::KernelParam& kp, uint32_t idx);
std::string emit_prefixed_kernel_sig(const ir::Kernel& kernel, const std::vector<ParamSlot>& slots);

LoadOperands extract_load_operands(const ir::Op& op, const ir::Kernel& kernel);
StoreOperands extract_store_operands(const ir::Op& op);
void collect_blocks_dfs(const ir::Kernel& kernel, ir::BlockId bid, std::vector<ir::BlockId>& order,
                        std::unordered_set<ir::BlockId>& seen);
void map_entry_args_to_params(const ir::Kernel& kernel, std::unordered_map<ir::ValueId, uint32_t>& val_to_param_idx);
void build_param_slots(const ir::Kernel& kernel, const std::unordered_set<uint32_t>& written_params,
                       std::vector<ParamSlot>& param_slots,
                       std::unordered_map<uint32_t, std::string>& param_msl_name);
bool ptr_value_is_writable(const std::unordered_map<ir::ValueId, std::set<uint32_t>>& ptr_origins,
                           const std::vector<ParamSlot>& param_slots, ir::ValueId vid);
std::string msl_ptr_type_str(const std::unordered_map<ir::ValueId, std::set<uint32_t>>& ptr_origins,
                             const std::vector<ParamSlot>& param_slots, ir::ValueId vid, ir::ScalarDType pointee);
void populate_entry_param_exprs(const ir::Kernel& kernel,
                                const std::unordered_map<ir::ValueId, uint32_t>& val_to_param_idx,
                                const std::unordered_map<uint32_t, std::string>& param_msl_name,
                                std::unordered_map<ir::ValueId, std::string>& expr_map);
std::string emit_scalar_op_expr(const ir::Kernel& kernel, const ir::Op& op,
                                const std::unordered_map<ir::ValueId, std::string>& expr_map, HelperFlags& helpers,
                                std::string& err, bool allow_unit_tile_ops = false,
                                const char* unsupported_err = "Unsupported scalar op kind in MetalCodegen");

std::optional<uint32_t> param_index_of(const ir::Kernel& kernel, ir::ValueId vid);
std::optional<std::string> scalar_long_expr(const ir::Kernel& kernel, ir::ValueId vid, bool allow_bool = true);
std::optional<int64_t> positive_constexpr_param(const ir::Kernel& kernel, const char* name);
int64_t positive_constexpr_param(const ir::Kernel& kernel, const char* name, int64_t fallback);

void collect_epilogue_dag_helpers(const ir::EpilogueDAGAttr& dag, HelperFlags& helpers);
void collect_epilogue_dag_param_indices(const ir::EpilogueDAGAttr& dag, std::set<uint32_t>& read_params);
void emit_epilogue_dag_msl(std::ostringstream& os, const ir::Kernel& kernel, const ir::EpilogueDAGAttr& dag,
                           const std::string& val_var, const std::string& ch_expr, const std::string& elem_idx_expr,
                           const std::function<std::string(uint32_t)>& param_name_fn, HelperFlags& helpers,
                           const std::string& indent = "  ");

}  // namespace m1triton::codegen::metal
