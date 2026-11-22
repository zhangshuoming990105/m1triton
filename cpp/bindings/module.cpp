#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "codegen/metal/launch_spec.hpp"
#include "codegen/metal/metal_codegen.hpp"
#include "ir/attrs.hpp"
#include "ir/builder.hpp"
#include "ir/ir.hpp"
#include "ir/printer.hpp"
#include "ir/target_spec.hpp"
#include "ir/types.hpp"
#include "ir/verifier.hpp"
#include "passes/pass_manager.hpp"
#include "runtime/metal_runtime.hpp"

namespace py = pybind11;

namespace pybind11::detail {
template <>
struct is_copy_constructible<m1triton::ir::Kernel> : std::false_type {};
template <>
struct is_copy_constructible<m1triton::ir::Module> : std::false_type {};
}  // namespace pybind11::detail

namespace {

void write_py_stderr(const std::string& text) {
  py::module_::import("sys").attr("stderr").attr("write")(text);
}

void bind_ir_submodule(py::module_& m) {
  namespace ir = m1triton::ir;
  auto ir_m = m.def_submodule("ir", "M1Triton pure C++20 SSA IR bindings");

  ir_m.attr("INVALID_ID") = ir::kInvalidId;

  py::enum_<ir::ScalarDType>(ir_m, "ScalarDType")
      .value("F32", ir::ScalarDType::F32)
      .value("F16", ir::ScalarDType::F16)
      .value("I32", ir::ScalarDType::I32)
      .value("U32", ir::ScalarDType::U32)
      .value("I64", ir::ScalarDType::I64)
      .value("Bool", ir::ScalarDType::Bool);

  py::enum_<ir::AddrSpace>(ir_m, "AddrSpace")
      .value("Device", ir::AddrSpace::Device)
      .value("Constant", ir::AddrSpace::Constant)
      .value("Threadgroup", ir::AddrSpace::Threadgroup);

  py::enum_<ir::MatrixStructure>(ir_m, "MatrixStructure")
      .value("Dense", ir::MatrixStructure::Dense)
      .value("UpperTriangular", ir::MatrixStructure::UpperTriangular)
      .value("LowerTriangular", ir::MatrixStructure::LowerTriangular)
      .value("Symmetric", ir::MatrixStructure::Symmetric)
      .value("Diagonal", ir::MatrixStructure::Diagonal)
      .value("CausalMask", ir::MatrixStructure::CausalMask);

  py::enum_<ir::LayoutKind>(ir_m, "LayoutKind")
      .value("Unassigned", ir::LayoutKind::Unassigned)
      .value("Blocked", ir::LayoutKind::Blocked)
      .value("ThreadgroupStaged", ir::LayoutKind::ThreadgroupStaged)
      .value("SimdgroupFragment", ir::LayoutKind::SimdgroupFragment)
      .value("Replicated", ir::LayoutKind::Replicated);

  py::enum_<ir::SwizzleMode>(ir_m, "SwizzleMode")
      .value("None", ir::SwizzleMode::None)
      .value("PadBank4", ir::SwizzleMode::PadBank4)
      .value("PadBank8", ir::SwizzleMode::PadBank8)
      .value("Xor8", ir::SwizzleMode::Xor8);

  py::enum_<ir::FragRole>(ir_m, "FragRole")
      .value("MatrixA", ir::FragRole::MatrixA)
      .value("MatrixB", ir::FragRole::MatrixB)
      .value("Accumulator", ir::FragRole::Accumulator);

  py::enum_<ir::UnaryOp>(ir_m, "UnaryOp")
      .value("Exp", ir::UnaryOp::Exp)
      .value("Log", ir::UnaryOp::Log)
      .value("Sqrt", ir::UnaryOp::Sqrt)
      .value("Rsqrt", ir::UnaryOp::Rsqrt)
      .value("Abs", ir::UnaryOp::Abs)
      .value("Neg", ir::UnaryOp::Neg)
      .value("Tanh", ir::UnaryOp::Tanh)
      .value("Erf", ir::UnaryOp::Erf)
      .value("Floor", ir::UnaryOp::Floor)
      .value("Ceil", ir::UnaryOp::Ceil)
      .value("Sin", ir::UnaryOp::Sin)
      .value("Cos", ir::UnaryOp::Cos)
      .value("Sigmoid", ir::UnaryOp::Sigmoid)
      .value("Not", ir::UnaryOp::Not);

  py::enum_<ir::BinaryOp>(ir_m, "BinaryOp")
      .value("Add", ir::BinaryOp::Add)
      .value("Sub", ir::BinaryOp::Sub)
      .value("Mul", ir::BinaryOp::Mul)
      .value("Div", ir::BinaryOp::Div)
      .value("Rem", ir::BinaryOp::Rem)
      .value("Max", ir::BinaryOp::Max)
      .value("Min", ir::BinaryOp::Min)
      .value("Pow", ir::BinaryOp::Pow)
      .value("And", ir::BinaryOp::And)
      .value("Or", ir::BinaryOp::Or)
      .value("Xor", ir::BinaryOp::Xor)
      .value("Shl", ir::BinaryOp::Shl)
      .value("Shr", ir::BinaryOp::Shr)
      .value("CmpEq", ir::BinaryOp::CmpEq)
      .value("CmpNe", ir::BinaryOp::CmpNe)
      .value("CmpLt", ir::BinaryOp::CmpLt)
      .value("CmpLe", ir::BinaryOp::CmpLe)
      .value("CmpGt", ir::BinaryOp::CmpGt)
      .value("CmpGe", ir::BinaryOp::CmpGe);

  py::enum_<ir::ReduceKind>(ir_m, "ReduceKind")
      .value("Sum", ir::ReduceKind::Sum)
      .value("Max", ir::ReduceKind::Max)
      .value("Min", ir::ReduceKind::Min)
      .value("Argmax", ir::ReduceKind::Argmax)
      .value("Argmin", ir::ReduceKind::Argmin);

  py::enum_<ir::ReduceResultMode>(ir_m, "ReduceResultMode")
      .value("ValueOnly", ir::ReduceResultMode::ValueOnly)
      .value("IndexOnly", ir::ReduceResultMode::IndexOnly)
      .value("ValueAndIndex", ir::ReduceResultMode::ValueAndIndex);

  py::enum_<ir::ScanKind>(ir_m, "ScanKind")
      .value("Cumsum", ir::ScanKind::Cumsum)
      .value("Cumprod", ir::ScanKind::Cumprod);

  py::enum_<ir::PoolKind>(ir_m, "PoolKind")
      .value("Max", ir::PoolKind::Max)
      .value("Avg", ir::PoolKind::Avg);

  py::enum_<ir::NormKind>(ir_m, "NormKind")
      .value("LayerNorm", ir::NormKind::LayerNorm)
      .value("BatchNorm", ir::NormKind::BatchNorm)
      .value("InstanceNorm", ir::NormKind::InstanceNorm)
      .value("GroupNorm", ir::NormKind::GroupNorm)
      .value("RMSNorm", ir::NormKind::RMSNorm)
      .value("FrobeniusNorm", ir::NormKind::FrobeniusNorm)
      .value("L1Norm", ir::NormKind::L1Norm)
      .value("L2Norm", ir::NormKind::L2Norm);

  py::enum_<ir::LossKind>(ir_m, "LossKind")
      .value("MSE", ir::LossKind::MSE)
      .value("CrossEntropy", ir::LossKind::CrossEntropy)
      .value("Huber", ir::LossKind::Huber)
      .value("KLDiv", ir::LossKind::KLDiv)
      .value("TripletMargin", ir::LossKind::TripletMargin)
      .value("Hinge", ir::LossKind::Hinge);

  py::enum_<ir::BarrierScope>(ir_m, "BarrierScope")
      .value("Simdgroup", ir::BarrierScope::Simdgroup)
      .value("Threadgroup", ir::BarrierScope::Threadgroup)
      .value("DeviceMem", ir::BarrierScope::DeviceMem);

  py::enum_<ir::TensorLayoutHint>(ir_m, "TensorLayoutHint")
      .value("NCHW", ir::TensorLayoutHint::NCHW)
      .value("NHWC", ir::TensorLayoutHint::NHWC)
      .value("NCDHW", ir::TensorLayoutHint::NCDHW)
      .value("RowMajor", ir::TensorLayoutHint::RowMajor)
      .value("ColMajor", ir::TensorLayoutHint::ColMajor);

  py::enum_<ir::VerifyStage>(ir_m, "VerifyStage")
      .value("HighLevel", ir::VerifyStage::HighLevel)
      .value("PostTile", ir::VerifyStage::PostTile)
      .value("PreCodegen", ir::VerifyStage::PreCodegen);

  py::class_<ir::PtrType>(ir_m, "PtrType")
      .def(py::init<>())
      .def(py::init([](ir::ScalarDType pointee, ir::AddrSpace space) {
             return ir::PtrType{pointee, space};
           }),
           py::arg("pointee") = ir::ScalarDType::F32,
           py::arg("space") = ir::AddrSpace::Device)
      .def_readwrite("pointee", &ir::PtrType::pointee)
      .def_readwrite("space", &ir::PtrType::space)
      .def("__eq__", [](const ir::PtrType& a, const ir::PtrType& b) { return a == b; });

  py::class_<ir::StructureAttr>(ir_m, "StructureAttr")
      .def(py::init<>())
      .def(py::init([](ir::MatrixStructure kind, int32_t diagonal_offset, bool is_unit_diagonal) {
             return ir::StructureAttr{kind, diagonal_offset, is_unit_diagonal};
           }),
           py::arg("kind") = ir::MatrixStructure::Dense,
           py::arg("diagonal_offset") = int32_t{0},
           py::arg("is_unit_diagonal") = false)
      .def_readwrite("kind", &ir::StructureAttr::kind)
      .def_readwrite("diagonal_offset", &ir::StructureAttr::diagonal_offset)
      .def_readwrite("is_unit_diagonal", &ir::StructureAttr::is_unit_diagonal)
      .def("__eq__", [](const ir::StructureAttr& a, const ir::StructureAttr& b) { return a == b; });

  py::class_<ir::TileSpecAttr>(ir_m, "TileSpecAttr")
      .def(py::init<>())
      .def_readwrite("bm", &ir::TileSpecAttr::bm)
      .def_readwrite("bn", &ir::TileSpecAttr::bn)
      .def_readwrite("bk", &ir::TileSpecAttr::bk)
      .def_readwrite("num_simdgroups", &ir::TileSpecAttr::num_simdgroups)
      .def_readwrite("double_buffer", &ir::TileSpecAttr::double_buffer)
      .def_readwrite("split_k", &ir::TileSpecAttr::split_k)
      .def_readwrite("swizzle_w", &ir::TileSpecAttr::swizzle_w)
      .def("__eq__", [](const ir::TileSpecAttr& a, const ir::TileSpecAttr& b) { return a == b; });

  py::class_<ir::VecWidthAttr>(ir_m, "VecWidthAttr")
      .def(py::init<>())
      .def(py::init([](uint32_t width) { return ir::VecWidthAttr{width}; }), py::arg("width") = uint32_t{1})
      .def_readwrite("width", &ir::VecWidthAttr::width)
      .def("__eq__", [](const ir::VecWidthAttr& a, const ir::VecWidthAttr& b) { return a == b; });

  py::class_<ir::LayoutHintAttr>(ir_m, "LayoutHintAttr")
      .def(py::init<>())
      .def_readwrite("layout", &ir::LayoutHintAttr::layout)
      .def_readwrite("prefer_vec4", &ir::LayoutHintAttr::prefer_vec4)
      .def("__eq__", [](const ir::LayoutHintAttr& a, const ir::LayoutHintAttr& b) { return a == b; });

  py::class_<ir::PersistentGridAttr>(ir_m, "PersistentGridAttr")
      .def(py::init<>())
      .def_readwrite("enabled", &ir::PersistentGridAttr::enabled)
      .def_readwrite("max_tgs_per_core", &ir::PersistentGridAttr::max_tgs_per_core)
      .def_readwrite("swizzle_w", &ir::PersistentGridAttr::swizzle_w)
      .def("__eq__", [](const ir::PersistentGridAttr& a, const ir::PersistentGridAttr& b) { return a == b; });

  py::class_<ir::OpHints>(ir_m, "OpHints")
      .def(py::init<>())
      .def_readwrite("tile_spec", &ir::OpHints::tile_spec)
      .def_readwrite("structure", &ir::OpHints::structure)
      .def_readwrite("vec_width", &ir::OpHints::vec_width)
      .def_readwrite("layout_hint", &ir::OpHints::layout_hint)
      .def_readwrite("persistent_grid", &ir::OpHints::persistent_grid)
      .def("__eq__", [](const ir::OpHints& a, const ir::OpHints& b) { return a == b; });

  py::class_<ir::LoadStoreAttr>(ir_m, "LoadStoreAttr")
      .def(py::init<>())
      .def_readwrite("alignment_bytes", &ir::LoadStoreAttr::alignment_bytes)
      .def_readwrite("vec_width", &ir::LoadStoreAttr::vec_width)
      .def_readwrite("has_mask", &ir::LoadStoreAttr::has_mask)
      .def_readwrite("has_other", &ir::LoadStoreAttr::has_other)
      .def_readwrite("is_volatile", &ir::LoadStoreAttr::is_volatile)
      .def_readwrite("non_temporal", &ir::LoadStoreAttr::non_temporal)
      .def("__eq__", [](const ir::LoadStoreAttr& a, const ir::LoadStoreAttr& b) { return a == b; });

  py::class_<ir::DotAttr>(ir_m, "DotAttr")
      .def(py::init<>())
      .def_readwrite("acc_dtype", &ir::DotAttr::acc_dtype)
      .def_readwrite("allow_tf32", &ir::DotAttr::allow_tf32)
      .def_readwrite("transpose_a", &ir::DotAttr::transpose_a)
      .def_readwrite("transpose_b", &ir::DotAttr::transpose_b)
      .def_readwrite("lhs_struct", &ir::DotAttr::lhs_struct)
      .def_readwrite("rhs_struct", &ir::DotAttr::rhs_struct)
      .def_readwrite("out_struct", &ir::DotAttr::out_struct)
      .def("__eq__", [](const ir::DotAttr& a, const ir::DotAttr& b) { return a == b; });

  py::class_<ir::ReduceAttr>(ir_m, "ReduceAttr")
      .def(py::init<>())
      .def_readwrite("kind", &ir::ReduceAttr::kind)
      .def_readwrite("result_mode", &ir::ReduceAttr::result_mode)
      .def_readwrite("axis", &ir::ReduceAttr::axis)
      .def_readwrite("keep_dims", &ir::ReduceAttr::keep_dims)
      .def("__eq__", [](const ir::ReduceAttr& a, const ir::ReduceAttr& b) { return a == b; });

  py::class_<ir::ScanAttr>(ir_m, "ScanAttr")
      .def(py::init<>())
      .def_readwrite("kind", &ir::ScanAttr::kind)
      .def_readwrite("axis", &ir::ScanAttr::axis)
      .def_readwrite("exclusive", &ir::ScanAttr::exclusive)
      .def_readwrite("reverse", &ir::ScanAttr::reverse)
      .def_readwrite("has_mask", &ir::ScanAttr::has_mask)
      .def("__eq__", [](const ir::ScanAttr& a, const ir::ScanAttr& b) { return a == b; });

  py::class_<ir::OnlineSoftmaxAttr>(ir_m, "OnlineSoftmaxAttr")
      .def(py::init<>())
      .def_readwrite("axis", &ir::OnlineSoftmaxAttr::axis)
      .def_readwrite("log_softmax", &ir::OnlineSoftmaxAttr::log_softmax)
      .def_readwrite("tg_threads", &ir::OnlineSoftmaxAttr::tg_threads)
      .def_readwrite("vec_width", &ir::OnlineSoftmaxAttr::vec_width)
      .def("__eq__", [](const ir::OnlineSoftmaxAttr& a, const ir::OnlineSoftmaxAttr& b) { return a == b; });

  py::class_<ir::WelfordNormAttr>(ir_m, "WelfordNormAttr")
      .def(py::init<>())
      .def_readwrite("kind", &ir::WelfordNormAttr::kind)
      .def_readwrite("eps", &ir::WelfordNormAttr::eps)
      .def_readwrite("axis", &ir::WelfordNormAttr::axis)
      .def_readwrite("num_groups", &ir::WelfordNormAttr::num_groups)
      .def_readwrite("stage", &ir::WelfordNormAttr::stage)
      .def_readwrite("has_weight", &ir::WelfordNormAttr::has_weight)
      .def_readwrite("has_bias", &ir::WelfordNormAttr::has_bias)
      .def_readwrite("rms_sram_cache", &ir::WelfordNormAttr::rms_sram_cache)
      .def_readwrite("tg_threads", &ir::WelfordNormAttr::tg_threads)
      .def_readwrite("vec_width", &ir::WelfordNormAttr::vec_width)
      .def("__eq__", [](const ir::WelfordNormAttr& a, const ir::WelfordNormAttr& b) { return a == b; });

  py::class_<ir::PoolAttr>(ir_m, "PoolAttr")
      .def(py::init<>())
      .def_readwrite("kind", &ir::PoolAttr::kind)
      .def_readwrite("ndim", &ir::PoolAttr::ndim)
      .def_readwrite("kernel_size", &ir::PoolAttr::kernel_size)
      .def_readwrite("stride", &ir::PoolAttr::stride)
      .def_readwrite("padding", &ir::PoolAttr::padding)
      .def_readwrite("dilation", &ir::PoolAttr::dilation)
      .def_readwrite("count_include_pad", &ir::PoolAttr::count_include_pad)
      .def_readwrite("ceil_mode", &ir::PoolAttr::ceil_mode)
      .def_readwrite("use_sram_halo", &ir::PoolAttr::use_sram_halo)
      .def_readwrite("tg_x", &ir::PoolAttr::tg_x)
      .def_readwrite("tg_y", &ir::PoolAttr::tg_y)
      .def("__eq__", [](const ir::PoolAttr& a, const ir::PoolAttr& b) { return a == b; });

  py::class_<ir::ConvAttr>(ir_m, "ConvAttr")
      .def(py::init<>())
      .def_readwrite("ndim", &ir::ConvAttr::ndim)
      .def_readwrite("kernel_size", &ir::ConvAttr::kernel_size)
      .def_readwrite("stride", &ir::ConvAttr::stride)
      .def_readwrite("padding", &ir::ConvAttr::padding)
      .def_readwrite("dilation", &ir::ConvAttr::dilation)
      .def_readwrite("output_padding", &ir::ConvAttr::output_padding)
      .def_readwrite("groups", &ir::ConvAttr::groups)
      .def_readwrite("transposed", &ir::ConvAttr::transposed)
      .def_readwrite("has_bias", &ir::ConvAttr::has_bias)
      .def_readwrite("is_depthwise", &ir::ConvAttr::is_depthwise)
      .def_readwrite("is_pointwise", &ir::ConvAttr::is_pointwise)
      .def_readwrite("use_sram_halo", &ir::ConvAttr::use_sram_halo)
      .def_readwrite("bm", &ir::ConvAttr::bm)
      .def_readwrite("bn", &ir::ConvAttr::bn)
      .def_readwrite("bk", &ir::ConvAttr::bk)
      .def_readwrite("num_simdgroups", &ir::ConvAttr::num_simdgroups)
      .def_readwrite("layout", &ir::ConvAttr::layout)
      .def("__eq__", [](const ir::ConvAttr& a, const ir::ConvAttr& b) { return a == b; });

  py::class_<ir::FlashAttnAttr>(ir_m, "FlashAttnAttr")
      .def(py::init<>())
      .def_readwrite("scale", &ir::FlashAttnAttr::scale)
      .def_readwrite("causal", &ir::FlashAttnAttr::causal)
      .def_readwrite("causal_offset", &ir::FlashAttnAttr::causal_offset)
      .def_readwrite("bm", &ir::FlashAttnAttr::bm)
      .def_readwrite("bn", &ir::FlashAttnAttr::bn)
      .def_readwrite("bd", &ir::FlashAttnAttr::bd)
      .def_readwrite("num_simdgroups", &ir::FlashAttnAttr::num_simdgroups)
      .def("__eq__", [](const ir::FlashAttnAttr& a, const ir::FlashAttnAttr& b) { return a == b; });

  py::class_<ir::LossAttr>(ir_m, "LossAttr")
      .def(py::init<>())
      .def_readwrite("kind", &ir::LossAttr::kind)
      .def_readwrite("margin", &ir::LossAttr::margin)
      .def_readwrite("beta", &ir::LossAttr::beta)
      .def_readwrite("eps", &ir::LossAttr::eps)
      .def_readwrite("stage", &ir::LossAttr::stage)
      .def_readwrite("tg_threads", &ir::LossAttr::tg_threads)
      .def("__eq__", [](const ir::LossAttr& a, const ir::LossAttr& b) { return a == b; });

  py::class_<ir::TileType>(ir_m, "TileType")
      .def(py::init<>())
      .def_readwrite("shape", &ir::TileType::shape)
      .def_readwrite("elem", &ir::TileType::elem)
      .def_readwrite("structure", &ir::TileType::structure)
      .def("__eq__", [](const ir::TileType& a, const ir::TileType& b) { return a == b; });

  py::class_<ir::Type>(ir_m, "Type")
      .def(py::init<>())
      .def_static("scalar", &ir::Type::scalar, py::arg("dt"))
      .def_static("ptr", &ir::Type::ptr, py::arg("pointee"), py::arg("space") = ir::AddrSpace::Device)
      .def_static(
          "tile",
          [](std::vector<int64_t> shape, ir::ScalarDType elem, ir::StructureAttr structure) {
            return ir::Type::tile(std::move(shape), ir::ElemType{elem}, ir::LayoutAttr{}, structure);
          },
          py::arg("shape"),
          py::arg("elem"),
          py::arg("structure") = ir::StructureAttr{})
      .def_static(
          "tile",
          [](std::vector<int64_t> shape, ir::PtrType elem, ir::StructureAttr structure) {
            return ir::Type::tile(std::move(shape), ir::ElemType{elem}, ir::LayoutAttr{}, structure);
          },
          py::arg("shape"),
          py::arg("elem"),
          py::arg("structure") = ir::StructureAttr{})
      .def("is_scalar", &ir::Type::is_scalar)
      .def("is_ptr", &ir::Type::is_ptr)
      .def("is_tile", &ir::Type::is_tile)
      .def("as_scalar", &ir::Type::as_scalar)
      .def("as_ptr", &ir::Type::as_ptr, py::return_value_policy::copy)
      .def("as_tile", &ir::Type::as_tile, py::return_value_policy::copy)
      .def("shape",
           [](const ir::Type& ty) -> std::vector<int64_t> {
             return ty.is_tile() ? ty.as_tile().shape : std::vector<int64_t>{};
           })
      .def("is_float_elem", &ir::Type::is_float_elem)
      .def("is_int_elem", &ir::Type::is_int_elem)
      .def("is_bool_elem", &ir::Type::is_bool_elem)
      .def("is_ptr_elem", &ir::Type::is_ptr_elem)
      .def("scalar_elem", &ir::Type::scalar_elem)
      .def("ptr_elem", &ir::Type::ptr_elem)
      .def("elem_bytes", &ir::Type::elem_bytes)
      .def("same_shape_and_elem", &ir::Type::same_shape_and_elem)
      .def("__eq__", [](const ir::Type& a, const ir::Type& b) { return a == b; })
      .def("__repr__", [](const ir::Type& ty) { return ir::type_to_string(ty); });

  py::class_<ir::ForLoopHandle>(ir_m, "ForLoopHandle")
      .def_readonly("op_id", &ir::ForLoopHandle::op_id)
      .def_readonly("body_block", &ir::ForLoopHandle::body_block)
      .def_readonly("iv", &ir::ForLoopHandle::iv)
      .def_readonly("iter_args", &ir::ForLoopHandle::iter_args)
      .def_readonly("results", &ir::ForLoopHandle::results);

  py::class_<ir::IfHandle>(ir_m, "IfHandle")
      .def_readonly("op_id", &ir::IfHandle::op_id)
      .def_readonly("then_block", &ir::IfHandle::then_block)
      .def_readonly("else_block", &ir::IfHandle::else_block)
      .def_readonly("results", &ir::IfHandle::results);

  py::class_<ir::KernelParam>(ir_m, "KernelParam")
      .def_readonly("name", &ir::KernelParam::name)
      .def_readonly("type", &ir::KernelParam::type)
      .def_readonly("is_constexpr", &ir::KernelParam::is_constexpr)
      .def_readonly("constexpr_val", &ir::KernelParam::constexpr_val)
      .def_readonly("divisibility", &ir::KernelParam::divisibility)
      .def_readwrite("writable", &ir::KernelParam::writable);

  py::class_<ir::Kernel>(ir_m, "Kernel")
      .def_readwrite("name", &ir::Kernel::name)
      .def_readwrite("hints", &ir::Kernel::hints)
      .def_readonly("entry_block", &ir::Kernel::entry_block)
      .def_readonly("params", &ir::Kernel::params)
      .def("value_type", [](const ir::Kernel& k, ir::ValueId id) -> ir::Type { return k.value(id).type; })
      .def("value_def_op", [](const ir::Kernel& k, ir::ValueId id) -> ir::OpId { return k.value(id).def_op; })
      .def("set_value_debug_name",
           [](ir::Kernel& k, ir::ValueId id, const std::string& name) {
             if (id < k.values.size() && k.values[id] && k.values[id]->debug_name.empty()) {
               k.values[id]->debug_name = name;
             }
           })
      .def("set_op_hints",
           [](ir::Kernel& k, ir::OpId op_id, const ir::OpHints& hints) {
             if (op_id < k.ops.size() && k.ops[op_id]) {
               k.ops[op_id]->hints = hints;
             }
           })
      .def("set_value_structure",
           [](ir::Kernel& k, ir::ValueId id, const ir::StructureAttr& s) {
             if (id < k.values.size() && k.values[id] && k.values[id]->type.is_tile()) {
               k.values[id]->type.as_tile_mut().structure = s;
             }
           })
      .def("set_param_writable",
           [](ir::Kernel& k, size_t idx, bool writable) {
             if (idx < k.params.size()) {
               k.params[idx].writable = writable;
             }
           })
      .def("block_has_terminator",
           [](const ir::Kernel& k, ir::BlockId block_id) {
             const auto& blk = k.block(block_id);
             if (blk.ops.empty()) return false;
             return ir::is_terminator_op(k.op(blk.ops.back()).kind);
           });

  py::class_<ir::Module>(ir_m, "Module")
      .def(py::init<>())
      .def("add_kernel", &ir::Module::add_kernel, py::arg("name"), py::return_value_policy::reference_internal)
      .def(
          "find_kernel",
          [](const ir::Module& mod, const std::string& name) -> const ir::Kernel* { return mod.find_kernel(name); },
          py::arg("name"),
          py::return_value_policy::reference_internal)
      .def("num_kernels", [](const ir::Module& mod) { return mod.kernels.size(); })
      .def(
          "kernel",
          [](ir::Module& mod, size_t idx) -> ir::Kernel& { return *mod.kernels.at(idx); },
          py::arg("index"),
          py::return_value_policy::reference_internal);

  py::class_<ir::IRBuilder>(ir_m, "IRBuilder")
      .def(py::init<ir::Kernel&>(), py::arg("kernel"), py::keep_alive<1, 2>())
      .def("kernel", &ir::IRBuilder::kernel, py::return_value_policy::reference_internal)
      .def("current_block", &ir::IRBuilder::current_block)
      .def("set_insertion_block", &ir::IRBuilder::set_insertion_block, py::arg("block_id"))
      .def("add_param",
           &ir::IRBuilder::add_param,
           py::arg("name"),
           py::arg("ty"),
           py::arg("is_constexpr") = false,
           py::arg("constexpr_val") = int64_t{0},
           py::arg("divisibility") = uint32_t{1},
           py::arg("writable") = false)
      .def("emit_const_i32", &ir::IRBuilder::emit_const_i32, py::arg("v"))
      .def("emit_const_u32", &ir::IRBuilder::emit_const_u32, py::arg("v"))
      .def("emit_const_i64", &ir::IRBuilder::emit_const_i64, py::arg("v"))
      .def("emit_const_f32", &ir::IRBuilder::emit_const_f32, py::arg("v"))
      .def("emit_const_f16", &ir::IRBuilder::emit_const_f16, py::arg("v"))
      .def("emit_const_bool", &ir::IRBuilder::emit_const_bool, py::arg("v"))
      .def("emit_program_id", &ir::IRBuilder::emit_program_id, py::arg("axis"))
      .def("emit_num_programs", &ir::IRBuilder::emit_num_programs, py::arg("axis"))
      .def("emit_arange", &ir::IRBuilder::emit_arange, py::arg("start"), py::arg("end"))
      .def("emit_splat", &ir::IRBuilder::emit_splat, py::arg("scalar_val"), py::arg("shape"))
      .def("emit_expand_dims", &ir::IRBuilder::emit_expand_dims, py::arg("val"), py::arg("axis"))
      .def("emit_broadcast", &ir::IRBuilder::emit_broadcast, py::arg("val"), py::arg("target_shape"))
      .def("emit_reshape", &ir::IRBuilder::emit_reshape, py::arg("val"), py::arg("target_shape"))
      .def("emit_permute", &ir::IRBuilder::emit_permute, py::arg("val"), py::arg("perm"))
      .def("emit_cast", &ir::IRBuilder::emit_cast, py::arg("val"), py::arg("target_dtype"))
      .def("emit_unary", &ir::IRBuilder::emit_unary, py::arg("op"), py::arg("val"))
      .def("emit_binary", &ir::IRBuilder::emit_binary, py::arg("op"), py::arg("lhs"), py::arg("rhs"))
      .def("emit_select", &ir::IRBuilder::emit_select, py::arg("cond"), py::arg("true_val"), py::arg("false_val"))
      .def("emit_ptr_add", &ir::IRBuilder::emit_ptr_add, py::arg("ptr"), py::arg("offset"))
      .def("emit_load",
           &ir::IRBuilder::emit_load,
           py::arg("ptrs"),
           py::arg("mask") = ir::kInvalidId,
           py::arg("other") = ir::kInvalidId,
           py::arg("attr") = ir::LoadStoreAttr{})
      .def("emit_store",
           &ir::IRBuilder::emit_store,
           py::arg("ptrs"),
           py::arg("value"),
           py::arg("mask") = ir::kInvalidId,
           py::arg("attr") = ir::LoadStoreAttr{})
      .def("emit_atomic_add",
           &ir::IRBuilder::emit_atomic_add,
           py::arg("ptrs"),
           py::arg("value"),
           py::arg("mask") = ir::kInvalidId)
      .def("emit_dot",
           &ir::IRBuilder::emit_dot,
           py::arg("lhs"),
           py::arg("rhs"),
           py::arg("acc") = ir::kInvalidId,
           py::arg("attr") = ir::DotAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_reduce", &ir::IRBuilder::emit_reduce, py::arg("input"), py::arg("attr"))
      .def("emit_scan",
           &ir::IRBuilder::emit_scan,
           py::arg("input"),
           py::arg("attr"),
           py::arg("mask") = ir::kInvalidId)
      .def("emit_online_softmax",
           &ir::IRBuilder::emit_online_softmax,
           py::arg("x_ptr"),
           py::arg("out_ptr"),
           py::arg("num_rows"),
           py::arg("row_len"),
           py::arg("attr") = ir::OnlineSoftmaxAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_welford_norm",
           &ir::IRBuilder::emit_welford_norm,
           py::arg("operands"),
           py::arg("attr") = ir::WelfordNormAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_pool",
           &ir::IRBuilder::emit_pool,
           py::arg("operands"),
           py::arg("attr") = ir::PoolAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_row_scan",
           &ir::IRBuilder::emit_row_scan,
           py::arg("x_ptr"),
           py::arg("out_ptr"),
           py::arg("num_rows"),
           py::arg("row_len"),
           py::arg("attr") = ir::ScanAttr{},
           py::arg("mask_ptr") = ir::kInvalidId,
           py::arg("hints") = ir::OpHints{})
      .def("emit_implicit_gemm_conv",
           &ir::IRBuilder::emit_implicit_gemm_conv,
           py::arg("operands"),
           py::arg("attr") = ir::ConvAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_fused_loss",
           &ir::IRBuilder::emit_fused_loss,
           py::arg("operands"),
           py::arg("attr") = ir::LossAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("emit_flash_attention",
           &ir::IRBuilder::emit_flash_attention,
           py::arg("operands"),
           py::arg("attr") = ir::FlashAttnAttr{},
           py::arg("hints") = ir::OpHints{})
      .def("begin_for_loop",
           &ir::IRBuilder::begin_for_loop,
           py::arg("lo"),
           py::arg("hi"),
           py::arg("step"),
           py::arg("iter_inits"))
      .def("end_for_loop",
           &ir::IRBuilder::end_for_loop,
           py::arg("handle"),
           py::arg("yield_vals"),
           py::arg("resume_block"))
      .def("begin_if",
           &ir::IRBuilder::begin_if,
           py::arg("cond"),
           py::arg("result_types"),
           py::arg("with_else") = true)
      .def("emit_yield", &ir::IRBuilder::emit_yield, py::arg("vals"))
      .def("emit_return", &ir::IRBuilder::emit_return);

  py::class_<ir::IRPrinter>(ir_m, "IRPrinter")
      .def_static("print", py::overload_cast<const ir::Kernel&>(&ir::IRPrinter::print), py::arg("kernel"))
      .def_static("print", py::overload_cast<const ir::Module&>(&ir::IRPrinter::print), py::arg("module"));

  py::class_<ir::VerifyResult>(ir_m, "VerifyResult")
      .def_readonly("ok", &ir::VerifyResult::ok)
      .def_readonly("errors", &ir::VerifyResult::errors)
      .def("summary", &ir::VerifyResult::summary);

  py::class_<ir::IRVerifier>(ir_m, "IRVerifier")
      .def_static("verify",
                  py::overload_cast<const ir::Kernel&, const ir::TargetSpec&, ir::VerifyStage>(
                      &ir::IRVerifier::verify),
                  py::arg("kernel"),
                  py::arg("target"),
                  py::arg("stage") = ir::VerifyStage::HighLevel)
      .def_static("verify",
                  py::overload_cast<const ir::Module&, const ir::TargetSpec&, ir::VerifyStage>(
                      &ir::IRVerifier::verify),
                  py::arg("module"),
                  py::arg("target"),
                  py::arg("stage") = ir::VerifyStage::HighLevel);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "M1Triton C++20/Metal core bindings";

  py::class_<m1triton::ir::TargetSpec>(m, "TargetSpec")
      .def_static("apple_m1_max", &m1triton::ir::TargetSpec::apple_m1_max)
      .def_readonly("name", &m1triton::ir::TargetSpec::name)
      .def_readonly("gpu_cores", &m1triton::ir::TargetSpec::gpu_cores)
      .def_readonly("simd_width", &m1triton::ir::TargetSpec::simd_width)
      .def_readonly("max_threads_per_threadgroup", &m1triton::ir::TargetSpec::max_threads_per_threadgroup)
      .def_readonly("max_threadgroup_memory_bytes", &m1triton::ir::TargetSpec::max_threadgroup_memory_bytes)
      .def_readonly("sgmm_m", &m1triton::ir::TargetSpec::sgmm_m)
      .def_readonly("sgmm_n", &m1triton::ir::TargetSpec::sgmm_n)
      .def_readonly("sgmm_k", &m1triton::ir::TargetSpec::sgmm_k)
      .def_readonly("vector_bytes", &m1triton::ir::TargetSpec::vector_bytes)
      .def_readonly("peak_bandwidth_gbs", &m1triton::ir::TargetSpec::peak_bandwidth_gbs)
      .def_readonly("peak_fp32_tflops", &m1triton::ir::TargetSpec::peak_fp32_tflops)
      .def_readonly("peak_fp16_tflops", &m1triton::ir::TargetSpec::peak_fp16_tflops)
      .def("fits_threadgroup_memory", &m1triton::ir::TargetSpec::fits_threadgroup_memory);

  bind_ir_submodule(m);

  py::class_<m1triton::codegen::metal::ParamSlot>(m, "ParamSlot")
      .def(py::init<>())
      .def_readonly("param_index", &m1triton::codegen::metal::ParamSlot::param_index)
      .def_readonly("buffer_index", &m1triton::codegen::metal::ParamSlot::buffer_index)
      .def_readonly("name", &m1triton::codegen::metal::ParamSlot::name)
      .def_readonly("is_pointer", &m1triton::codegen::metal::ParamSlot::is_pointer)
      .def_readonly("writable", &m1triton::codegen::metal::ParamSlot::writable)
      .def_readonly("dtype", &m1triton::codegen::metal::ParamSlot::dtype);

  py::class_<m1triton::codegen::metal::LaunchSpec>(m, "LaunchSpec")
      .def(py::init<>())
      .def_readonly("entry_name", &m1triton::codegen::metal::LaunchSpec::entry_name)
      .def_readonly("tg_x", &m1triton::codegen::metal::LaunchSpec::tg_x)
      .def_readonly("tg_y", &m1triton::codegen::metal::LaunchSpec::tg_y)
      .def_readonly("tg_z", &m1triton::codegen::metal::LaunchSpec::tg_z)
      .def_readonly("threadgroup_mem_bytes", &m1triton::codegen::metal::LaunchSpec::threadgroup_mem_bytes)
      .def_readonly("vec_width", &m1triton::codegen::metal::LaunchSpec::vec_width)
      .def_readonly("fast_math", &m1triton::codegen::metal::LaunchSpec::fast_math)
      .def_readonly("param_slots", &m1triton::codegen::metal::LaunchSpec::param_slots);

  py::class_<m1triton::codegen::metal::CompiledMSL>(m, "CompiledMSL")
      .def(py::init<>())
      .def_readonly("ok", &m1triton::codegen::metal::CompiledMSL::ok)
      .def_readonly("msl_source", &m1triton::codegen::metal::CompiledMSL::msl_source)
      .def_readonly("launch_spec", &m1triton::codegen::metal::CompiledMSL::launch_spec)
      .def_readonly("error", &m1triton::codegen::metal::CompiledMSL::error);

  m.def(
      "compile_kernel_ir",
      [](m1triton::ir::Module& mod,
         const m1triton::ir::TargetSpec& target,
         bool verify_ir,
         bool dump_ir,
         bool dump_msl) {
        if (mod.kernels.empty()) {
          throw std::runtime_error("Cannot compile empty M1Triton IR Module");
        }
        std::string pre_ir_text = m1triton::ir::IRPrinter::print(mod);
        if (verify_ir) {
          auto pre_vr = m1triton::ir::IRVerifier::verify(mod, target, m1triton::ir::VerifyStage::HighLevel);
          if (!pre_vr.ok) {
            throw std::runtime_error("Pre-pass IR verification failed:\n" + pre_vr.summary());
          }
        }

        auto pm = m1triton::passes::PassManager::build_m2_pipeline();
        std::ostringstream pass_dump;
        m1triton::passes::PassContext ctx{target, verify_ir, dump_ir, dump_ir ? &pass_dump : nullptr};
        auto pass_res = pm.run(mod, ctx);
        if (!pass_res.ok) {
          throw std::runtime_error("M1 optimization pipeline failed: " + pass_res.diagnostic);
        }

        std::string post_ir_text = m1triton::ir::IRPrinter::print(mod);
        if (dump_ir) {
          std::ostringstream os;
          os << "=== M1Triton Pre-Pass IR ===\n" << pre_ir_text << "\n";
          if (!pass_dump.str().empty()) {
            os << "=== M1Triton Pass Trace ===\n" << pass_dump.str() << "\n";
          }
          os << "=== M1Triton Post-Pass IR ===\n" << post_ir_text << "\n";
          write_py_stderr(os.str());
        }

        auto compiled = m1triton::codegen::metal::MetalCodegen::emit(*mod.kernels[0], target);
        if (dump_msl && compiled.ok) {
          std::ostringstream os;
          os << "=== M1Triton Generated MSL (" << compiled.launch_spec.entry_name << ") ===\n"
             << compiled.msl_source << "\n";
          write_py_stderr(os.str());
        }
        return std::make_tuple(std::move(compiled), std::move(pre_ir_text), std::move(post_ir_text));
      },
      py::arg("module"),
      py::arg("target"),
      py::arg("verify_ir") = true,
      py::arg("dump_ir") = false,
      py::arg("dump_msl") = false);

  py::class_<m1triton::runtime::DeviceInfo>(m, "DeviceInfo")
      .def_readonly("name", &m1triton::runtime::DeviceInfo::name)
      .def_readonly("max_threads_per_threadgroup", &m1triton::runtime::DeviceInfo::max_threads_per_threadgroup)
      .def_readonly("max_threadgroup_memory_bytes", &m1triton::runtime::DeviceInfo::max_threadgroup_memory_bytes)
      .def_readonly("recommended_max_working_set_bytes",
                    &m1triton::runtime::DeviceInfo::recommended_max_working_set_bytes)
      .def_readonly("has_unified_memory", &m1triton::runtime::DeviceInfo::has_unified_memory)
      .def_readonly("supports_metal3", &m1triton::runtime::DeviceInfo::supports_metal3)
      .def_readonly("supports_indirect_command_buffers",
                    &m1triton::runtime::DeviceInfo::supports_indirect_command_buffers)
      .def_readonly("available", &m1triton::runtime::DeviceInfo::available);

  py::class_<m1triton::runtime::BufferArg>(m, "BufferArg")
      .def(py::init<>())
      .def_readwrite("handle", &m1triton::runtime::BufferArg::handle)
      .def_readwrite("byte_offset", &m1triton::runtime::BufferArg::byte_offset)
      .def_readwrite("nbytes", &m1triton::runtime::BufferArg::nbytes)
      .def_readwrite("index", &m1triton::runtime::BufferArg::index)
      .def_readwrite("is_metal_buffer", &m1triton::runtime::BufferArg::is_metal_buffer)
      .def_readwrite("writable", &m1triton::runtime::BufferArg::writable);

  py::class_<m1triton::runtime::ScalarArg>(m, "ScalarArg")
      .def(py::init<>())
      .def_readwrite("index", &m1triton::runtime::ScalarArg::index)
      .def_readwrite("bytes", &m1triton::runtime::ScalarArg::bytes);

  py::class_<m1triton::runtime::LaunchConfig>(m, "LaunchConfig")
      .def(py::init<>())
      .def_readwrite("grid_x", &m1triton::runtime::LaunchConfig::grid_x)
      .def_readwrite("grid_y", &m1triton::runtime::LaunchConfig::grid_y)
      .def_readwrite("grid_z", &m1triton::runtime::LaunchConfig::grid_z)
      .def_readwrite("tg_x", &m1triton::runtime::LaunchConfig::tg_x)
      .def_readwrite("tg_y", &m1triton::runtime::LaunchConfig::tg_y)
      .def_readwrite("tg_z", &m1triton::runtime::LaunchConfig::tg_z)
      .def_readwrite("threadgroup_mem_bytes", &m1triton::runtime::LaunchConfig::threadgroup_mem_bytes)
      .def_readwrite("wait", &m1triton::runtime::LaunchConfig::wait);

  m.def("device_info", []() { return m1triton::runtime::MetalRuntime::instance().device_info(); });
  m.def("get_dispatch_count", []() { return m1triton::runtime::MetalRuntime::instance().dispatch_count(); });
  m.def("reset_dispatch_count", []() { m1triton::runtime::MetalRuntime::instance().reset_dispatch_count(); });
  m.def("get_last_kernel_name", []() { return m1triton::runtime::MetalRuntime::instance().last_kernel_name(); });
  m.def("synchronize", []() { m1triton::runtime::MetalRuntime::instance().synchronize(); });

  m.def(
      "compile_kernel",
      [](const std::string& cache_key, const std::string& src, const std::string& fn, bool fast_math) {
        std::string err;
        uint64_t kid = m1triton::runtime::MetalRuntime::instance().compile_kernel(
            cache_key, src, fn, fast_math, &err);
        if (kid == 0) {
          throw std::runtime_error(err);
        }
        return kid;
      },
      py::arg("cache_key"),
      py::arg("msl_source"),
      py::arg("fn_name"),
      py::arg("fast_math") = false);

  m.def(
      "launch_kernel",
      [](uint64_t kid,
         const std::vector<m1triton::runtime::BufferArg>& buffers,
         const std::vector<m1triton::runtime::ScalarArg>& scalars,
         const m1triton::runtime::LaunchConfig& cfg) {
        std::string err;
        double gpu_ms = m1triton::runtime::MetalRuntime::instance().launch_kernel(
            kid, buffers, scalars, cfg, &err);
        if (gpu_ms < 0.0) {
          throw std::runtime_error(err);
        }
        return gpu_ms;
      });

  m.def(
      "kernel_max_threads_per_threadgroup",
      [](uint64_t kernel_id) {
        return m1triton::runtime::MetalRuntime::instance().kernel_max_threads_per_threadgroup(kernel_id);
      },
      py::arg("kernel_id"));

  m.def(
      "kernel_supports_icb",
      [](uint64_t kernel_id) {
        return m1triton::runtime::MetalRuntime::instance().kernel_supports_icb(kernel_id);
      },
      py::arg("kernel_id"));

  m.def("begin_batch", []() {
    std::string err;
    if (!m1triton::runtime::MetalRuntime::instance().begin_batch(&err)) {
      throw std::runtime_error(err);
    }
  });

  m.def(
      "end_batch",
      [](bool wait) {
        std::string err;
        double gpu_ms = m1triton::runtime::MetalRuntime::instance().end_batch(wait, &err);
        if (gpu_ms < 0.0) {
          throw std::runtime_error(err);
        }
        return gpu_ms;
      },
      py::arg("wait") = true);

  m.def("abort_batch", []() { m1triton::runtime::MetalRuntime::instance().abort_batch(); });
  m.def("in_batch", []() { return m1triton::runtime::MetalRuntime::instance().in_batch(); });

  m.def("begin_graph_capture", []() {
    std::string err;
    if (!m1triton::runtime::MetalRuntime::instance().begin_graph_capture(&err)) {
      throw std::runtime_error(err);
    }
  });

  m.def("end_graph_capture", []() {
    std::string err;
    uint64_t gid = m1triton::runtime::MetalRuntime::instance().end_graph_capture(&err);
    if (gid == 0) {
      throw std::runtime_error(err);
    }
    return gid;
  });

  m.def("abort_graph_capture", []() { m1triton::runtime::MetalRuntime::instance().abort_graph_capture(); });
  m.def("in_graph_capture", []() { return m1triton::runtime::MetalRuntime::instance().in_graph_capture(); });

  m.def(
      "replay_graph",
      [](uint64_t graph_id, bool wait) {
        std::string err;
        double gpu_ms = m1triton::runtime::MetalRuntime::instance().replay_graph(graph_id, wait, &err);
        if (gpu_ms < 0.0) {
          throw std::runtime_error(err);
        }
        return gpu_ms;
      },
      py::arg("graph_id"),
      py::arg("wait") = true);

  m.def(
      "destroy_graph",
      [](uint64_t graph_id) { m1triton::runtime::MetalRuntime::instance().destroy_graph(graph_id); },
      py::arg("graph_id"));

  m.def(
      "graph_command_count",
      [](uint64_t graph_id) { return m1triton::runtime::MetalRuntime::instance().graph_command_count(graph_id); },
      py::arg("graph_id"));

  m.def("compile_msl_smoke", [](const std::string& src, const std::string& fn) {
    std::string err;
    bool ok = m1triton::runtime::MetalRuntime::instance().compile_msl_smoke(src, fn, &err);
    if (!ok) {
      throw std::runtime_error(err);
    }
    return true;
  });
}
