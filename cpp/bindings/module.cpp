#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "ir/target_spec.hpp"
#include "runtime/metal_runtime.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "M1Triton C++20/Metal core bindings";

  py::class_<m1triton::ir::TargetSpec>(m, "TargetSpec")
      .def_static("apple_m1_max", &m1triton::ir::TargetSpec::apple_m1_max)
      .def_readonly("name", &m1triton::ir::TargetSpec::name)
      .def_readonly("gpu_cores", &m1triton::ir::TargetSpec::gpu_cores)
      .def_readonly("simd_width", &m1triton::ir::TargetSpec::simd_width)
      .def_readonly("max_threads_per_threadgroup", &m1triton::ir::TargetSpec::max_threads_per_threadgroup)
      .def_readonly("max_threadgroup_memory_bytes", &m1triton::ir::TargetSpec::max_threadgroup_memory_bytes)
      .def_readonly("peak_bandwidth_gbs", &m1triton::ir::TargetSpec::peak_bandwidth_gbs)
      .def_readonly("peak_fp32_tflops", &m1triton::ir::TargetSpec::peak_fp32_tflops)
      .def_readonly("peak_fp16_tflops", &m1triton::ir::TargetSpec::peak_fp16_tflops)
      .def("fits_threadgroup_memory", &m1triton::ir::TargetSpec::fits_threadgroup_memory);

  py::class_<m1triton::runtime::DeviceInfo>(m, "DeviceInfo")
      .def_readonly("name", &m1triton::runtime::DeviceInfo::name)
      .def_readonly("max_threads_per_threadgroup", &m1triton::runtime::DeviceInfo::max_threads_per_threadgroup)
      .def_readonly("max_threadgroup_memory_bytes", &m1triton::runtime::DeviceInfo::max_threadgroup_memory_bytes)
      .def_readonly("recommended_max_working_set_bytes",
                    &m1triton::runtime::DeviceInfo::recommended_max_working_set_bytes)
      .def_readonly("has_unified_memory", &m1triton::runtime::DeviceInfo::has_unified_memory)
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

  m.def("compile_msl_smoke", [](const std::string& src, const std::string& fn) {
    std::string err;
    bool ok = m1triton::runtime::MetalRuntime::instance().compile_msl_smoke(src, fn, &err);
    if (!ok) {
      throw std::runtime_error(err);
    }
    return true;
  });
}
