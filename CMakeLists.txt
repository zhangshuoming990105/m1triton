cmake_minimum_required(VERSION 3.24)
project(m1triton LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(m1triton_ir STATIC
  cpp/ir/attrs.cpp
  cpp/ir/builder.cpp
  cpp/ir/ir.cpp
  cpp/ir/parser.cpp
  cpp/ir/printer.cpp
  cpp/ir/target_spec.cpp
  cpp/ir/types.cpp
  cpp/ir/verifier.cpp
)
target_include_directories(m1triton_ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(m1triton_ir PRIVATE -Wall -Wextra -Werror)

add_library(m1triton_passes STATIC
  cpp/passes/canonicalize.cpp
  cpp/passes/contiguity_analysis.cpp
  cpp/passes/cse.cpp
  cpp/passes/dce.cpp
  cpp/passes/licm.cpp
  cpp/passes/pass_manager.cpp
  cpp/passes/tile_layout.cpp
)
target_include_directories(m1triton_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(m1triton_passes PUBLIC m1triton_ir)
target_compile_options(m1triton_passes PRIVATE -Wall -Wextra -Werror)

add_library(m1triton_codegen_metal STATIC
  cpp/codegen/metal/metal_codegen.cpp
)
target_include_directories(m1triton_codegen_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(m1triton_codegen_metal PUBLIC m1triton_ir)
target_compile_options(m1triton_codegen_metal PRIVATE -Wall -Wextra -Werror)

add_library(m1triton_runtime STATIC
  cpp/runtime/metal_runtime.mm
)
target_include_directories(m1triton_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(m1triton_runtime PRIVATE -fobjc-arc -Wall -Wextra -Werror)
target_link_libraries(m1triton_runtime PUBLIC
  m1triton_ir
  "-framework Metal"
  "-framework Foundation"
)

pybind11_add_module(_C MODULE cpp/bindings/module.cpp)
target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(_C PRIVATE
  m1triton_runtime
  m1triton_codegen_metal
  m1triton_passes
  m1triton_ir
)
set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/m1triton"
)

add_executable(m1triton-opt tools/m1triton_opt.cpp)
target_link_libraries(m1triton-opt PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(m1triton-opt PRIVATE -Wall -Wextra -Werror)

add_executable(test_target_spec tests/cpp/test_target_spec.cpp)
target_link_libraries(test_target_spec PRIVATE m1triton_ir)
target_compile_options(test_target_spec PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_target_spec COMMAND test_target_spec)

add_executable(test_ir_and_passes tests/cpp/test_ir_and_passes.cpp)
target_link_libraries(test_ir_and_passes PRIVATE m1triton_passes m1triton_ir)
target_compile_options(test_ir_and_passes PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_ir_and_passes COMMAND test_ir_and_passes)

add_executable(test_metal_codegen tests/cpp/test_metal_codegen.cpp)
target_link_libraries(test_metal_codegen PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_metal_codegen PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_metal_codegen COMMAND test_metal_codegen)
