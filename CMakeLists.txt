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
  cpp/passes/conv_lowering.cpp
  cpp/passes/cse.cpp
  cpp/passes/dce.cpp
  cpp/passes/epilogue_fusion.cpp
  cpp/passes/flash_attn_lowering.cpp
  cpp/passes/licm.cpp
  cpp/passes/loss_lowering.cpp
  cpp/passes/online_softmax_welford.cpp
  cpp/passes/pass_manager.cpp
  cpp/passes/pool_lowering.cpp
  cpp/passes/simdgroup_matmul.cpp
  cpp/passes/structure_pruning.cpp
  cpp/passes/threadgroup_staging.cpp
  cpp/passes/tile_layout.cpp
)
target_include_directories(m1triton_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(m1triton_passes PUBLIC m1triton_ir)
target_compile_options(m1triton_passes PRIVATE -Wall -Wextra -Werror)

add_library(m1triton_codegen_metal STATIC
  cpp/codegen/metal/conv_dw_emitter.cpp
  cpp/codegen/metal/conv_std_emitter.cpp
  cpp/codegen/metal/conv_transpose_emitter.cpp
  cpp/codegen/metal/flash_attn_emitter.cpp
  cpp/codegen/metal/loss_emitter.cpp
  cpp/codegen/metal/metal_codegen.cpp
  cpp/codegen/metal/msl_common.cpp
  cpp/codegen/metal/norm_softmax_emitter.cpp
  cpp/codegen/metal/pool_emitter.cpp
  cpp/codegen/metal/reduce_scan_emitter.cpp
  cpp/codegen/metal/row_scan_emitter.cpp
  cpp/codegen/metal/sgmm_emitter.cpp
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

add_executable(test_m2_passes tests/cpp/test_m2_passes.cpp)
target_link_libraries(test_m2_passes PRIVATE m1triton_passes m1triton_ir)
target_compile_options(test_m2_passes PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_m2_passes COMMAND test_m2_passes)

add_executable(test_metal_codegen tests/cpp/test_metal_codegen.cpp)
target_link_libraries(test_metal_codegen PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_metal_codegen PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_metal_codegen COMMAND test_metal_codegen)

add_executable(test_sgmm_emitter tests/cpp/test_sgmm_emitter.cpp)
target_link_libraries(test_sgmm_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_sgmm_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_sgmm_emitter COMMAND test_sgmm_emitter)

add_executable(test_reduce_scan_emitter tests/cpp/test_reduce_scan_emitter.cpp)
target_link_libraries(test_reduce_scan_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_reduce_scan_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_reduce_scan_emitter COMMAND test_reduce_scan_emitter)

add_executable(test_m3_epilogue tests/cpp/test_m3_epilogue.cpp)
target_link_libraries(test_m3_epilogue PRIVATE m1triton_passes m1triton_ir)
target_compile_options(test_m3_epilogue PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_m3_epilogue COMMAND test_m3_epilogue)

add_executable(test_norm_softmax_emitter tests/cpp/test_norm_softmax_emitter.cpp)
target_link_libraries(test_norm_softmax_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_norm_softmax_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_norm_softmax_emitter COMMAND test_norm_softmax_emitter)

add_executable(test_pool_emitter tests/cpp/test_pool_emitter.cpp)
target_link_libraries(test_pool_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_pool_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_pool_emitter COMMAND test_pool_emitter)

add_executable(test_row_scan_emitter tests/cpp/test_row_scan_emitter.cpp)
target_link_libraries(test_row_scan_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_row_scan_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_row_scan_emitter COMMAND test_row_scan_emitter)

add_executable(test_conv_std_emitter tests/cpp/test_conv_std_emitter.cpp)
target_link_libraries(test_conv_std_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_conv_std_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_conv_std_emitter COMMAND test_conv_std_emitter)

add_executable(test_conv_transpose_emitter tests/cpp/test_conv_transpose_emitter.cpp)
target_link_libraries(test_conv_transpose_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_conv_transpose_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_conv_transpose_emitter COMMAND test_conv_transpose_emitter)

add_executable(test_conv_dw_emitter tests/cpp/test_conv_dw_emitter.cpp)
target_link_libraries(test_conv_dw_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_conv_dw_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_conv_dw_emitter COMMAND test_conv_dw_emitter)

add_executable(test_loss_emitter tests/cpp/test_loss_emitter.cpp)
target_link_libraries(test_loss_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_loss_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_loss_emitter COMMAND test_loss_emitter)

add_executable(test_flash_attn_emitter tests/cpp/test_flash_attn_emitter.cpp)
target_link_libraries(test_flash_attn_emitter PRIVATE m1triton_codegen_metal m1triton_passes m1triton_ir)
target_compile_options(test_flash_attn_emitter PRIVATE -UNDEBUG -Wall -Wextra -Werror)
add_test(NAME test_flash_attn_emitter COMMAND test_flash_attn_emitter)
