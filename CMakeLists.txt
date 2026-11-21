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
  cpp/ir/target_spec.cpp
)
target_include_directories(m1triton_ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(m1triton_ir PRIVATE -Wall -Wextra -Werror)

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
target_link_libraries(_C PRIVATE m1triton_runtime m1triton_ir)
set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/m1triton"
)

add_executable(test_target_spec tests/cpp/test_target_spec.cpp)
target_link_libraries(test_target_spec PRIVATE m1triton_ir)
target_compile_options(test_target_spec PRIVATE -Wall -Wextra -Werror)
add_test(NAME test_target_spec COMMAND test_target_spec)
