cmake_minimum_required(VERSION 3.18)

project(cuda_playground LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add_bench
    src/vector_add_bench.cu
    src/vector_add.cu
    src/cuda_utils.cu
)

target_include_directories(vector_add_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(vector_add_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(matrix_multiply_bench
    src/matrix_multiply_bench.cu
    src/matrix_multiply.cu
    src/cuda_utils.cu
)

target_include_directories(matrix_multiply_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(matrix_multiply_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(daxpy_bench
    src/daxpy_bench.cu
    src/daxpy.cu
    src/cuda_utils.cu
)

target_include_directories(daxpy_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(daxpy_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(mc_option_pricing_bench
    src/mc_option_pricing_bench.cu
    src/mc_option_pricing.cu
    src/cuda_utils.cu
)

target_include_directories(mc_option_pricing_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mc_option_pricing_bench PRIVATE CUDA::curand)

set_target_properties(mc_option_pricing_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)