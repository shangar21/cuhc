cmake_minimum_required(VERSION 3.24)

# Must be set before project(), which otherwise fills in nvcc's default (sm_75).
# "native" compiles real machine code for the GPUs in this computer. That matters:
# a driver can refuse to JIT PTX from a newer nvcc ("the provided PTX was compiled
# with an unsupported toolchain"). To build for other GPUs (e.g. without a GPU, or
# for a wheel), pass -DCMAKE_CUDA_ARCHITECTURES="80;86;89", or with pip:
#   CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;86;89" pip install .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuHierarchicalClustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# The benchmark is for development only; pip builds (SKBUILD) skip it.
if(NOT SKBUILD)
  add_executable(bench_distance bench/bench_distance.cu)
  target_include_directories(bench_distance PRIVATE src)
  target_link_libraries(bench_distance PRIVATE CUDA::cublas)
  # -lineinfo lets Nsight Compute map stalls back to source lines.
  target_compile_options(bench_distance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# Shared library behind the Python package (loaded with ctypes). A development
# build writes it into python/cuhc/, so the package works straight from the
# source tree; a pip build installs it into the package instead (below).
add_library(cuhc SHARED src/capi.cu)
target_include_directories(cuhc PRIVATE src)
set_target_properties(cuhc PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT SKBUILD)
  set_target_properties(cuhc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/cuhc)
endif()
target_compile_options(cuhc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# pip install (scikit-build-core): put the library into the installed cuhc package.
install(TARGETS cuhc LIBRARY DESTINATION cuhc)
