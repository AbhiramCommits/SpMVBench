cmake_minimum_required(VERSION 3.20)
project(SpMVBench LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPMVBENCH_ENABLE_CUDA "Build the CUDA kernel path" ON)
option(SPMVBENCH_BUILD_TESTS "Build the GoogleTest suite" ON)
option(SPMVBENCH_BUILD_BENCH "Build the bench executable" ON)

message(STATUS "SpMVBench: build type = ${CMAKE_BUILD_TYPE}")

# ---------------------------------------------------------------------------
# CUDA detection (auto)
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

set(SPMVBENCH_CUDA_ON OFF)
if(SPMVBENCH_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(SPMVBENCH_CUDA_ON ON)
  message(STATUS "SpMVBench: CUDA enabled with ${CMAKE_CUDA_COMPILER} "
                 "(arch=${CMAKE_CUDA_ARCHITECTURES})")
elseif(SPMVBENCH_ENABLE_CUDA)
  message(STATUS "SpMVBench: no CUDA compiler found -> CPU/OpenMP path only")
else()
  message(STATUS "SpMVBench: CUDA disabled by option -> CPU/OpenMP path only")
endif()

# ---------------------------------------------------------------------------
# OpenMP (optional but preferred)
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)

add_library(spmvbench_omp INTERFACE)
set(SPMVBENCH_OPENMP OFF)

if(OpenMP_CXX_FOUND)
  target_link_libraries(spmvbench_omp INTERFACE OpenMP::OpenMP_CXX)
  set(SPMVBENCH_OPENMP ON)
  message(STATUS "SpMVBench: OpenMP enabled via find_package (${OpenMP_CXX_FLAGS})")
elseif(APPLE)
  # Apple's clang has no OpenMP runtime built in; fall back to Homebrew libomp.
  find_path(SPMVBENCH_LIBOMP_INCLUDE omp.h
    HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
  find_library(SPMVBENCH_LIBOMP_LIB omp
    HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
  if(SPMVBENCH_LIBOMP_INCLUDE AND SPMVBENCH_LIBOMP_LIB)
    target_include_directories(spmvbench_omp INTERFACE ${SPMVBENCH_LIBOMP_INCLUDE})
    target_compile_options(spmvbench_omp INTERFACE -Xpreprocessor -fopenmp)
    target_link_libraries(spmvbench_omp INTERFACE ${SPMVBENCH_LIBOMP_LIB})
    set(SPMVBENCH_OPENMP ON)
    message(STATUS "SpMVBench: OpenMP enabled via Homebrew libomp "
                   "(${SPMVBENCH_LIBOMP_LIB})")
  else()
    message(STATUS "SpMVBench: OpenMP not found (libomp missing) -> serial fallback")
  endif()
else()
  message(STATUS "SpMVBench: OpenMP not found -> serial fallback")
endif()

if(SPMVBENCH_OPENMP)
  # Also give the compile-time definition used by host code.
  target_compile_definitions(spmvbench_omp INTERFACE SPMVBENCH_HAS_OPENMP=1)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(SPMVBENCH_CORE_SOURCES
  src/mtx_io.cpp
  src/formats.cpp
  src/stats.cpp
  src/cpu/spmv_cpu.cpp
  src/cpu/spmm_cpu.cpp
)

# Optional sources are folded in when present, which keeps the build working
# while the tree is assembled and is a no-op for a complete checkout.
foreach(_opt src/select.cpp src/validate.cpp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_opt})
    list(APPEND SPMVBENCH_CORE_SOURCES ${_opt})
  endif()
endforeach()

if(SPMVBENCH_CUDA_ON)
  foreach(_cu
      src/cuda/spmv_cuda.cu
      src/cuda/spmm_cuda.cu
      src/cuda/cusparse_baseline.cu
      src/cuda/cusparse_spmm.cu)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_cu})
      list(APPEND SPMVBENCH_CORE_SOURCES ${_cu})
    endif()
  endforeach()
endif()

add_library(spmvbench_core ${SPMVBENCH_CORE_SOURCES})
target_include_directories(spmvbench_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(spmvbench_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>)

if(SPMVBENCH_CUDA_ON)
  target_compile_definitions(spmvbench_core PUBLIC SPMVBENCH_ENABLE_CUDA=1)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(spmvbench_core PRIVATE CUDA::cusparse CUDA::cudart)
  set_target_properties(spmvbench_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(spmvbench_core PUBLIC spmvbench_omp)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
if(SPMVBENCH_BUILD_BENCH)
  add_executable(bench bench/bench_main.cpp)
  target_link_libraries(bench PRIVATE spmvbench_core)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SPMVBENCH_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  enable_testing()
  add_subdirectory(tests)
endif()
