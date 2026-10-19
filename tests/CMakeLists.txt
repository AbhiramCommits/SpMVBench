# GoogleTest suite for SpMVBench.
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: prevent overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(spmvbench_tests
  test_mtx_io.cpp
  test_formats.cpp
  test_stats.cpp
  test_spmv_correctness.cpp
  test_spmm_correctness.cpp
  test_select.cpp
  test_cuda_correctness.cpp
)

target_link_libraries(spmvbench_tests PRIVATE spmvbench_core GTest::gtest_main)
target_compile_definitions(spmvbench_tests PRIVATE
  SPMVBENCH_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(SPMVBENCH_CUDA_ON)
  # The host-compiled test file includes the CUDA headers and calls into the
  # explicitly instantiated SpmvRunner/SpMM symbols from spmvbench_core.
  target_link_libraries(spmvbench_tests PRIVATE CUDA::cudart CUDA::cusparse)
  target_compile_definitions(spmvbench_tests PRIVATE SPMVBENCH_ENABLE_CUDA=1)
  set_target_properties(spmvbench_tests PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

include(GoogleTest)
gtest_discover_tests(spmvbench_tests DISCOVERY_TIMEOUT 60)
