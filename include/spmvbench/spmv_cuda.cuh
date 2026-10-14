#pragma once

// CUDA SpMV interface (compiled only when SPMVBENCH_ENABLE_CUDA is defined).
//
// This header is safe to include from plain C++ translation units: when CUDA
// is disabled it expands to nothing. It declares the POD device matrix views,
// the host-side launch/benchmark wrapper `SpmvRunner`, and the cuSPARSE
// baseline entry points. Kernel bodies live in src/cuda/*.cu.

#ifdef SPMVBENCH_ENABLE_CUDA

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/select.hpp"

namespace spmvbench {

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                 \
  do {                                                                   \
    cudaError_t _e = (expr);                                             \
    if (_e != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(_e));                              \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

#define CUSPARSE_CHECK(expr)                                                 \
  do {                                                                       \
    cusparseStatus_t _s = (expr);                                            \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "cuSPARSE error %s:%d: %d\n", __FILE__, __LINE__, \
                   static_cast<int>(_s));                                    \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Device-side matrix views (flat pointers, trivially copyable)
// ---------------------------------------------------------------------------
template <typename T>
struct DeviceCsr {
  const Index* rowptr = nullptr;
  const Index* colind = nullptr;
  const T* values = nullptr;
  int m = 0;
  int n = 0;
  std::int64_t nnz = 0;
};

template <typename T>
struct DeviceEll {
  const Index* colind = nullptr;  // column-major, size m*max_row_nnz
  const T* values = nullptr;
  int m = 0;
  int n = 0;
  int max_row_nnz = 0;
};

template <typename T>
struct DeviceBcsr {
  const Index* rowptr = nullptr;
  const Index* colind = nullptr;
  const T* values = nullptr;
  int m = 0;
  int n = 0;
  int R = 1;
  int C = 1;
  int mb = 0;
  std::int64_t block_nnz = 0;
};

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------
struct SpmvResult {
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  int iters = 0;
  std::string kernel;
};

// ---------------------------------------------------------------------------
// SpmvRunner: owns device buffers for one CSR matrix, uploads x once, and can
// time any of the implemented kernels.
// ---------------------------------------------------------------------------
template <typename T, typename Acc>
class SpmvRunner {
 public:
  explicit SpmvRunner(const CsrMatrix<T>& A);
  ~SpmvRunner();

  SpmvRunner(const SpmvRunner&) = delete;
  SpmvRunner& operator=(const SpmvRunner&) = delete;

  // Upload the input vector (length n) to the device.
  void set_x(const std::vector<T>& x);

  // Run `iters` timed iterations of kernel `id` (plus `warmup` untimed ones)
  // and return timing. `lanes` applies to CsrVector, `block` to Bcsr.
  SpmvResult run(KernelId id, int iters, int lanes = 4, int block = 4,
                 int warmup = 5);

  // Copy the last result vector back to the host.
  std::vector<Acc> result() const;

 private:
  void upload_matrix();
  void free_buffers();

  const CsrMatrix<T> A_;
  DeviceCsr<T> d_csr_{};
  DeviceEll<T> d_ell_{};
  DeviceBcsr<T> d_bcsr2_{};  // 2x2
  DeviceBcsr<T> d_bcsr4_{};  // 4x4
  T* d_values_ = nullptr;
  Index* d_colind_ = nullptr;
  Index* d_rowptr_ = nullptr;
  T* d_ell_values_ = nullptr;
  Index* d_ell_colind_ = nullptr;
  T* d_b2_values_ = nullptr;
  Index* d_b2_colind_ = nullptr;
  Index* d_b2_rowptr_ = nullptr;
  T* d_b4_values_ = nullptr;
  Index* d_b4_colind_ = nullptr;
  Index* d_b4_rowptr_ = nullptr;
  T* d_x_ = nullptr;
  Acc* d_y_ = nullptr;
  int* d_counter_ = nullptr;
};

// ---------------------------------------------------------------------------
// cuSPARSE baselines
// ---------------------------------------------------------------------------
// y = A * x using cusparseSpMV with a CSR descriptor and ALG_DEFAULT.
SpmvResult cusparse_spmv_csr_f32(const CsrMatrix<float>& A,
                                 const std::vector<float>& x,
                                 std::vector<float>& y, int iters, int warmup);

SpmvResult cusparse_spmv_csr_f64(const CsrMatrix<double>& A,
                                 const std::vector<double>& x,
                                 std::vector<double>& y, int iters, int warmup);

}  // namespace spmvbench

#endif  // SPMVBENCH_ENABLE_CUDA
