#pragma once

// CUDA SpMM interface (compiled only when SPMVBENCH_ENABLE_CUDA is defined).
//
// spmm_csr_tiled: sparse CSR (m x n) times dense row-major B (n x k) -> Y
// (m x k), with shared-memory tiling over the k dimension and one warp per
// sparse row. A load-balanced variant bins rows into short/medium/long
// buckets and launches a separate grid per bucket.

#ifdef SPMVBENCH_ENABLE_CUDA

#include <cstdint>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/spmv_cuda.cuh"

namespace spmvbench {

struct SpmmResult {
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double median_ms = 0.0;
  double max_ms = 0.0;
  // Host<->device traffic (H2D B + D2H Y) timed separately from the kernel.
  double transfer_ms = 0.0;
  int iters = 0;
  std::string kernel;
};

// Time the tiled SpMM kernel. `k` is the dense width and `tile` the shared
// memory tile width (16/32/64). `balanced` selects the bucketed variant.
SpmmResult spmm_csr_tiled_f32(const CsrMatrix<float>& A,
                              const std::vector<float>& B, int k, int tile,
                              bool balanced, int iters, int warmup,
                              std::vector<float>& Y);
SpmmResult spmm_csr_tiled_f64(const CsrMatrix<double>& A,
                              const std::vector<double>& B, int k, int tile,
                              bool balanced, int iters, int warmup,
                              std::vector<double>& Y);

// cuSPARSE SpMM baseline (CSR x dense, ALG_DEFAULT).
SpmmResult cusparse_spmm_f32(const CsrMatrix<float>& A,
                             const std::vector<float>& B, int k, int iters,
                             int warmup, std::vector<float>& Y);
SpmmResult cusparse_spmm_f64(const CsrMatrix<double>& A,
                             const std::vector<double>& B, int k, int iters,
                             int warmup, std::vector<double>& Y);

}  // namespace spmvbench

#endif  // SPMVBENCH_ENABLE_CUDA
