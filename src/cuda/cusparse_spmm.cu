// cuSPARSE SpMM baseline: Y (m x k) = A (m x n, CSR) * B (n x k, dense
// row-major). Compiled only when SPMVBENCH_ENABLE_CUDA.

#include "spmvbench/spmm_cuda.cuh"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "spmvbench/timer.hpp"

namespace spmvbench {
namespace {

template <typename T>
cudaDataType_t cuda_dtype();
template <>
cudaDataType_t cuda_dtype<float>() {
  return CUDA_R_32F;
}
template <>
cudaDataType_t cuda_dtype<double>() {
  return CUDA_R_64F;
}

template <typename T>
SpmmResult run_cusparse_spmm(const CsrMatrix<T>& A, const std::vector<T>& B,
                             int k, int iters, int warmup, std::vector<T>& Y) {
  const int m = A.m;
  const int n = A.n;

  Index* d_rowptr = nullptr;
  Index* d_colind = nullptr;
  T* d_values = nullptr;
  T* d_B = nullptr;
  T* d_C = nullptr;
  void* d_buffer = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowptr),
                        (std::size_t)(m + 1) * sizeof(Index)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_colind),
      (std::size_t)std::max<std::int64_t>(A.nnz, 1) * sizeof(Index)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_values),
                 (std::size_t)std::max<std::int64_t>(A.nnz, 1) * sizeof(T)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B),
                        (std::size_t)n * k * sizeof(T)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C),
                        (std::size_t)m * k * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_rowptr, A.rowptr.data(), (m + 1) * sizeof(Index),
                        cudaMemcpyHostToDevice));
  if (A.nnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_colind, A.colind.data(), A.nnz * sizeof(Index),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.nnz * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), (std::size_t)n * k * sizeof(T),
                        cudaMemcpyHostToDevice));

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA = nullptr;
  cusparseDnMatDescr_t matB = nullptr;
  cusparseDnMatDescr_t matC = nullptr;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseCreateCsr(
      &matA, m, n, A.nnz, d_rowptr, d_colind, d_values, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_dtype<T>()));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, n, k, k, d_B, cuda_dtype<T>(),
                                     CUSPARSE_ORDER_ROW));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, m, k, k, d_C, cuda_dtype<T>(),
                                     CUSPARSE_ORDER_ROW));

  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  const cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;

  std::size_t buffer_size = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      cuda_dtype<T>(), alg, &buffer_size));
  if (buffer_size > 0)
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

  for (int i = 0; i < warmup; ++i) {
    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, cuda_dtype<T>(), alg,
                                d_buffer));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(std::max(1, iters)));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                matB, &beta, matC, cuda_dtype<T>(), alg,
                                d_buffer));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }

  Y.resize((std::size_t)m * k);
  // Separate transfer timing: H2D B + D2H C, kernel excluded.
  std::vector<double> xfer;
  xfer.reserve(10);
  for (int i = 0; i < 10; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), (std::size_t)n * k * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y.data(), d_C, (std::size_t)m * k * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    xfer.push_back(static_cast<double>(ms));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  cusparseDestroyDnMat(matB);
  cusparseDestroyDnMat(matC);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
  cudaFree(d_rowptr);
  cudaFree(d_colind);
  cudaFree(d_values);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_buffer);

  const TimingSummary st = summarize(samples);
  SpmmResult r;
  r.iters = st.iters;
  r.mean_ms = st.mean_ms;
  r.min_ms = st.min_ms;
  r.median_ms = st.median_ms;
  r.max_ms = st.max_ms;
  r.transfer_ms = summarize(xfer).median_ms;
  r.kernel = "cusparse_spmm";
  return r;
}

}  // namespace

SpmmResult cusparse_spmm_f32(const CsrMatrix<float>& A,
                             const std::vector<float>& B, int k, int iters,
                             int warmup, std::vector<float>& Y) {
  return run_cusparse_spmm<float>(A, B, k, iters, warmup, Y);
}

SpmmResult cusparse_spmm_f64(const CsrMatrix<double>& A,
                             const std::vector<double>& B, int k, int iters,
                             int warmup, std::vector<double>& Y) {
  return run_cusparse_spmm<double>(A, B, k, iters, warmup, Y);
}

}  // namespace spmvbench
