// cuSPARSE SpMV baseline (compiled only when SPMVBENCH_ENABLE_CUDA).
//
// Mirrors the benchmark protocol of SpmvRunner: upload once, warm up, then
// time `iters` launches with CUDA events and report mean/min/max ms.

#include "spmvbench/spmv_cuda.cuh"

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
SpmvResult run_cusparse_spmv(const CsrMatrix<T>& A, const std::vector<T>& x,
                             std::vector<T>& y, int iters, int warmup) {
  const int m = A.m;
  const int n = A.n;
  const std::int64_t nnz = A.nnz;

  Index* d_rowptr = nullptr;
  Index* d_colind = nullptr;
  T* d_values = nullptr;
  T* d_x = nullptr;
  T* d_y = nullptr;
  void* d_buffer = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowptr),
                        static_cast<std::size_t>(m + 1) * sizeof(Index)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_colind),
                 static_cast<std::size_t>(std::max<std::int64_t>(nnz, 1)) *
                     sizeof(Index)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_values),
      static_cast<std::size_t>(std::max<std::int64_t>(nnz, 1)) * sizeof(T)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x),
                        static_cast<std::size_t>(std::max(n, 1)) * sizeof(T)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y),
                        static_cast<std::size_t>(std::max(m, 1)) * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(d_rowptr, A.rowptr.data(), (m + 1) * sizeof(Index),
                        cudaMemcpyHostToDevice));
  if (nnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_colind, A.colind.data(), nnz * sizeof(Index),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), nnz * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA = nullptr;
  cusparseDnVecDescr_t vecX = nullptr;
  cusparseDnVecDescr_t vecY = nullptr;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  CUSPARSE_CHECK(cusparseCreateCsr(
      &matA, m, n, nnz, d_rowptr, d_colind, d_values, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_dtype<T>()));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_x, cuda_dtype<T>()));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, cuda_dtype<T>()));

  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;

  std::size_t buffer_size = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, op, &alpha, matA, vecX, &beta,
                                         vecY, cuda_dtype<T>(), alg,
                                         &buffer_size));
  if (buffer_size > 0)
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

  for (int i = 0; i < warmup; ++i) {
    CUSPARSE_CHECK(cusparseSpMV(handle, op, &alpha, matA, vecX, &beta, vecY,
                                cuda_dtype<T>(), alg, d_buffer));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(std::max(1, iters)));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUSPARSE_CHECK(cusparseSpMV(handle, op, &alpha, matA, vecX, &beta, vecY,
                                cuda_dtype<T>(), alg, d_buffer));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(static_cast<double>(ms));
  }

  y.resize(m);
  // Separate transfer timing: H2D x + D2H y, kernel excluded.
  std::vector<double> xfer;
  xfer.reserve(10);
  for (int i = 0; i < 10; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(
        cudaMemcpy(d_x, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(y.data(), d_y, m * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    xfer.push_back(static_cast<double>(ms));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
  cudaFree(d_rowptr);
  cudaFree(d_colind);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_buffer);

  const TimingSummary st = summarize(samples);
  SpmvResult r;
  r.iters = st.iters;
  r.mean_ms = st.mean_ms;
  r.min_ms = st.min_ms;
  r.median_ms = st.median_ms;
  r.max_ms = st.max_ms;
  r.transfer_ms = summarize(xfer).median_ms;
  r.kernel = "cusparse_spmv";
  return r;
}

}  // namespace

SpmvResult cusparse_spmv_csr_f32(const CsrMatrix<float>& A,
                                 const std::vector<float>& x,
                                 std::vector<float>& y, int iters, int warmup) {
  return run_cusparse_spmv<float>(A, x, y, iters, warmup);
}

SpmvResult cusparse_spmv_csr_f64(const CsrMatrix<double>& A,
                                 const std::vector<double>& x,
                                 std::vector<double>& y, int iters,
                                 int warmup) {
  return run_cusparse_spmv<double>(A, x, y, iters, warmup);
}

}  // namespace spmvbench
