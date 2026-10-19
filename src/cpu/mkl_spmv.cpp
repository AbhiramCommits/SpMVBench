// Optional Intel MKL CSR SpMV baseline.
//
// mkl_sparse_d_mv with a CSR descriptor created zero-based from our arrays.
// The matrix is created and optimized once per call; timing covers only the
// repeated mkl_sparse_d_mv invocations.

#include "spmvbench/mkl_spmv.hpp"

#include <vector>

#ifdef SPMVBENCH_HAS_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

namespace spmvbench {

bool mkl_available() {
#ifdef SPMVBENCH_HAS_MKL
  return true;
#else
  return false;
#endif
}

#ifdef SPMVBENCH_HAS_MKL

TimingSummary mkl_spmv_f64(const CsrMatrix<double>& A, const double* x,
                           double* y, int iters, int warmup) {
  sparse_matrix_t mklA = nullptr;
  MKL_INT rows = A.m;
  MKL_INT cols = A.n;
  // MKL uses MKL_INT (int32 by default), matching our Index type.
  sparse_status_t st =
      mkl_sparse_d_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, rows, cols,
                              const_cast<MKL_INT*>(A.rowptr.data()),
                              const_cast<MKL_INT*>(A.rowptr.data() + 1),
                              const_cast<MKL_INT*>(A.colind.data()),
                              const_cast<double*>(A.values.data()));
  if (st != SPARSE_STATUS_SUCCESS)
    return TimingSummary{};

  matrix_descr descr{};
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_optimize(mklA);

  const double alpha = 1.0;
  const double beta = 0.0;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < warmup; ++i) {
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklA, descr, x, beta,
                    y);
  }
  CpuTimer timer;
  for (int i = 0; i < iters; ++i) {
    timer.start();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mklA, descr, x, beta,
                    y);
    timer.stop();
    samples.push_back(timer.ms());
  }
  mkl_sparse_destroy(mklA);
  return summarize(samples);
}

#else  // !SPMVBENCH_HAS_MKL

TimingSummary mkl_spmv_f64(const CsrMatrix<double>&, const double*, double*,
                           int, int) {
  return TimingSummary{};
}

#endif  // SPMVBENCH_HAS_MKL

}  // namespace spmvbench
