#pragma once

// Optional Intel MKL CSR SpMV baseline (mkl_sparse_d_mv). Compiled only when
// SPMVBENCH_HAS_MKL is defined; otherwise the stubs report unavailability.

#include "spmvbench/csr.hpp"
#include "spmvbench/timer.hpp"

namespace spmvbench {

bool mkl_available();

// fp64 only: mkl_sparse_d_mv. Returns an empty summary when MKL is absent.
TimingSummary mkl_spmv_f64(const CsrMatrix<double>& A, const double* x,
                           double* y, int iters, int warmup);

}  // namespace spmvbench
