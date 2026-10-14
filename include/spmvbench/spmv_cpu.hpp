#pragma once

// Portable OpenMP CPU kernels: the reference/fallback path that is always
// compiled. Every function is templated on the *storage* value type `T` and
// the *accumulator/output* type `Acc`, so mixed precision is expressible.

#include "spmvbench/bcsr.hpp"
#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/ell.hpp"

namespace spmvbench {

// y = A * x, OpenMP parallel over rows (static schedule).
template <typename T, typename Acc>
void spmv_csr_cpu(const CsrMatrix<T>& A, const T* x, Acc* y, int nthreads = 0);

// y = A * x, strictly serial. Portable scalar fallback and single-thread
// baseline for speedup reporting.
template <typename T, typename Acc>
void spmv_csr_scalar_cpu(const CsrMatrix<T>& A, const T* x, Acc* y);

// y = A * x, OpenMP dynamic schedule so skewed row lengths do not stall.
template <typename T, typename Acc>
void spmv_csr_merge_cpu(const CsrMatrix<T>& A, const T* x, Acc* y,
                        int nthreads = 0);

// y = A * x over padded ELL columns.
template <typename T, typename Acc>
void spmv_ell_cpu(const EllMatrix<T>& A, const T* x, Acc* y, int nthreads = 0);

// y = A * x over R x C dense blocks (runtime block shape).
template <typename T, typename Acc>
void spmv_bcsr_cpu(const BcsrMatrix<T>& A, const T* x, Acc* y,
                   int nthreads = 0);

// Kahan-compensated y = A * x (extra accuracy datapoint).
template <typename T, typename Acc>
void spmv_csr_kahan_cpu(const CsrMatrix<T>& A, const T* x, Acc* y,
                        int nthreads = 0);

// Y = A * B with row-major B (n x k) and Y (m x k), OpenMP over rows.
template <typename T, typename Acc>
void spmm_csr_cpu(const CsrMatrix<T>& A, const T* B, Acc* Y, int k,
                  int nthreads = 0);

}  // namespace spmvbench
