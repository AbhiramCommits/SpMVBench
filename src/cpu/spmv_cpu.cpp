// OpenMP CPU SpMV kernels and the Kahan-compensated variant.
//
// These are the portable reference implementations used for validation and
// as the CPU benchmark path when no CUDA GPU is present.

#include "spmvbench/spmv_cpu.hpp"

#include <cmath>
#include <cstddef>

#include "spmvbench/omp_compat.hpp"

namespace spmvbench {

template <typename T, typename Acc>
void spmv_csr_cpu(const CsrMatrix<T>& A, const T* x, Acc* y, int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < A.m; ++i) {
    Acc sum = Acc(0);
    for (Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
    }
    y[i] = sum;
  }
}

template <typename T, typename Acc>
void spmv_csr_scalar_cpu(const CsrMatrix<T>& A, const T* x, Acc* y) {
  for (int i = 0; i < A.m; ++i) {
    Acc sum = Acc(0);
    for (Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
    }
    y[i] = sum;
  }
}

template <typename T, typename Acc>
void spmv_csr_merge_cpu(const CsrMatrix<T>& A, const T* x, Acc* y,
                        int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
  // Dynamic scheduling so the long rows of a skewed matrix do not pile up on
  // a single thread.
#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < A.m; ++i) {
    Acc sum = Acc(0);
    for (Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
    }
    y[i] = sum;
  }
}

template <typename T, typename Acc>
void spmv_ell_cpu(const EllMatrix<T>& A, const T* x, Acc* y, int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
  const int m = A.m;
  const int J = A.max_row_nnz;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; ++i) {
    Acc sum = Acc(0);
    for (int j = 0; j < J; ++j) {
      const Index p = static_cast<Index>(j) * m + i;  // column-major
      sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
    }
    y[i] = sum;
  }
}

template <typename T, typename Acc>
void spmv_bcsr_cpu(const BcsrMatrix<T>& A, const T* x, Acc* y, int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
  const int R = A.R;
  const int C = A.C;
#pragma omp parallel for schedule(static)
  for (int bi = 0; bi < A.mb; ++bi) {
    // Accumulate a whole block row into a small local buffer.
    Acc acc[16];  // covers R up to 16, only R<=4 used in practice
    for (int r = 0; r < R; ++r) acc[r] = Acc(0);
    for (Index bp = A.rowptr[bi]; bp < A.rowptr[bi + 1]; ++bp) {
      const Index bj = A.colind[bp];
      const T* blk = &A.values[static_cast<std::size_t>(bp) * R * C];
      for (int r = 0; r < R; ++r) {
        Acc s = Acc(0);
        for (int c = 0; c < C; ++c) {
          const int gc = bj * C + c;  // global column
          if (gc >= A.n) break;       // trailing partial block
          s += static_cast<Acc>(blk[r * C + c]) * static_cast<Acc>(x[gc]);
        }
        acc[r] += s;
      }
    }
    for (int r = 0; r < R; ++r) {
      const int gi = bi * R + r;
      if (gi < A.m) y[gi] = acc[r];
    }
  }
}

template <typename T, typename Acc>
void spmv_csr_kahan_cpu(const CsrMatrix<T>& A, const T* x, Acc* y,
                        int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < A.m; ++i) {
    Acc sum = Acc(0);
    Acc comp = Acc(0);
    for (Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      const Acc prod =
          static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
      const Acc t = sum + prod;
      if (std::abs(sum) >= std::abs(prod)) {
        comp += (sum - t) + prod;
      } else {
        comp += (prod - t) + sum;
      }
      sum = t;
    }
    y[i] = sum + comp;
  }
}

// Explicit instantiations for the precision combinations we exercise.
#define SPMVBENCH_INSTANTIATE_SPMV(T, Acc)                                   \
  template void spmv_csr_cpu<T, Acc>(const CsrMatrix<T>&, const T*, Acc*,    \
                                     int);                                   \
  template void spmv_csr_scalar_cpu<T, Acc>(const CsrMatrix<T>&, const T*,   \
                                            Acc*);                           \
  template void spmv_csr_merge_cpu<T, Acc>(const CsrMatrix<T>&, const T*,    \
                                           Acc*, int);                       \
  template void spmv_ell_cpu<T, Acc>(const EllMatrix<T>&, const T*, Acc*,    \
                                     int);                                   \
  template void spmv_bcsr_cpu<T, Acc>(const BcsrMatrix<T>&, const T*, Acc*,  \
                                      int);                                  \
  template void spmv_csr_kahan_cpu<T, Acc>(const CsrMatrix<T>&, const T*,    \
                                           Acc*, int);

SPMVBENCH_INSTANTIATE_SPMV(float, float)
SPMVBENCH_INSTANTIATE_SPMV(float, double)
SPMVBENCH_INSTANTIATE_SPMV(double, double)

#undef SPMVBENCH_INSTANTIATE_SPMV

}  // namespace spmvbench
