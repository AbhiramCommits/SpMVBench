// OpenMP CPU SpMM (sparse x dense) reference kernel: Y = A * B where B and Y
// are row-major dense matrices with k columns.

#include "spmvbench/spmv_cpu.hpp"

#include "spmvbench/omp_compat.hpp"

namespace spmvbench {

template <typename T, typename Acc>
void spmm_csr_cpu(const CsrMatrix<T>& A, const T* B, Acc* Y, int k,
                  int nthreads) {
  SPMVBENCH_SET_THREADS(nthreads);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < A.m; ++i) {
    // Zero the output row once.
    Acc* yrow = Y + static_cast<std::size_t>(i) * k;
    for (int j = 0; j < k; ++j)
      yrow[j] = Acc(0);
    for (Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      const Index col = A.colind[p];
      const Acc a = static_cast<Acc>(A.values[p]);
      const T* brow = B + static_cast<std::size_t>(col) * k;
      for (int j = 0; j < k; ++j) {
        yrow[j] += a * static_cast<Acc>(brow[j]);
      }
    }
  }
}

#define SPMVBENCH_INSTANTIATE_SPMM(T, Acc)                                     \
  template void spmm_csr_cpu<T, Acc>(const CsrMatrix<T>&, const T*, Acc*, int, \
                                     int);

SPMVBENCH_INSTANTIATE_SPMM(float, float)
SPMVBENCH_INSTANTIATE_SPMM(float, double)
SPMVBENCH_INSTANTIATE_SPMM(double, double)

#undef SPMVBENCH_INSTANTIATE_SPMM

}  // namespace spmvbench
