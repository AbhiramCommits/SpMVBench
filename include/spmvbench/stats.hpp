#pragma once

// Nonzero-distribution statistics used by the kernel-selection heuristic.

#include <cstdint>
#include <string>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"

namespace spmvbench {

struct RowStats {
  std::int64_t nnz = 0;  // total nonzeros
  int rows = 0;
  int cols = 0;

  double mean_row_nnz = 0.0;
  double median_row_nnz = 0.0;
  int min_row_nnz = 0;
  int max_row_nnz = 0;
  double stddev_row_nnz = 0.0;
  double cv_row_nnz = 0.0;        // coefficient of variation = stddev / mean
  double max_over_mean = 0.0;     // skew indicator

  // Histogram of row lengths (histogram[i] = rows with exactly i nnz).
  std::vector<std::int64_t> row_nnz_histogram;

  // ELL padding overhead for the natural max_row_nnz padding.
  double ell_padding_overhead = 1.0;

  // Fraction of 4x4 blocks in the matrix that are > 50% populated.
  double frac_dense_blocks_4x4 = 0.0;

  // Structural density nnz / (rows * cols).
  double density = 0.0;

  std::string to_string() const;
};

// Compute stats from raw CSR arrays. `rowptr` has m + 1 entries, `colind` has
// nnz entries (only read to detect dense 4x4 blocks).
RowStats compute_row_stats(const Index* rowptr, const Index* colind, int m,
                           int n, std::int64_t nnz);

// Convenience wrapper for a typed CSR matrix.
template <typename T>
RowStats compute_row_stats(const CsrMatrix<T>& csr) {
  return compute_row_stats(csr.rowptr.data(), csr.colind.data(), csr.m, csr.n,
                           csr.nnz);
}

}  // namespace spmvbench
