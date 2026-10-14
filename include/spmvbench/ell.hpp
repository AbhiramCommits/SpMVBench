#pragma once

// ELLPACK (ELL) storage: a dense `m x max_row_nnz` matrix of (column, value)
// pairs, stored column-major so that consecutive rows are contiguous.
//
// `colind[j * m + i]` is the column of row `i`'s j-th stored entry.

#include <cstdint>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"

namespace spmvbench {

template <typename T>
struct EllMatrix {
  using value_type = T;

  int m = 0;                 // rows
  int n = 0;                 // columns
  int max_row_nnz = 0;       // padded columns per row
  std::int64_t nnz = 0;      // original (unpadded) nonzeros
  std::int64_t padded_nnz = 0;  // m * max_row_nnz
  std::vector<Index> colind;    // size padded_nnz, column-major
  std::vector<T> values;        // size padded_nnz, column-major

  // Padding overhead ratio: how many more entries we store than we need.
  double padding_overhead() const {
    if (nnz == 0) return 1.0;
    return static_cast<double>(padded_nnz) / static_cast<double>(nnz);
  }
};

// Convert a CSR matrix to ELL, padding short rows with column 0 / value 0.
template <typename T>
EllMatrix<T> csr_to_ell(const CsrMatrix<T>& csr);

}  // namespace spmvbench
