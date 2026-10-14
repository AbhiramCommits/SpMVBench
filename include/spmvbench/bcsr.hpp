#pragma once

// Blocked CSR (BCSR): CSR over a grid of R x C dense blocks. Each stored
// block keeps R*C values in row-major order. This is the format that wins
// when the nonzero pattern has locally dense structure.

#include <cstdint>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"

namespace spmvbench {

template <typename T>
struct BcsrMatrix {
  using value_type = T;

  int m = 0;   // original rows
  int n = 0;   // original columns
  int R = 0;   // block rows
  int C = 0;   // block cols
  int mb = 0;  // number of block rows  (ceil(m / R))
  int nb = 0;  // number of block cols  (ceil(n / C))
  std::int64_t nnz = 0;        // original (unpadded) nonzeros
  std::int64_t block_nnz = 0;  // number of stored R x C blocks
  std::vector<Index> rowptr;   // size mb + 1
  std::vector<Index> colind;   // size block_nnz, block column indices
  std::vector<T> values;       // size block_nnz * R * C, row-major blocks

  // Fraction of stored block slots that are actually nonzero.
  double density() const {
    const std::int64_t slots = block_nnz * static_cast<std::int64_t>(R) * C;
    if (slots == 0) return 0.0;
    return static_cast<double>(nnz) / static_cast<double>(slots);
  }
};

// Build a BCSR matrix from CSR with compile-time block shape <R, C>.
template <typename T, int R, int C>
BcsrMatrix<T> csr_to_bcsr(const CsrMatrix<T>& csr);

}  // namespace spmvbench
