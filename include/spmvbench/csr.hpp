#pragma once

// Compressed Sparse Row (CSR) matrix and its host-side construction helpers.

#include <cstdint>
#include <string>
#include <vector>

#include "spmvbench/dtypes.hpp"

namespace spmvbench {

// Canonical CSR: row `i` owns the entries `rowptr[i] .. rowptr[i+1]` in
// `values` / `colind`. Column indices within a row are normally sorted and
// duplicate-free, but the kernels do not rely on that.
template <typename T>
struct CsrMatrix {
  using value_type = T;

  int m = 0;               // number of rows
  int n = 0;               // number of columns
  std::int64_t nnz = 0;    // number of stored nonzeros
  std::vector<Index> rowptr;  // size m + 1
  std::vector<Index> colind;  // size nnz
  std::vector<T> values;      // size nnz

  CsrMatrix() = default;
  CsrMatrix(int rows, int cols) : m(rows), n(cols), rowptr(rows + 1, 0) {}

  // Total bytes of the three CSR arrays (used by the bandwidth model).
  std::int64_t storage_bytes() const {
    return static_cast<std::int64_t>(nnz) * (sizeof(T) + sizeof(Index)) +
           static_cast<std::int64_t>(m + 1) * sizeof(Index);
  }
};

// A deterministic description of a matrix that is independent of the value
// type, so it can be computed once and shared by every kernel.
struct MatrixInfo {
  std::string name;
  int m = 0;
  int n = 0;
  std::int64_t nnz = 0;
};

}  // namespace spmvbench
