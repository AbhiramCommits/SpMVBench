#pragma once

// Matrix Market reader.
//
// Supported: `%%MatrixMarket matrix coordinate <real|pattern> <general|symmetric>`,
// 1-based indices. Symmetric input is expanded to a full general matrix.
//
// Explicitly rejected (with a clear error): `complex` values, `array` format,
// and any object that is not a `matrix`. Skew-symmetric is rejected too.

#include <string>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"

namespace spmvbench {

// Read a Matrix Market file into CSR, casting stored values to T.
// `pattern` entries are given the value 1.
template <typename T>
CsrMatrix<T> read_matrix_market(const std::string& path);

// Load `data/ref` reference data etc. live in other translation units; this
// header only exposes the reader.
}  // namespace spmvbench
