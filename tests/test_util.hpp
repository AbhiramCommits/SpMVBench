#pragma once

// Shared helpers for the SpMVBench GoogleTest suite.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "spmvbench/csr.hpp"

namespace spmvtest {

inline std::string write_temp(const std::string& name, const std::string& body) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary);
  out << body;
  out.close();
  return path.string();
}

// Build a CSR from explicit per-row (column, value) lists (may be unsorted).
inline spmvbench::CsrMatrix<double> make_csr(
    int m, int n, const std::vector<std::vector<std::pair<int, double>>>& rows) {
  spmvbench::CsrMatrix<double> A;
  A.m = m;
  A.n = n;
  A.rowptr.assign(m + 1, 0);
  for (int i = 0; i < m; ++i) {
    A.rowptr[i + 1] = A.rowptr[i] + static_cast<spmvbench::Index>(rows[i].size());
  }
  A.nnz = A.rowptr[m];
  A.colind.resize(static_cast<std::size_t>(A.nnz));
  A.values.resize(static_cast<std::size_t>(A.nnz));
  for (int i = 0; i < m; ++i) {
    spmvbench::Index p = A.rowptr[i];
    for (const auto& [c, v] : rows[i]) {
      A.colind[p] = c;
      A.values[p] = v;
      ++p;
    }
  }
  return A;
}

inline std::vector<double> naive_spmv(const spmvbench::CsrMatrix<double>& A,
                                      const std::vector<double>& x) {
  std::vector<double> y(A.m, 0.0);
  for (int i = 0; i < A.m; ++i) {
    double s = 0.0;
    for (spmvbench::Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      s += A.values[p] * x[A.colind[p]];
    }
    y[i] = s;
  }
  return y;
}

template <typename U>
inline bool approx_eq(const std::vector<U>& a, const std::vector<double>& b,
                      double tol = 1e-12) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
    const double r = std::max(std::fabs(b[i]), 1.0);
    if (d > tol * r) return false;
  }
  return true;
}

}  // namespace spmvtest
