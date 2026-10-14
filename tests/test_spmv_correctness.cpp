// SpMV correctness: every CPU kernel vs a naive fp64 reference, on hand-built
// edge cases (empty rows, zero first/last row, single row, a heavy row,
// unsorted indices) and on the generated matrices when present.

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "spmvbench/bcsr.hpp"
#include "spmvbench/ell.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/spmv_cpu.hpp"
#include "test_util.hpp"

namespace {

void check_all_kernels(const spmvbench::CsrMatrix<double>& A,
                       const std::vector<double>& x, double tol = 1e-12) {
  const std::vector<double> ref = spmvtest::naive_spmv(A, x);
  std::vector<double> y(A.m);

  spmvbench::spmv_csr_scalar_cpu<double, double>(A, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "csr_scalar";

  spmvbench::spmv_csr_cpu<double, double>(A, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "csr_vector";

  spmvbench::spmv_csr_merge_cpu<double, double>(A, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "csr_merge";

  const auto ell = spmvbench::csr_to_ell(A);
  spmvbench::spmv_ell_cpu<double, double>(ell, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "ell";

  const auto b2 = spmvbench::csr_to_bcsr<double, 2, 2>(A);
  spmvbench::spmv_bcsr_cpu<double, double>(b2, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "bcsr2x2";

  const auto b4 = spmvbench::csr_to_bcsr<double, 4, 4>(A);
  spmvbench::spmv_bcsr_cpu<double, double>(b4, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref, tol)) << "bcsr4x4";
}

}  // namespace

TEST(SpmvCorrectness, EmptyFirstAndLastRows) {
  // rows: {}, {0,1}, {}, {2,3,4}, {}
  const auto A = spmvtest::make_csr(
      5, 5,
      {{}, {{0, 1.5}, {1, 2.0}}, {}, {{2, 3.0}, {3, 4.0}, {4, 5.0}}, {}});
  const std::vector<double> x = {1, 2, 3, 4, 5};
  check_all_kernels(A, x);
}

TEST(SpmvCorrectness, SingleRow) {
  // One row, unsorted columns.
  const auto A = spmvtest::make_csr(
      1, 4, {{{3, 1.0}, {0, 2.0}, {2, 3.0}, {1, 4.0}}});
  const std::vector<double> x = {1, 1, 1, 1};
  check_all_kernels(A, x);
  // 1 + 2 + 3 + 4 = 10
  std::vector<double> y(1);
  spmvbench::spmv_csr_cpu<double, double>(A, x.data(), y.data());
  EXPECT_DOUBLE_EQ(y[0], 10.0);
}

TEST(SpmvCorrectness, OneHeavyRow) {
  // 4 rows, 20 nonzeros, row 1 holds 6 of them (30%).
  std::vector<std::vector<std::pair<int, double>>> rows(4);
  rows[0] = {{0, 1}, {1, 1}, {2, 1}};                                  // 3
  rows[1] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}};          // 6
  rows[2] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}};                  // 5
  rows[3] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}};          // 6
  const auto A = spmvtest::make_csr(4, 6, rows);
  EXPECT_EQ(A.nnz, 20);
  std::vector<double> x(6, 1.0);
  check_all_kernels(A, x);
}

TEST(SpmvCorrectness, UnsortedColumns) {
  const auto A = spmvtest::make_csr(
      3, 5,
      {{{4, 2.0}, {1, 3.0}, {0, 1.0}},
       {{3, 4.0}, {2, 5.0}},
       {{0, 6.0}, {4, 7.0}, {2, 8.0}, {1, 9.0}}});
  const std::vector<double> x = {0.5, -1.0, 2.0, 3.0, -4.0};
  check_all_kernels(A, x);
}

TEST(SpmvCorrectness, AllZeroMatrix) {
  const auto A = spmvtest::make_csr(3, 3, {{}, {}, {}});
  const std::vector<double> x = {1, 2, 3};
  check_all_kernels(A, x);
}

TEST(SpmvCorrectness, GeneratedMatrices) {
  const std::string data_dir = std::string(SPMVBENCH_SOURCE_DIR) + "/data";
  for (const char* name : {"tiny_3x3.mtx", "banded_100k.mtx",
                           "diag_dominant_small.mtx"}) {
    const std::string path = data_dir + "/" + name;
    if (!std::filesystem::exists(path)) {
      std::fprintf(stderr, "skip %s (run tools/gen_matrices.py)\n", path.c_str());
      continue;
    }
    const auto A = spmvbench::read_matrix_market<double>(path);
    std::vector<double> x(A.n);
    for (int i = 0; i < A.n; ++i) x[i] = (i % 7) - 3.0;
    check_all_kernels(A, x, 1e-10);
  }
}
