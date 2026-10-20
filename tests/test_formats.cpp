// Format conversion tests: CSR <-> ELL <-> BCSR preserve nnz and values, and
// padding math is correct.

#include <gtest/gtest.h>

#include "spmvbench/bcsr.hpp"
#include "spmvbench/ell.hpp"
#include "spmvbench/spmv_cpu.hpp"
#include "test_util.hpp"

namespace {

spmvbench::CsrMatrix<double> sample() {
  // 3 x 4, row lengths 2,1,4 -> max_row_nnz = 4
  return spmvtest::make_csr(3, 4,
                            {{{0, 1.0}, {3, 2.0}},
                             {{1, 3.0}},
                             {{0, 4.0}, {1, 5.0}, {2, 6.0}, {3, 7.0}}});
}

}  // namespace

TEST(Formats, EllPaddingMath) {
  const auto A = sample();
  const auto ell = spmvbench::csr_to_ell(A);
  EXPECT_EQ(ell.m, 3);
  EXPECT_EQ(ell.n, 4);
  EXPECT_EQ(ell.max_row_nnz, 4);
  EXPECT_EQ(ell.nnz, 7);
  EXPECT_EQ(ell.padded_nnz, 12);  // 3 rows * 4 padded columns
  EXPECT_DOUBLE_EQ(ell.padding_overhead(), 12.0 / 7.0);
  EXPECT_EQ(ell.values.size(), 12u);
  EXPECT_EQ(ell.colind.size(), 12u);
  ASSERT_EQ(ell.row_lengths.size(), 3u);
  EXPECT_EQ(ell.row_lengths[0], 2);
  EXPECT_EQ(ell.row_lengths[1], 1);
  EXPECT_EQ(ell.row_lengths[2], 4);

  // Padding entries are column 0 / value 0.
  // Row 1 has a single entry at j=0; j=1,2,3 are padding.
  for (int j = 1; j < 4; ++j) {
    const std::size_t p = static_cast<std::size_t>(j) * 3 + 1;
    EXPECT_EQ(ell.colind[p], 0);
    EXPECT_DOUBLE_EQ(ell.values[p], 0.0);
  }
}

TEST(Formats, EllPreservesValuesAndSpMV) {
  const auto A = sample();
  const auto ell = spmvbench::csr_to_ell(A);
  const std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
  const auto ref = spmvtest::naive_spmv(A, x);

  std::vector<double> y(3);
  spmvbench::spmv_ell_cpu<double, double>(ell, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref));

  double sum_csr = 0.0, sum_ell = 0.0;
  for (double v : A.values)
    sum_csr += v;
  for (double v : ell.values)
    sum_ell += v;  // padding adds zeros
  EXPECT_DOUBLE_EQ(sum_csr, sum_ell);
}

TEST(Formats, Bcsr2x2) {
  const auto A = sample();
  const auto b = spmvbench::csr_to_bcsr<double, 2, 2>(A);
  EXPECT_EQ(b.R, 2);
  EXPECT_EQ(b.C, 2);
  EXPECT_EQ(b.mb, 2);
  EXPECT_EQ(b.nb, 2);
  // block row 0: rows 0-1 touch columns {0,3,1} -> block cols {0,1}
  // block row 1: row 2 touches {0,1,2,3}        -> block cols {0,1}
  EXPECT_EQ(b.block_nnz, 4);
  EXPECT_EQ(b.nnz, A.nnz);
  EXPECT_EQ(b.values.size(), 4u * 2 * 2);

  const std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
  const auto ref = spmvtest::naive_spmv(A, x);
  std::vector<double> y(3);
  spmvbench::spmv_bcsr_cpu<double, double>(b, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref));
}

TEST(Formats, Bcsr4x4) {
  const auto A = sample();
  const auto b = spmvbench::csr_to_bcsr<double, 4, 4>(A);
  EXPECT_EQ(b.R, 4);
  EXPECT_EQ(b.C, 4);
  EXPECT_EQ(b.mb, 1);
  EXPECT_EQ(b.nb, 1);
  EXPECT_EQ(b.block_nnz, 1);

  const std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
  const auto ref = spmvtest::naive_spmv(A, x);
  std::vector<double> y(3);
  spmvbench::spmv_bcsr_cpu<double, double>(b, x.data(), y.data());
  EXPECT_TRUE(spmvtest::approx_eq(y, ref));
}

TEST(Formats, BcsrDensity) {
  const auto A = sample();
  const auto b = spmvbench::csr_to_bcsr<double, 4, 4>(A);
  // 7 nonzeros in one 16-slot block.
  EXPECT_DOUBLE_EQ(b.density(), 7.0 / 16.0);
}
