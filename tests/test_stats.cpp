// Row-statistics tests with hand-computed expected values.

#include <gtest/gtest.h>

#include <cmath>

#include "spmvbench/stats.hpp"
#include "test_util.hpp"

TEST(Stats, HandComputedUniformish) {
  // Row lengths {1, 2, 3, 4}: deterministic stats.
  std::vector<std::vector<std::pair<int, double>>> rows(4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j <= i; ++j) rows[i].push_back({j, 1.0});
  const auto A = spmvtest::make_csr(4, 4, rows);

  const auto s = spmvbench::compute_row_stats(A);
  EXPECT_EQ(s.rows, 4);
  EXPECT_EQ(s.cols, 4);
  EXPECT_EQ(s.nnz, 10);
  EXPECT_DOUBLE_EQ(s.mean_row_nnz, 2.5);
  EXPECT_DOUBLE_EQ(s.median_row_nnz, 2.5);  // even count -> average of 2 and 3
  EXPECT_EQ(s.min_row_nnz, 1);
  EXPECT_EQ(s.max_row_nnz, 4);
  EXPECT_NEAR(s.stddev_row_nnz, std::sqrt(1.25), 1e-12);
  EXPECT_NEAR(s.cv_row_nnz, std::sqrt(1.25) / 2.5, 1e-12);
  EXPECT_NEAR(s.max_over_mean, 4.0 / 2.5, 1e-12);
  EXPECT_DOUBLE_EQ(s.density, 10.0 / 16.0);
  EXPECT_DOUBLE_EQ(s.ell_padding_overhead, 16.0 / 10.0);

  ASSERT_EQ(s.row_nnz_histogram.size(), 5u);
  EXPECT_EQ(s.row_nnz_histogram[0], 0);
  for (int k = 1; k <= 4; ++k) EXPECT_EQ(s.row_nnz_histogram[k], 1);
}

TEST(Stats, ConstantRowsHaveZeroCv) {
  std::vector<std::vector<std::pair<int, double>>> rows(3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 8; ++j) rows[i].push_back({j, 1.0});
  const auto A = spmvtest::make_csr(3, 8, rows);
  const auto s = spmvbench::compute_row_stats(A);
  EXPECT_DOUBLE_EQ(s.cv_row_nnz, 0.0);
  EXPECT_DOUBLE_EQ(s.ell_padding_overhead, 1.0);
  EXPECT_EQ(s.median_row_nnz, 8.0);
}

TEST(Stats, SkewedCv) {
  // lengths {1, 1, 1, 1000}: for k points with one dominant value the CV
  // approaches sqrt(k-1) = sqrt(3) ~ 1.73 for k = 4.
  std::vector<std::vector<std::pair<int, double>>> rows(4);
  rows[0].push_back({0, 1.0});
  rows[1].push_back({1, 1.0});
  rows[2].push_back({2, 1.0});
  for (int j = 0; j < 1000; ++j) rows[3].push_back({j, 1.0});
  const auto A = spmvtest::make_csr(4, 1000, rows);
  const auto s = spmvbench::compute_row_stats(A);
  EXPECT_DOUBLE_EQ(s.mean_row_nnz, 1003.0 / 4.0);
  EXPECT_GT(s.cv_row_nnz, 1.5);
  EXPECT_GT(s.max_over_mean, 2.0);
}

TEST(Stats, DenseBlockFraction) {
  // A 4x4 fully dense matrix is one >50% block.
  std::vector<std::vector<std::pair<int, double>>> rows(4);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) rows[i].push_back({j, 1.0});
  const auto A = spmvtest::make_csr(4, 4, rows);
  const auto s = spmvbench::compute_row_stats(A);
  EXPECT_DOUBLE_EQ(s.frac_dense_blocks_4x4, 1.0);
}
