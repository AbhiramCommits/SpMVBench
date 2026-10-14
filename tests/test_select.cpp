// Heuristic tests: synthetic RowStats must return the documented branch.

#include <gtest/gtest.h>

#include "spmvbench/select.hpp"

using spmvbench::choose_spmv;
using spmvbench::KernelId;
using spmvbench::RowStats;

TEST(Select, TinyProblemScalar) {
  RowStats s;
  s.nnz = 1000;
  s.mean_row_nnz = 10;
  s.cv_row_nnz = 0.0;
  EXPECT_EQ(choose_spmv(s).kernel, KernelId::CsrScalar);
}

TEST(Select, ShortRowsScalar) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 2.0;
  s.cv_row_nnz = 0.3;
  EXPECT_EQ(choose_spmv(s).kernel, KernelId::CsrScalar);
}

TEST(Select, DenseBlocksBcsr) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 8.0;
  s.cv_row_nnz = 0.1;
  s.frac_dense_blocks_4x4 = 0.9;
  const auto c = choose_spmv(s);
  EXPECT_EQ(c.kernel, KernelId::Bcsr);
  EXPECT_EQ(c.block_r, 4);
  EXPECT_EQ(c.block_c, 4);
}

TEST(Select, SkewedMerge) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 27.0;
  s.cv_row_nnz = 5.0;
  s.max_over_mean = 200.0;
  EXPECT_EQ(choose_spmv(s).kernel, KernelId::CsrMerge);
}

TEST(Select, LowPaddingEll) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 5.0;
  s.cv_row_nnz = 0.1;
  s.max_over_mean = 1.2;
  s.ell_padding_overhead = 1.05;
  EXPECT_EQ(choose_spmv(s).kernel, KernelId::Ell);
}

TEST(Select, MediumUniformVectorWithLanes) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 10.0;
  s.cv_row_nnz = 0.2;
  s.max_over_mean = 2.0;
  s.ell_padding_overhead = 2.5;  // too much padding for ELL
  const auto c = choose_spmv(s);
  EXPECT_EQ(c.kernel, KernelId::CsrVector);
  EXPECT_EQ(c.lanes, 16);  // next_pow2(10)

  s.mean_row_nnz = 20.0;
  const auto c2 = choose_spmv(s);
  EXPECT_EQ(c2.kernel, KernelId::CsrVector);
  EXPECT_EQ(c2.lanes, 32);  // next_pow2(20)
}

TEST(Select, ReasonIsNonEmpty) {
  RowStats s;
  s.nnz = 1'000'000;
  s.mean_row_nnz = 20.0;
  s.cv_row_nnz = 0.1;
  s.ell_padding_overhead = 5.0;
  EXPECT_FALSE(choose_spmv(s).reason.empty());
}
