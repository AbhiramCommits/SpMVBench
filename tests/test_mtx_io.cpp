// Matrix Market reader tests: symmetric expansion, 1-based indices, pattern
// matrices, and malformed-header errors.

#include <gtest/gtest.h>

#include "spmvbench/mtx_io.hpp"
#include "test_util.hpp"

using spmvbench::read_matrix_market;

TEST(MtxIo, SymmetricExpansion) {
  const std::string body =
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "% a tiny symmetric matrix\n"
      "3 3 4\n"
      "1 1 1.0\n"
      "2 1 2.0\n"
      "2 2 3.0\n"
      "3 3 4.0\n";
  const std::string path = spmvtest::write_temp("sb_sym.mtx", body);
  auto A = read_matrix_market<double>(path);

  EXPECT_EQ(A.m, 3);
  EXPECT_EQ(A.n, 3);
  // (2,1) expands to (1,2) as well -> 5 stored entries.
  EXPECT_EQ(A.nnz, 5);
  // Row 0: (0,1.0) and (1,2.0)
  ASSERT_EQ(A.rowptr[1] - A.rowptr[0], 2);
  EXPECT_EQ(A.colind[A.rowptr[0]], 0);
  EXPECT_DOUBLE_EQ(A.values[A.rowptr[0]], 1.0);
  EXPECT_EQ(A.colind[A.rowptr[0] + 1], 1);
  EXPECT_DOUBLE_EQ(A.values[A.rowptr[0] + 1], 2.0);
}

TEST(MtxIo, OneBasedIndices) {
  const std::string body =
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "2 1 7.5\n";
  const std::string path = spmvtest::write_temp("sb_one_based.mtx", body);
  auto A = read_matrix_market<double>(path);
  ASSERT_EQ(A.nnz, 1);
  EXPECT_EQ(A.rowptr[0], 0);  // row 1 (0-based) is empty
  EXPECT_EQ(A.rowptr[1], 0);
  EXPECT_EQ(A.colind[0], 0);  // 1-based column 1 -> index 0
  EXPECT_DOUBLE_EQ(A.values[0], 7.5);
}

TEST(MtxIo, PatternGivesUnitValues) {
  const std::string body =
      "%%MatrixMarket matrix coordinate pattern general\n"
      "2 2 2\n"
      "1 2\n"
      "2 1\n";
  const std::string path = spmvtest::write_temp("sb_pattern.mtx", body);
  auto A = read_matrix_market<double>(path);
  ASSERT_EQ(A.nnz, 2);
  for (std::int64_t i = 0; i < A.nnz; ++i)
    EXPECT_DOUBLE_EQ(A.values[i], 1.0);
}

TEST(MtxIo, RejectsComplex) {
  const std::string body =
      "%%MatrixMarket matrix coordinate complex general\n"
      "1 1 1\n"
      "1 1 1.0 2.0\n";
  const std::string path = spmvtest::write_temp("sb_complex.mtx", body);
  EXPECT_THROW(read_matrix_market<double>(path), std::exception);
}

TEST(MtxIo, RejectsArrayFormat) {
  const std::string body =
      "%%MatrixMarket matrix array real general\n"
      "2 2\n"
      "1.0\n2.0\n3.0\n4.0\n";
  const std::string path = spmvtest::write_temp("sb_array.mtx", body);
  EXPECT_THROW(read_matrix_market<double>(path), std::exception);
}

TEST(MtxIo, RejectsMissingBanner) {
  const std::string body = "3 3 1\n1 1 1.0\n";
  const std::string path = spmvtest::write_temp("sb_nobanner.mtx", body);
  EXPECT_THROW(read_matrix_market<double>(path), std::exception);
}

TEST(MtxIo, RejectsOutOfRangeIndex) {
  const std::string body =
      "%%MatrixMarket matrix coordinate real general\n"
      "2 2 1\n"
      "3 1 1.0\n";
  const std::string path = spmvtest::write_temp("sb_range.mtx", body);
  EXPECT_THROW(read_matrix_market<double>(path), std::exception);
}
