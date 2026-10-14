// SpMM correctness for k = 1, 15, 32, 64 vs a naive reference.

#include <gtest/gtest.h>

#include <vector>

#include "spmvbench/spmv_cpu.hpp"
#include "test_util.hpp"

namespace {

spmvbench::CsrMatrix<double> sample() {
  return spmvtest::make_csr(
      6, 7,
      {{{0, 1.0}, {2, 2.0}, {6, 3.0}},
       {{1, 4.0}},
       {},
       {{0, 5.0}, {3, 6.0}, {4, 7.0}, {6, 8.0}},
       {{2, 9.0}, {5, 1.0}},
       {{1, 2.0}, {2, 3.0}, {3, 4.0}, {4, 5.0}, {5, 6.0}, {6, 7.0}}});
}

std::vector<double> naive_spmm(const spmvbench::CsrMatrix<double>& A,
                               const std::vector<double>& B, int k) {
  const int m = A.m, n = A.n;
  std::vector<double> Y(static_cast<std::size_t>(m) * k, 0.0);
  for (int i = 0; i < m; ++i)
    for (spmvbench::Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      const int c = A.colind[p];
      for (int j = 0; j < k; ++j)
        Y[static_cast<std::size_t>(i) * k + j] += A.values[p] * B[static_cast<std::size_t>(c) * k + j];
    }
  return Y;
}

void run_k(int k) {
  const auto A = sample();
  std::vector<double> B(static_cast<std::size_t>(A.n) * k);
  for (std::size_t i = 0; i < B.size(); ++i) B[i] = (static_cast<int>(i % 11) - 5) * 0.5;
  const auto ref = naive_spmm(A, B, k);

  std::vector<double> Y(static_cast<std::size_t>(A.m) * k);
  spmvbench::spmm_csr_cpu<double, double>(A, B.data(), Y.data(), k);
  EXPECT_TRUE(spmvtest::approx_eq(Y, ref, 1e-12)) << "k=" << k;
}

}  // namespace

TEST(SpmmCorrectness, K1) { run_k(1); }
TEST(SpmmCorrectness, K15) { run_k(15); }
TEST(SpmmCorrectness, K32) { run_k(32); }
TEST(SpmmCorrectness, K64) { run_k(64); }

TEST(SpmmCorrectness, MixedPrecisionAccumulate) {
  const auto A = sample();
  const int k = 32;
  std::vector<float> B(static_cast<std::size_t>(A.n) * k, 0.25f);
  spmvbench::CsrMatrix<float> Af;
  Af.m = A.m;
  Af.n = A.n;
  Af.nnz = A.nnz;
  Af.rowptr = A.rowptr;
  Af.colind = A.colind;
  Af.values.assign(A.values.begin(), A.values.end());
  std::vector<double> Y(static_cast<std::size_t>(A.m) * k);
  spmvbench::spmm_csr_cpu<float, double>(Af, B.data(), Y.data(), k);
  for (double v : Y) EXPECT_GE(v, 0.0);
}
