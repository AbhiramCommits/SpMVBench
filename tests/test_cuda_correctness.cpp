// CUDA correctness: every GPU SpMV kernel and the tiled/bucketed SpMM kernels
// are compared against the CPU reference on every synthetic matrix, in fp32
// and fp64.
//
// Tolerances are derived from the classical gamma_n bound rather than tuned:
// for two different summation orders of the same n terms,
//     |s1 - s2| <= |s1 - exact| + |s2 - exact| <= 2 * gamma_n * sum_j |a_ij
//     x_j|
// with gamma_n = n*u / (1 - n*u) and u the unit roundoff of the accumulator.
// The 2x accounts for comparing two independent summation orders; no other
// slack is added.

#include <gtest/gtest.h>

#ifdef SPMVBENCH_ENABLE_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/spmm_cuda.cuh"
#include "spmvbench/spmv_cpu.hpp"
#include "spmvbench/spmv_cuda.cuh"

namespace {

std::string data_dir() {
  return std::string(SPMVBENCH_SOURCE_DIR) + "/data";
}

// Exactly the matrices produced by tools/gen_matrices.py.
const std::vector<std::string>& synthetic_matrices() {
  static const std::vector<std::string> names = {
      "banded_100k", "uniform_random", "powerlaw_skew", "diag_dominant_small",
      "tiny_3x3"};
  return names;
}

template <typename T>
double unit_roundoff() {
  return std::is_same<T, float>::value ? std::ldexp(1.0, -24)
                                       : std::ldexp(1.0, -53);
}

double gamma_n(int n, double u) {
  const double nu = static_cast<double>(n) * u;
  return nu / (1.0 - nu);
}

template <typename T>
std::vector<T> make_x(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> bit(0.0, 1.0);
  std::uniform_real_distribution<double> exponent(-6.0, 6.0);
  std::vector<T> x(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double sign = bit(rng) < 0.5 ? -1.0 : 1.0;
    x[static_cast<std::size_t>(i)] =
        static_cast<T>(sign * std::pow(10.0, exponent(rng)));
  }
  return x;
}

template <typename T, typename Acc>
void check_spmv(const std::string& matrix, const std::string& kernel,
                const spmvbench::CsrMatrix<T>& A, const std::vector<T>& x,
                const std::vector<Acc>& y_ref, const std::vector<Acc>& y_gpu) {
  ASSERT_EQ(y_ref.size(), y_gpu.size());
  ASSERT_EQ(y_ref.size(), static_cast<std::size_t>(A.m));
  const double u = unit_roundoff<Acc>();
  double worst_ratio = 0.0, worst_diff = 0.0, worst_bound = 0.0;
  int worst_row = -1;
  for (int i = 0; i < A.m; ++i) {
    double abs_sum = 0.0;
    for (spmvbench::Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
      abs_sum += std::fabs(static_cast<double>(A.values[p]) *
                           static_cast<double>(x[A.colind[p]]));
    }
    const int n = static_cast<int>(A.rowptr[i + 1] - A.rowptr[i]);
    const double bound = 2.0 * gamma_n(n, u) * abs_sum + 1e-30;
    const double diff = std::fabs(static_cast<double>(y_gpu[i]) -
                                  static_cast<double>(y_ref[i]));
    const double ratio = diff / bound;
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_row = i;
      worst_diff = diff;
      worst_bound = bound;
    }
  }
  EXPECT_LE(worst_ratio, 1.0)
      << "matrix=" << matrix << " kernel=" << kernel
      << " accumulator=" << spmvbench::dtype_name<Acc>() << " row=" << worst_row
      << " |gpu-cpu|=" << worst_diff << " bound=" << worst_bound
      << " ratio=" << worst_ratio;
}

template <typename T, typename Acc>
void run_spmv_matrix(const std::string& name) {
  const std::string path = data_dir() + "/" + name + ".mtx";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << path << " missing; run tools/gen_matrices.py";
  }
  const auto A = spmvbench::read_matrix_market<T>(path);
  const std::vector<T> x = make_x<T>(A.n, 1234u);

  std::vector<Acc> y_ref(static_cast<std::size_t>(A.m));
  spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y_ref.data());

  spmvbench::SpmvRunner<T, Acc> runner(A);
  runner.set_x(x);

  auto run_and_check = [&](spmvbench::KernelId id, int lanes, int block,
                           const std::string& label) {
    runner.run(id, /*iters=*/1, lanes, block, /*warmup=*/0);
    const std::vector<Acc> got = runner.result();
    check_spmv<T, Acc>(name, label, A, x, y_ref, got);
  };

  run_and_check(spmvbench::KernelId::CsrScalar, 1, 1, "csr_scalar");
  for (int lanes : {2, 4, 8, 16, 32}) {
    run_and_check(spmvbench::KernelId::CsrVector, lanes, 1,
                  "csr_vector_lanes" + std::to_string(lanes));
  }
  run_and_check(spmvbench::KernelId::CsrWarp, 1, 1, "csr_warp");
  run_and_check(spmvbench::KernelId::CsrMerge, 1, 1, "csr_merge");
  run_and_check(spmvbench::KernelId::Ell, 1, 1, "ell");
  run_and_check(spmvbench::KernelId::Bcsr, 1, 2, "bcsr_2x2");
  run_and_check(spmvbench::KernelId::Bcsr, 1, 4, "bcsr_4x4");
}

template <typename T, typename Acc>
void check_spmm(const std::string& matrix, const std::string& kernel,
                const spmvbench::CsrMatrix<T>& A, const std::vector<T>& B,
                int k, const std::vector<Acc>& Y_ref,
                const std::vector<Acc>& Y_gpu) {
  ASSERT_EQ(Y_ref.size(), Y_gpu.size());
  ASSERT_EQ(Y_ref.size(), static_cast<std::size_t>(A.m) * k);
  const double u = unit_roundoff<Acc>();
  double worst_ratio = 0.0, worst_diff = 0.0, worst_bound = 0.0;
  int worst_i = -1, worst_j = -1;
  for (int i = 0; i < A.m; ++i) {
    const int n = static_cast<int>(A.rowptr[i + 1] - A.rowptr[i]);
    const double g = gamma_n(n, u);
    for (int j = 0; j < k; ++j) {
      double abs_sum = 0.0;
      for (spmvbench::Index p = A.rowptr[i]; p < A.rowptr[i + 1]; ++p) {
        const int c = A.colind[p];
        abs_sum += std::fabs(
            static_cast<double>(A.values[p]) *
            static_cast<double>(B[static_cast<std::size_t>(c) * k + j]));
      }
      const double bound = 2.0 * g * abs_sum + 1e-30;
      const std::size_t idx = static_cast<std::size_t>(i) * k + j;
      const double diff = std::fabs(static_cast<double>(Y_gpu[idx]) -
                                    static_cast<double>(Y_ref[idx]));
      const double ratio = diff / bound;
      if (ratio > worst_ratio) {
        worst_ratio = ratio;
        worst_i = i;
        worst_j = j;
        worst_diff = diff;
        worst_bound = bound;
      }
    }
  }
  EXPECT_LE(worst_ratio, 1.0)
      << "matrix=" << matrix << " kernel=" << kernel
      << " accumulator=" << spmvbench::dtype_name<Acc>() << " at (" << worst_i
      << "," << worst_j << ") |gpu-cpu|=" << worst_diff
      << " bound=" << worst_bound << " ratio=" << worst_ratio;
}

template <typename T, typename Acc>
void run_spmm_matrix(const std::string& name, int k = 32) {
  const std::string path = data_dir() + "/" + name + ".mtx";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << path << " missing; run tools/gen_matrices.py";
  }
  const auto A = spmvbench::read_matrix_market<T>(path);

  std::mt19937 rng(4321);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> B(static_cast<std::size_t>(A.n) * k);
  for (auto& v : B)
    v = static_cast<T>(dist(rng));

  std::vector<Acc> Y_ref(static_cast<std::size_t>(A.m) * k);
  spmvbench::spmm_csr_cpu<T, Acc>(A, B.data(), Y_ref.data(), k);

  for (int tile : {16, 32, 64}) {
    for (bool balanced : {false, true}) {
      std::vector<T> Y;
      if constexpr (std::is_same<T, float>::value) {
        spmvbench::spmm_csr_tiled_f32(A, B, k, tile, balanced, /*iters=*/1,
                                      /*warmup=*/0, Y);
      } else {
        spmvbench::spmm_csr_tiled_f64(A, B, k, tile, balanced, /*iters=*/1,
                                      /*warmup=*/0, Y);
      }
      ASSERT_EQ(Y.size(), Y_ref.size());
      std::vector<Acc> Ygpu(Y.size());
      for (std::size_t i = 0; i < Y.size(); ++i)
        Ygpu[i] = static_cast<Acc>(Y[i]);
      check_spmm<T, Acc>(name,
                         "spmm_tiled_tile" + std::to_string(tile) +
                             (balanced ? "_bucketed" : "_uniform"),
                         A, B, k, Y_ref, Ygpu);
    }
  }
}

}  // namespace

class CudaSpmvCorrectness : public ::testing::TestWithParam<std::string> {};

TEST_P(CudaSpmvCorrectness, Fp32) {
  run_spmv_matrix<float, float>(GetParam());
}

TEST_P(CudaSpmvCorrectness, Fp64) {
  run_spmv_matrix<double, double>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SyntheticMatrices, CudaSpmvCorrectness,
                         ::testing::ValuesIn(synthetic_matrices()));

class CudaSpmmCorrectness : public ::testing::TestWithParam<std::string> {};

TEST_P(CudaSpmmCorrectness, Fp32) {
  run_spmm_matrix<float, float>(GetParam());
}

TEST_P(CudaSpmmCorrectness, Fp64) {
  run_spmm_matrix<double, double>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(SyntheticMatrices, CudaSpmmCorrectness,
                         ::testing::ValuesIn(synthetic_matrices()));

// Tile-boundary coverage: k smaller than, equal to, and larger than the tiles.
TEST(CudaSpmmEdgeK, TinyK1) {
  run_spmm_matrix<double, double>("tiny_3x3", 1);
}
TEST(CudaSpmmEdgeK, TinyK15) {
  run_spmm_matrix<double, double>("tiny_3x3", 15);
}
TEST(CudaSpmmEdgeK, TinyK64) {
  run_spmm_matrix<double, double>("tiny_3x3", 64);
}

#endif  // SPMVBENCH_ENABLE_CUDA
