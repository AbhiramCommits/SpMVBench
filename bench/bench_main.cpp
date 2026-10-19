// SpMVBench command-line driver.
//
// Subcommands (see --help):
//   --stats    <matrix|--all>   load a .mtx and print nonzero-distribution
//   stats
//   --auto     <matrix|--all>   heuristic choice vs measured oracle
//   --validate                  numerical validation vs the SciPy reference
//   --bench                     benchmark harness (median/min, baselines)
//   --help                      print this message
//
// Benchmark methodology:
//   * CPU speedups are against the multithreaded OpenMP CSR kernel
//     (`csr_scalar_omp`). The single-threaded kernel is reported as context.
//     Intel MKL's mkl_sparse_d_mv is included when MKL is available.
//   * GPU speedups are against cuSPARSE SpMV / cuSPARSE SpMM for every GPU
//     kernel. Kernel time is measured with CUDA events (warmup + >=50 timed
//     iterations, median and min reported) and excludes host<->device copies;
//     those are reported in a separate transfer_ms column.
//   * Achieved bandwidth is the no-reuse traffic model divided by the median
//     kernel time, expressed as a percentage of the GPU's theoretical peak
//     (2 * memoryClockRate * busWidth/8), which is printed explicitly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/mkl_spmv.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/select.hpp"
#include "spmvbench/spmv_cpu.hpp"
#include "spmvbench/stats.hpp"
#include "spmvbench/timer.hpp"
#include "spmvbench/validate.hpp"

#ifdef SPMVBENCH_ENABLE_CUDA
#include "spmvbench/spmm_cuda.cuh"
#include "spmvbench/spmv_cuda.cuh"
#endif

namespace fs = std::filesystem;

namespace {

using spmvbench::Index;
using spmvbench::KernelId;

void print_help(const char* argv0) {
  std::printf(
      "SpMVBench - sparse linear algebra kernels, benchmarks and error "
      "analysis\n"
      "\n"
      "Usage:\n"
      "  %s --stats <matrix.mtx>        print nonzero-distribution statistics\n"
      "  %s --stats --all               stats for every matrix in data/\n"
      "  %s --auto  <matrix.mtx>        heuristic selection vs measured "
      "oracle\n"
      "  %s --auto  --all [--cpu]       oracle for every matrix\n"
      "  %s --validate                  validate every kernel vs SciPy "
      "reference\n"
      "  %s --bench [--iters N] [--cpu-only|--gpu-only]\n"
      "  %s --help                      print this message\n"
      "\n"
      "Benchmark baselines: OpenMP CSR on CPU; cuSPARSE SpMV/SpMM on GPU.\n",
      argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

std::string data_dir() {
  const char* env = std::getenv("SPMVBENCH_DATA");
  if (env && *env)
    return env;
  return "data";
}

std::vector<std::string> list_matrices() {
  std::vector<std::string> out;
  const std::string dir = data_dir();
  if (!fs::exists(dir))
    return out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".mtx") {
      out.push_back(e.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string base_name(const std::string& path) {
  return fs::path(path).stem().string();
}

int cmd_stats(const std::vector<std::string>& args) {
  bool all = false;
  std::string path;
  for (const auto& a : args) {
    if (a == "--all")
      all = true;
    else
      path = a;
  }

  std::vector<std::string> targets;
  if (all) {
    targets = list_matrices();
    if (targets.empty()) {
      std::fprintf(stderr, "no .mtx files found in %s\n", data_dir().c_str());
      return 1;
    }
  } else if (!path.empty()) {
    targets = {path};
  } else {
    std::fprintf(stderr, "--stats requires a matrix path or --all\n");
    return 1;
  }

  for (const auto& p : targets) {
    spmvbench::CsrMatrix<double> A;
    try {
      A = spmvbench::read_matrix_market<double>(p);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
    const spmvbench::RowStats s = spmvbench::compute_row_stats(A);
    std::printf("=== %s ===\n%s\n\n", base_name(p).c_str(),
                s.to_string().c_str());
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Heuristic vs oracle
// ---------------------------------------------------------------------------
struct KernelTiming {
  KernelId id;
  std::string name;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double median_ms = 0.0;
  double gflops = 0.0;
  bool applicable = true;
};

std::vector<double> make_x_deterministic(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = d(rng);
  return x;
}

template <typename T, typename Acc>
std::vector<KernelTiming> time_cpu_spmv(const spmvbench::CsrMatrix<T>& A,
                                        const std::vector<T>& x, int iters,
                                        int warmup) {
  const std::int64_t nnz = A.nnz;
  std::vector<Acc> y(A.m);
  std::vector<KernelTiming> out;
  auto gflops = [&](double ms) {
    return ms > 0 ? 2.0 * static_cast<double>(nnz) / (ms * 1e-3) / 1e9 : 0.0;
  };
  auto timeit = [&](auto&& fn) {
    for (int i = 0; i < warmup; ++i)
      fn();
    spmvbench::CpuTimer t;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      t.start();
      fn();
      t.stop();
      samples.push_back(t.ms());
    }
    return spmvbench::summarize(samples);
  };
  auto push = [&](KernelId id, const char* name,
                  const spmvbench::TimingSummary& s) {
    out.push_back({id, name, s.mean_ms, s.min_ms, s.median_ms,
                   gflops(s.median_ms), true});
  };

  push(KernelId::CsrScalar, "csr_scalar_st", timeit([&] {
         spmvbench::spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data());
       }));
  push(KernelId::CsrVector, "csr_scalar_omp",
       timeit([&] { spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y.data()); }));
  push(KernelId::CsrMerge, "csr_merge_omp", timeit([&] {
         spmvbench::spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data());
       }));
  {
    const auto ell = spmvbench::csr_to_ell(A);
    push(KernelId::Ell, "ell_omp", timeit([&] {
           spmvbench::spmv_ell_cpu<T, Acc>(ell, x.data(), y.data());
         }));
  }
  {
    const auto b = spmvbench::csr_to_bcsr<T, 4, 4>(A);
    push(KernelId::Bcsr, "bcsr_omp", timeit([&] {
           spmvbench::spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data());
         }));
  }
  return out;
}

struct AutoResult {
  std::string name;
  std::string device;
  spmvbench::RowStats stats;
  spmvbench::KernelChoice choice;
  std::vector<KernelTiming> timings;
  std::string best_name;
  double best_ms = 0.0;
  double chosen_ms = -1.0;
  double slowdown = 1.0;
  bool hit = false;
};

AutoResult finalize_auto(const std::string& name, const std::string& device,
                         const spmvbench::RowStats& stats,
                         const spmvbench::KernelChoice& choice,
                         std::vector<KernelTiming> timings) {
  AutoResult r;
  r.name = name;
  r.device = device;
  r.stats = stats;
  r.choice = choice;
  r.timings = std::move(timings);
  r.best_ms = 1e300;
  for (const auto& t : r.timings) {
    if (t.median_ms < r.best_ms) {
      r.best_ms = t.median_ms;
      r.best_name = t.name;
    }
    if (t.id == choice.kernel)
      r.chosen_ms = t.median_ms;
  }
  if (r.chosen_ms > 0 && r.best_ms > 0) {
    r.slowdown = r.chosen_ms / r.best_ms;
    r.hit = (r.chosen_ms <= r.best_ms * (1.0 + 1e-6));
  }
  return r;
}

AutoResult evaluate_matrix_cpu(const std::string& path, int iters, int warmup) {
  spmvbench::CsrMatrix<double> A = spmvbench::read_matrix_market<double>(path);
  const spmvbench::RowStats s = spmvbench::compute_row_stats(A);
  const spmvbench::KernelChoice choice = spmvbench::choose_spmv(s);
  const std::vector<double> x = make_x_deterministic(A.n, 12345u);
  auto timings = time_cpu_spmv<double, double>(A, x, iters, warmup);
  return finalize_auto(base_name(path), "cpu", s, choice, std::move(timings));
}

#ifdef SPMVBENCH_ENABLE_CUDA
AutoResult evaluate_matrix_gpu(const std::string& path, int iters, int warmup) {
  spmvbench::CsrMatrix<double> A = spmvbench::read_matrix_market<double>(path);
  const spmvbench::RowStats s = spmvbench::compute_row_stats(A);
  const spmvbench::KernelChoice choice = spmvbench::choose_spmv(s);
  const std::vector<double> x = make_x_deterministic(A.n, 12345u);

  spmvbench::SpmvRunner<double, double> runner(A);
  runner.set_x(x);

  std::vector<KernelTiming> timings;
  auto run = [&](KernelId id, const char* name, int lanes, int block) {
    const spmvbench::SpmvResult r = runner.run(id, iters, lanes, block, warmup);
    timings.push_back(
        {id, name, r.mean_ms, r.min_ms, r.median_ms,
         2.0 * static_cast<double>(A.nnz) / (r.median_ms * 1e-3) / 1e9, true});
  };
  run(KernelId::CsrScalar, "csr_scalar", 1, 1);
  run(KernelId::CsrVector, "csr_vector", choice.lanes, 1);
  run(KernelId::CsrWarp, "csr_warp", 1, 1);
  run(KernelId::CsrMerge, "csr_merge", 1, 1);
  run(KernelId::Ell, "ell", 1, 1);
  run(KernelId::Bcsr, "bcsr", 1, 4);
  return finalize_auto(base_name(path), "cuda", s, choice, std::move(timings));
}
#endif

void print_auto_detail(const AutoResult& r) {
  std::printf("=== %s [%s] ===\n%s\n\n", r.name.c_str(), r.device.c_str(),
              r.stats.to_string().c_str());
  std::printf("chosen kernel: %s (lanes=%d)\n",
              spmvbench::kernel_name(r.choice.kernel), r.choice.lanes);
  std::printf("reason: %s\n\n", r.choice.reason.c_str());

  std::printf("%-16s %10s %10s %10s %8s\n", "kernel", "median_ms", "min_ms",
              "GFLOP/s", "ratio");
  for (const auto& t : r.timings) {
    const bool is_choice = (t.id == r.choice.kernel);
    std::printf("%-16s %10.4f %10.4f %10.3f %8.3f %s\n", t.name.c_str(),
                t.median_ms, t.min_ms, t.gflops, r.best_ms / t.median_ms,
                is_choice ? "<- chosen" : "");
  }
  std::printf("\noracle best: %s (%.4f ms)\n", r.best_name.c_str(), r.best_ms);
  if (r.chosen_ms < 0) {
    std::printf(
        "heuristic chose %s which has no implementation in this build\n",
        spmvbench::kernel_name(r.choice.kernel));
  } else if (r.hit) {
    std::printf("heuristic picked the fastest kernel (hit)\n");
  } else {
    std::printf(
        "heuristic MISS: chose %s at %.4f ms, %.2fx slower than best (%s)\n",
        spmvbench::kernel_name(r.choice.kernel), r.chosen_ms, r.slowdown,
        r.best_name.c_str());
  }
}

int cmd_auto(const std::vector<std::string>& args) {
  bool all = false;
  bool force_cpu = false;
  std::string path;
  for (const auto& a : args) {
    if (a == "--all")
      all = true;
    else if (a == "--cpu")
      force_cpu = true;
    else
      path = a;
  }
  if (!all && path.empty()) {
    std::fprintf(stderr, "--auto requires a matrix path or --all\n");
    return 1;
  }

  std::vector<std::string> targets =
      all ? list_matrices() : std::vector<std::string>{path};
  if (targets.empty()) {
    std::fprintf(stderr, "no matrices found\n");
    return 1;
  }

  bool use_gpu = false;
#ifdef SPMVBENCH_ENABLE_CUDA
  use_gpu = !force_cpu && spmvbench::gpu_available();
#endif

  const int iters = 50, warmup = 10;
  std::vector<AutoResult> results;
  for (const auto& p : targets) {
    try {
#ifdef SPMVBENCH_ENABLE_CUDA
      if (use_gpu) {
        results.push_back(evaluate_matrix_gpu(p, iters, warmup));
        continue;
      }
#endif
      results.push_back(evaluate_matrix_cpu(p, iters, warmup));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }

  if (!all) {
    for (const auto& r : results)
      print_auto_detail(r);
  }

  std::printf("%-24s %-6s %-14s %-14s %10s %10s %6s %10s\n", "matrix", "device",
              "chosen", "oracle_best", "chosen_ms", "best_ms", "hit",
              "slowdown");
  int hits = 0;
  double worst = 1.0;
  for (const auto& r : results) {
    const std::string chosen =
        (r.chosen_ms < 0) ? "n/a" : spmvbench::kernel_name(r.choice.kernel);
    std::printf("%-24s %-6s %-14s %-14s %10.4f %10.4f %6s %10.2f\n",
                r.name.c_str(), r.device.c_str(), chosen.c_str(),
                r.best_name.c_str(), r.chosen_ms, r.best_ms,
                r.hit ? "yes" : "no", r.slowdown);
    if (r.hit)
      hits++;
    worst = std::max(worst, r.slowdown);
  }
  std::printf(
      "\nheuristic-vs-oracle: %d/%zu matrices picked the fastest kernel; "
      "worst slowdown %.2fx\n",
      hits, results.size(), worst);

  std::error_code ec;
  fs::create_directories("results", ec);
  std::ofstream csv("results/heuristic.csv");
  if (csv) {
    csv << "matrix,device,chosen,oracle_best,chosen_ms,best_ms,hit,slowdown\n";
    for (const auto& r : results) {
      const std::string chosen =
          (r.chosen_ms < 0) ? "n/a" : spmvbench::kernel_name(r.choice.kernel);
      csv << r.name << "," << r.device << "," << chosen << "," << r.best_name
          << "," << r.chosen_ms << "," << r.best_ms << ","
          << (r.hit ? "1" : "0") << "," << r.slowdown << "\n";
    }
    std::printf("wrote results/heuristic.csv\n");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------
struct BenchRow {
  std::string device;  // "cpu" | "cuda"
  std::string matrix, precision, kernel;
  bool is_baseline = false;
  double mean_ms = 0, min_ms = 0, median_ms = 0, max_ms = 0;
  double gflops = 0;
  double bw_lo = 0, bw_hi = 0, bw_pct_peak = 0;  // GB/s and % of GPU peak
  double transfer_ms = 0;
  double speedup = 0;  // 0 for baseline rows (not printed)
  std::string baseline;
};

double gbs(double bytes, double ms) {
  return ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0;
}

struct ByteModel {
  double lo = 0.0;  // compulsory traffic (x/B read once)
  double hi = 0.0;  // no-reuse traffic (x/B fetched per nonzero)
};

template <typename T>
ByteModel csr_bytes(const spmvbench::CsrMatrix<T>& A) {
  const double v = static_cast<double>(A.nnz) * sizeof(T);
  const double c = static_cast<double>(A.nnz) * sizeof(Index);
  const double r = static_cast<double>(A.m + 1) * sizeof(Index);
  const double y = static_cast<double>(A.m) * sizeof(T);
  const double x = static_cast<double>(A.n) * sizeof(T);
  return {v + c + r + x + y,
          v + c + r + static_cast<double>(A.nnz) * sizeof(T) + y};
}

double pct_peak(double bw_gbs, double peak_gbs) {
  return (peak_gbs > 0) ? 100.0 * bw_gbs / peak_gbs : 0.0;
}

// ---- CPU ------------------------------------------------------------------
template <typename T, typename Acc>
std::vector<BenchRow> bench_cpu_type(const std::string& name,
                                     const spmvbench::CsrMatrix<T>& A,
                                     const std::vector<T>& x, int iters,
                                     int warmup, const std::string& prec) {
  const std::int64_t nnz = A.nnz;
  const int m = A.m, n = A.n;
  std::vector<Acc> y(static_cast<std::size_t>(m));
  std::vector<BenchRow> rows;

  auto run = [&](auto&& fn) {
    for (int i = 0; i < warmup; ++i)
      fn();
    spmvbench::CpuTimer t;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      t.start();
      fn();
      t.stop();
      samples.push_back(t.ms());
    }
    return spmvbench::summarize(samples);
  };

  auto add = [&](const std::string& kernel, const spmvbench::TimingSummary& s,
                 double flops, const ByteModel& bm, bool baseline) {
    BenchRow r;
    r.device = "cpu";
    r.matrix = name;
    r.precision = prec;
    r.kernel = kernel;
    r.is_baseline = baseline;
    r.mean_ms = s.mean_ms;
    r.min_ms = s.min_ms;
    r.median_ms = s.median_ms;
    r.max_ms = s.max_ms;
    r.transfer_ms = 0.0;
    r.gflops = s.median_ms > 0 ? flops / (s.median_ms * 1e-3) / 1e9 : 0.0;
    r.bw_lo = gbs(bm.lo, s.median_ms);
    r.bw_hi = gbs(bm.hi, s.median_ms);
    r.bw_pct_peak = 0.0;  // no GPU peak on the CPU path
    rows.push_back(r);
  };

  const ByteModel bm = csr_bytes(A);
  // Multithreaded OpenMP CSR: the CPU baseline.
  const auto base =
      run([&] { spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y.data()); });
  add("csr_scalar_omp", base, 2.0 * nnz, bm, true);
  // Single-threaded context.
  add("csr_scalar_1t", run([&] {
        spmvbench::spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data());
      }),
      2.0 * nnz, bm, false);
  add("csr_merge_omp", run([&] {
        spmvbench::spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data());
      }),
      2.0 * nnz, bm, false);
  {
    const auto ell = spmvbench::csr_to_ell(A);
    const ByteModel eb = [&] {
      const double pn = static_cast<double>(ell.padded_nnz);
      const double v = pn * sizeof(T), c = pn * sizeof(Index);
      const double ybytes = static_cast<double>(m) * sizeof(Acc);
      return ByteModel{v + c + static_cast<double>(n) * sizeof(T) + ybytes,
                       v + c + pn * sizeof(T) + ybytes};
    }();
    add("ell_omp",
        run([&] { spmvbench::spmv_ell_cpu<T, Acc>(ell, x.data(), y.data()); }),
        2.0 * nnz, eb, false);
  }
  {
    const auto b = spmvbench::csr_to_bcsr<T, 4, 4>(A);
    const ByteModel bb = [&] {
      const double bn = static_cast<double>(b.block_nnz);
      const double v = bn * 16 * sizeof(T), c = bn * sizeof(Index);
      const double r = static_cast<double>(b.mb + 1) * sizeof(Index);
      const double ybytes = static_cast<double>(m) * sizeof(Acc);
      return ByteModel{v + c + r + static_cast<double>(n) * sizeof(T) + ybytes,
                       v + c + r + bn * 16 * sizeof(T) + ybytes};
    }();
    add("bcsr_omp",
        run([&] { spmvbench::spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data()); }),
        2.0 * nnz, bb, false);
  }
  // Optional MKL fp64 baseline (mkl_sparse_d_mv).
  if constexpr (std::is_same<T, double>::value) {
    if (spmvbench::mkl_available()) {
      const auto ms =
          spmvbench::mkl_spmv_f64(A, x.data(), y.data(), iters, warmup);
      add("mkl_sparse_d_mv", ms, 2.0 * nnz, bm, false);
    }
  }
  // CPU SpMM: only one implementation, so it is a reference row, not a
  // self-comparison.
  {
    const int k = 32;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<T> B(static_cast<std::size_t>(n) * k);
    for (auto& v : B)
      v = static_cast<T>(d(rng));
    std::vector<Acc> Y(static_cast<std::size_t>(m) * k);
    const ByteModel sb = csr_bytes(A);
    const double bbytes = static_cast<double>(n) * k * sizeof(T);
    const double ybytes = static_cast<double>(m) * k * sizeof(Acc);
    const ByteModel sbm{
        sb.lo + bbytes + ybytes,
        sb.hi + static_cast<double>(nnz) * k * sizeof(T) + ybytes};
    add("spmm_csr_omp_k32",
        run([&] { spmvbench::spmm_csr_cpu<T, Acc>(A, B.data(), Y.data(), k); }),
        2.0 * nnz * k, sbm, true);
  }

  const double base_ms = base.median_ms;
  for (auto& r : rows) {
    if (r.is_baseline) {
      r.baseline = r.kernel;
      r.speedup = 0.0;  // omitted from speedup tables
    } else if (base_ms > 0 && r.median_ms > 0) {
      r.baseline = "csr_scalar_omp";
      r.speedup = base_ms / r.median_ms;
    }
  }
  return rows;
}

#ifdef SPMVBENCH_ENABLE_CUDA
// ---- GPU ------------------------------------------------------------------
struct GpuSpmvSpec {
  KernelId id;
  const char* name;
  int lanes;
  int block;
};

template <typename T, typename Acc>
std::vector<BenchRow> bench_gpu_type(const std::string& name,
                                     const spmvbench::CsrMatrix<T>& A,
                                     const std::vector<T>& x, int iters,
                                     int warmup, const std::string& prec,
                                     const spmvbench::GpuInfo& gpu) {
  const std::int64_t nnz = A.nnz;
  const int m = A.m, n = A.n;
  std::vector<BenchRow> rows;

  auto add = [&](const std::string& kernel, bool is_baseline,
                 const spmvbench::SpmvResult& r, double flops,
                 const ByteModel& bm) {
    BenchRow row;
    row.device = "cuda";
    row.matrix = name;
    row.precision = prec;
    row.kernel = kernel;
    row.is_baseline = is_baseline;
    row.mean_ms = r.mean_ms;
    row.min_ms = r.min_ms;
    row.median_ms = r.median_ms;
    row.max_ms = r.max_ms;
    row.transfer_ms = r.transfer_ms;
    row.gflops = r.median_ms > 0 ? flops / (r.median_ms * 1e-3) / 1e9 : 0.0;
    row.bw_lo = gbs(bm.lo, r.median_ms);
    row.bw_hi = gbs(bm.hi, r.median_ms);
    row.bw_pct_peak = pct_peak(row.bw_hi, gpu.peak_gbs);
    row.baseline = "cusparse_spmv";
    rows.push_back(row);
  };

  spmvbench::SpmvRunner<T, Acc> runner(A);
  runner.set_x(x);

  // cuSPARSE SpMV baseline.
  std::vector<T> y_cs;
  spmvbench::SpmvResult cs;
  if constexpr (std::is_same<T, float>::value) {
    cs = spmvbench::cusparse_spmv_csr_f32(A, x, y_cs, iters, warmup);
  } else {
    cs = spmvbench::cusparse_spmv_csr_f64(A, x, y_cs, iters, warmup);
  }
  const ByteModel bm = csr_bytes(A);
  add("cusparse_spmv", true, cs, 2.0 * nnz, bm);

  const GpuSpmvSpec specs[] = {
      {KernelId::CsrScalar, "csr_scalar", 1, 1},
      {KernelId::CsrVector, "csr_vector_l4", 4, 1},
      {KernelId::CsrWarp, "csr_warp", 1, 1},
      {KernelId::CsrMerge, "csr_merge", 1, 1},
      {KernelId::Ell, "ell", 1, 1},
      {KernelId::Bcsr, "bcsr_4x4", 1, 4},
  };
  for (const auto& spec : specs) {
    ByteModel m2 = bm;
    if (spec.id == KernelId::Ell) {
      const double pn = static_cast<double>(runner.ell_padded_nnz());
      const double v = pn * sizeof(T), c = pn * sizeof(Index);
      const double ybytes = static_cast<double>(m) * sizeof(Acc);
      m2 = ByteModel{v + c + static_cast<double>(n) * sizeof(T) + ybytes,
                     v + c + pn * sizeof(T) + ybytes};
    } else if (spec.id == KernelId::Bcsr) {
      const double bn = static_cast<double>(runner.bcsr_block_nnz(spec.block));
      const double R = spec.block, C = spec.block;
      const double v = bn * R * C * sizeof(T), c = bn * sizeof(Index);
      const double r =
          static_cast<double>((A.m + spec.block - 1) / spec.block + 1) *
          sizeof(Index);
      const double ybytes = static_cast<double>(m) * sizeof(Acc);
      m2 = ByteModel{v + c + r + static_cast<double>(n) * sizeof(T) + ybytes,
                     v + c + r + bn * R * C * sizeof(T) + ybytes};
    }
    const spmvbench::SpmvResult r =
        runner.run(spec.id, iters, spec.lanes, spec.block, warmup);
    add(spec.name, false, r, 2.0 * nnz, m2);
  }

  // GPU SpMM: tiled + bucketed, against cuSPARSE SpMM.
  {
    const int k = 32;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<T> B(static_cast<std::size_t>(n) * k);
    for (auto& v : B)
      v = static_cast<T>(d(rng));
    const double bbytes = static_cast<double>(n) * k * sizeof(T);
    const double ybytes = static_cast<double>(m) * k * sizeof(Acc);
    const ByteModel sbm{
        bm.lo + bbytes + ybytes,
        bm.hi + static_cast<double>(nnz) * k * sizeof(T) + ybytes};

    std::vector<T> Ycs;
    spmvbench::SpmmResult csp;
    if constexpr (std::is_same<T, float>::value) {
      csp = spmvbench::cusparse_spmm_f32(A, B, k, iters, warmup, Ycs);
    } else {
      csp = spmvbench::cusparse_spmm_f64(A, B, k, iters, warmup, Ycs);
    }
    BenchRow brow;
    brow.device = "cuda";
    brow.matrix = name;
    brow.precision = prec;
    brow.kernel = "cusparse_spmm";
    brow.is_baseline = true;
    brow.mean_ms = csp.mean_ms;
    brow.min_ms = csp.min_ms;
    brow.median_ms = csp.median_ms;
    brow.max_ms = csp.max_ms;
    brow.transfer_ms = csp.transfer_ms;
    brow.gflops =
        csp.median_ms > 0 ? 2.0 * nnz * k / (csp.median_ms * 1e-3) / 1e9 : 0.0;
    brow.bw_lo = gbs(sbm.lo, csp.median_ms);
    brow.bw_hi = gbs(sbm.hi, csp.median_ms);
    brow.bw_pct_peak = pct_peak(brow.bw_hi, gpu.peak_gbs);
    brow.baseline = "cusparse_spmm";
    rows.push_back(brow);

    for (int tile : {16, 32, 64}) {
      for (bool balanced : {false, true}) {
        std::vector<T> Y;
        spmvbench::SpmmResult sp;
        if constexpr (std::is_same<T, float>::value) {
          sp = spmvbench::spmm_csr_tiled_f32(A, B, k, tile, balanced, iters,
                                             warmup, Y);
        } else {
          sp = spmvbench::spmm_csr_tiled_f64(A, B, k, tile, balanced, iters,
                                             warmup, Y);
        }
        BenchRow r;
        r.device = "cuda";
        r.matrix = name;
        r.precision = prec;
        r.kernel = "spmm_tiled_t" + std::to_string(tile) +
                   (balanced ? "_bucketed" : "_uniform");
        r.is_baseline = false;
        r.mean_ms = sp.mean_ms;
        r.min_ms = sp.min_ms;
        r.median_ms = sp.median_ms;
        r.max_ms = sp.max_ms;
        r.transfer_ms = sp.transfer_ms;
        r.gflops = sp.median_ms > 0
                       ? 2.0 * nnz * k / (sp.median_ms * 1e-3) / 1e9
                       : 0.0;
        r.bw_lo = gbs(sbm.lo, sp.median_ms);
        r.bw_hi = gbs(sbm.hi, sp.median_ms);
        r.bw_pct_peak = pct_peak(r.bw_hi, gpu.peak_gbs);
        r.baseline = "cusparse_spmm";
        r.speedup = (csp.median_ms > 0 && sp.median_ms > 0)
                        ? csp.median_ms / sp.median_ms
                        : 0.0;
        rows.push_back(r);
      }
    }
  }

  const double cs_ms = cs.median_ms;
  for (auto& r : rows) {
    if (r.is_baseline)
      continue;
    if (r.baseline == "cusparse_spmv" && cs_ms > 0 && r.median_ms > 0)
      r.speedup = cs_ms / r.median_ms;
  }
  return rows;
}
#endif  // SPMVBENCH_ENABLE_CUDA

int cmd_bench(const std::vector<std::string>& args) {
  int iters = 50, warmup = 10;
  bool cpu_only = false, gpu_only = false;
  std::string only;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--iters" && i + 1 < args.size())
      iters = std::stoi(args[++i]);
    else if (args[i] == "--warmup" && i + 1 < args.size())
      warmup = std::stoi(args[++i]);
    else if (args[i] == "--cpu-only")
      cpu_only = true;
    else if (args[i] == "--gpu-only")
      gpu_only = true;
    else
      only = args[i];
  }

  std::vector<std::string> targets = list_matrices();
  if (!only.empty())
    targets = {only};
  if (targets.empty()) {
    std::fprintf(stderr, "no matrices found in %s\n", data_dir().c_str());
    return 1;
  }

  bool use_gpu = false;
  std::string gpu_name = "cpu-only";
  int gpu_major = 0, gpu_minor = 0;
  double gpu_peak = 0.0;
#ifdef SPMVBENCH_ENABLE_CUDA
  use_gpu = !cpu_only && spmvbench::gpu_available();
  if (use_gpu) {
    const spmvbench::GpuInfo g = spmvbench::query_gpu();
    gpu_name = g.name;
    gpu_major = g.major;
    gpu_minor = g.minor;
    gpu_peak = g.peak_gbs;
  }
#endif
  if (gpu_only && !use_gpu) {
    std::fprintf(stderr, "no CUDA GPU available for --gpu-only\n");
    return 1;
  }

  if (use_gpu) {
    std::printf(
        "[bench] GPU: %s (sm_%d%d), assumed theoretical peak HBM bandwidth = "
        "%.1f GB/s\n",
        gpu_name.c_str(), gpu_major, gpu_minor, gpu_peak);
    std::printf(
        "[bench] peak is the vendor spec value assumed from the device name; "
        "bandwidth %% is the no-reuse model vs this peak\n");
  }
  std::printf("[bench] MKL baseline: %s\n",
              spmvbench::mkl_available() ? "available" : "not available");

  std::vector<BenchRow> all;
  for (const auto& p : targets) {
    const std::string name = base_name(p);
    std::fprintf(stderr, "[bench] %s ...\n", name.c_str());
    spmvbench::CsrMatrix<double> Ad = spmvbench::read_matrix_market<double>(p);
    const std::vector<double> xd = make_x_deterministic(Ad.n, 999u);

    if (!gpu_only) {
      for (auto& r :
           bench_cpu_type<double, double>(name, Ad, xd, iters, warmup, "fp64"))
        all.push_back(r);
      spmvbench::CsrMatrix<float> Af = spmvbench::read_matrix_market<float>(p);
      std::vector<float> xf(xd.begin(), xd.end());
      for (auto& r :
           bench_cpu_type<float, float>(name, Af, xf, iters, warmup, "fp32"))
        all.push_back(r);
    }
#ifdef SPMVBENCH_ENABLE_CUDA
    if (use_gpu) {
      const spmvbench::GpuInfo gpu{gpu_name, gpu_major, gpu_minor, gpu_peak,
                                   gpu_peak > 0.0};
      for (auto& r : bench_gpu_type<double, double>(name, Ad, xd, iters, warmup,
                                                    "fp64", gpu))
        all.push_back(r);
      spmvbench::CsrMatrix<float> Af = spmvbench::read_matrix_market<float>(p);
      std::vector<float> xf(xd.begin(), xd.end());
      for (auto& r : bench_gpu_type<float, float>(name, Af, xf, iters, warmup,
                                                  "fp32", gpu))
        all.push_back(r);
    }
#endif
  }

  fs::create_directories("results");
  std::ofstream csv("results/bench.csv");
  csv << "device,matrix,precision,kernel,is_baseline,median_ms,min_ms,mean_ms,"
         "max_ms,gflops,bw_lo_gbs,bw_hi_gbs,bw_pct_peak,transfer_ms,speedup,"
         "baseline\n";
  for (const auto& r : all) {
    csv << r.device << "," << r.matrix << "," << r.precision << "," << r.kernel
        << "," << (r.is_baseline ? 1 : 0) << "," << r.median_ms << ","
        << r.min_ms << "," << r.mean_ms << "," << r.max_ms << "," << r.gflops
        << "," << r.bw_lo << "," << r.bw_hi << "," << r.bw_pct_peak << ","
        << r.transfer_ms << ",";
    if (!r.is_baseline)
      csv << r.speedup;
    csv << "," << r.baseline << "\n";
  }

  std::ofstream info("results/gpu_info.txt");
  if (use_gpu) {
    info << "name=" << gpu_name << "\n"
         << "compute_capability=" << gpu_major << "." << gpu_minor << "\n"
         << "assumed_peak_gbs=" << gpu_peak << "\n"
         << "peak_source=vendor spec assumed from device name\n"
         << "bandwidth_model=no-reuse (x/B fetched per nonzero)\n"
         << "mkl_available=" << (spmvbench::mkl_available() ? 1 : 0) << "\n";
  } else {
    info << "name=cpu-only\nassumed_peak_gbs=0\n"
         << "mkl_available=" << (spmvbench::mkl_available() ? 1 : 0) << "\n";
  }

  std::ofstream md("results/bench.md");
  md << "# SpMVBench benchmark results\n\n";
  if (use_gpu) {
    md << "GPU: **" << gpu_name << "** (sm_" << gpu_major << gpu_minor
       << "). Assumed theoretical peak bandwidth: **" << gpu_peak
       << " GB/s** (vendor spec assumed from the device name). "
          "Achieved bandwidth uses the no-reuse traffic model.\n\n";
  } else {
    md << "CPU-only run. GPU baselines are not available.\n\n";
  }
  md << "MKL `mkl_sparse_d_mv` baseline: "
     << (spmvbench::mkl_available() ? "available" : "not available") << ".\n\n";
  md << "Median and min over " << iters << " timed iterations (warmup "
     << warmup
     << "). Kernel time excludes host<->device transfers; `transfer` is the "
        "separate H2D+D2H time.\n\n";

  // Baselines table (no self-comparison).
  md << "## Baselines\n\n"
     << "| device | matrix | prec | baseline | median ms | min ms | GFLOP/s | "
        "BW GB/s | % peak | transfer ms |\n"
     << "|--------|--------|------|----------|----------:|-------:|--------:|"
        "--------:|-------:|------------:|\n";
  for (const auto& r : all) {
    if (!r.is_baseline)
      continue;
    md << "| " << r.device << " | " << r.matrix << " | " << r.precision << " | "
       << r.kernel << " | " << r.median_ms << " | " << r.min_ms << " | "
       << r.gflops << " | " << r.bw_hi << " | " << r.bw_pct_peak << " | "
       << r.transfer_ms << " |\n";
  }

  md << "\n## Kernels (speedup vs baseline)\n\n"
     << "| device | matrix | prec | kernel | median ms | min ms | GFLOP/s | "
        "BW GB/s | % peak | transfer ms | speedup |\n"
     << "|--------|--------|------|--------|----------:|-------:|--------:|"
        "--------:|-------:|------------:|--------:|\n";
  for (const auto& r : all) {
    if (r.is_baseline)
      continue;
    md << "| " << r.device << " | " << r.matrix << " | " << r.precision << " | "
       << r.kernel << " | " << r.median_ms << " | " << r.min_ms << " | "
       << r.gflops << " | " << r.bw_hi << " | " << r.bw_pct_peak << " | "
       << r.transfer_ms << " | " << r.speedup << " |\n";
  }

  std::printf(
      "[bench] wrote results/bench.csv (%zu rows), results/bench.md, "
      "results/gpu_info.txt\n",
      all.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);

  if (args.empty() || args[0] == "--help" || args[0] == "-h") {
    print_help(argv[0]);
    return args.empty() ? 1 : 0;
  }

  const std::string cmd = args[0];
  args.erase(args.begin());

  if (cmd == "--stats")
    return cmd_stats(args);
  if (cmd == "--auto")
    return cmd_auto(args);
  if (cmd == "--validate") {
    spmvbench::ValidationOptions opts;
    opts.data_dir = data_dir();
    opts.ref_dir = data_dir() + "/ref";
    return spmvbench::run_validation(opts);
  }
  if (cmd == "--bench")
    return cmd_bench(args);

  std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
  print_help(argv[0]);
  return 1;
}
