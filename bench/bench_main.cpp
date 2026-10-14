// SpMVBench command-line driver.
//
// Subcommands (see --help):
//   --stats    <matrix|--all>   load a .mtx and print nonzero-distribution
//   stats
//   --auto     <matrix>         print stats, heuristic choice, and oracle table
//   --validate                  numerical validation vs the SciPy reference
//   --bench                     benchmark harness (GFLOP/s, bandwidth,
//   speedups)
//   --help                      print this message

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/select.hpp"
#include "spmvbench/spmv_cpu.hpp"
#include "spmvbench/stats.hpp"
#include "spmvbench/timer.hpp"
#include "spmvbench/validate.hpp"

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
      "  %s --validate                  validate every kernel vs SciPy "
      "reference\n"
      "  %s --bench                     run the benchmark harness\n"
      "  %s --help                      print this message\n"
      "\n"
      "Matrices are Matrix Market (.mtx) files, typically under data/.\n",
      argv0, argv0, argv0, argv0, argv0, argv0);
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
// CPU SpMV kernel timing (shared by --auto and --bench)
// ---------------------------------------------------------------------------
struct KernelTiming {
  KernelId id;
  std::string name;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double gflops = 0.0;
  bool applicable = true;
};

template <typename T, typename Acc>
std::vector<KernelTiming> time_cpu_spmv(const spmvbench::CsrMatrix<T>& A,
                                        const std::vector<T>& x, int iters,
                                        int warmup, int repeats = 7) {
  const std::int64_t nnz = A.nnz;
  std::vector<Acc> y(A.m);
  std::vector<KernelTiming> out;
  auto gflops = [&](double ms) {
    return ms > 0 ? 2.0 * static_cast<double>(nnz) / (ms * 1e-3) / 1e9 : 0.0;
  };
  // Robust timing: take the best (minimum) mean over `repeats` independent
  // rounds, which suppresses OS scheduling and thermal noise on a laptop CPU.
  auto timeit = [&](auto&& fn) {
    for (int i = 0; i < warmup; ++i)
      fn();
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
      spmvbench::CpuTimer t;
      t.start();
      for (int i = 0; i < iters; ++i)
        fn();
      t.stop();
      best = std::min(best, t.ms() / std::max(1, iters));
    }
    return best;
  };
  auto push = [&](KernelId id, const char* name, double ms) {
    out.push_back({id, name, ms, ms, gflops(ms), true});
  };

  // Serial scalar.
  push(KernelId::CsrScalar, "csr_scalar", timeit([&] {
         spmvbench::spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data());
       }));
  // Row-parallel CSR (the vector-group analogue on CPU).
  push(KernelId::CsrVector, "csr_vector",
       timeit([&] { spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y.data()); }));
  // Load-balanced CSR.
  push(KernelId::CsrMerge, "csr_merge", timeit([&] {
         spmvbench::spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data());
       }));
  // ELL.
  {
    const auto ell = spmvbench::csr_to_ell(A);
    push(KernelId::Ell, "ell", timeit([&] {
           spmvbench::spmv_ell_cpu<T, Acc>(ell, x.data(), y.data());
         }));
  }
  // BCSR 4x4.
  {
    const auto b = spmvbench::csr_to_bcsr<T, 4, 4>(A);
    push(KernelId::Bcsr, "bcsr", timeit([&] {
           spmvbench::spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data());
         }));
  }
  return out;
}

std::vector<double> make_x_deterministic(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = d(rng);
  return x;
}

struct AutoResult {
  std::string name;
  spmvbench::RowStats stats;
  spmvbench::KernelChoice choice;
  std::vector<KernelTiming> timings;
  std::string best_name;
  double best_ms = 0.0;
  double chosen_ms = -1.0;  // -1 if the chosen kernel has no CPU implementation
  double slowdown = 1.0;
  bool hit = false;
};

AutoResult evaluate_matrix(const std::string& path, int iters, int warmup) {
  AutoResult r;
  r.name = base_name(path);
  spmvbench::CsrMatrix<double> A = spmvbench::read_matrix_market<double>(path);
  r.stats = spmvbench::compute_row_stats(A);
  r.choice = spmvbench::choose_spmv(r.stats);

  const std::vector<double> x = make_x_deterministic(A.n, 12345u);
  r.timings = time_cpu_spmv<double, double>(A, x, iters, warmup);

  r.best_ms = 1e300;
  for (const auto& t : r.timings) {
    if (t.mean_ms < r.best_ms) {
      r.best_ms = t.mean_ms;
      r.best_name = t.name;
    }
    if (t.id == r.choice.kernel)
      r.chosen_ms = t.mean_ms;
  }
  if (r.chosen_ms > 0) {
    r.slowdown = r.chosen_ms / r.best_ms;
    r.hit = (r.chosen_ms <= r.best_ms * (1.0 + 1e-6));
  }
  return r;
}

void print_auto_detail(const AutoResult& r) {
  std::printf("=== %s ===\n%s\n\n", r.name.c_str(),
              r.stats.to_string().c_str());
  std::printf("chosen kernel: %s (lanes=%d)\n",
              spmvbench::kernel_name(r.choice.kernel), r.choice.lanes);
  std::printf("reason: %s\n\n", r.choice.reason.c_str());

  std::printf("%-12s %10s %10s %8s\n", "kernel", "mean_ms", "GFLOP/s", "ratio");
  for (const auto& t : r.timings) {
    const bool is_choice = (t.id == r.choice.kernel);
    std::printf("%-12s %10.4f %10.3f %8.3f %s\n", t.name.c_str(), t.mean_ms,
                t.gflops, r.best_ms / t.mean_ms, is_choice ? "<- chosen" : "");
  }
  std::printf("\noracle best: %s (%.4f ms)\n", r.best_name.c_str(), r.best_ms);
  if (r.chosen_ms < 0) {
    std::printf(
        "heuristic chose %s which has no CPU implementation in this build\n",
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
  std::string path;
  for (const auto& a : args) {
    if (a == "--all")
      all = true;
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

  const int iters = 20, warmup = 3;
  std::vector<AutoResult> results;
  for (const auto& p : targets) {
    try {
      results.push_back(evaluate_matrix(p, iters, warmup));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }

  if (!all) {
    for (const auto& r : results)
      print_auto_detail(r);
  }

  std::printf("%-24s %-12s %-12s %10s %10s %6s %10s\n", "matrix", "chosen",
              "oracle_best", "chosen_ms", "best_ms", "hit", "slowdown");
  int hits = 0;
  double worst = 1.0;
  for (const auto& r : results) {
    const std::string chosen =
        (r.chosen_ms < 0) ? "n/a" : spmvbench::kernel_name(r.choice.kernel);
    std::printf("%-24s %-12s %-12s %10.4f %10.4f %6s %10.2f\n", r.name.c_str(),
                chosen.c_str(), r.best_name.c_str(), r.chosen_ms, r.best_ms,
                r.hit ? "yes" : "no", r.slowdown);
    if (r.hit)
      hits++;
    worst = std::max(worst, r.slowdown);
  }
  std::printf(
      "\nheuristic-vs-oracle: %d/%zu matrices picked the fastest kernel; "
      "worst slowdown %.2fx\n",
      hits, results.size(), worst);

  // Persist a traceable artifact.
  std::error_code ec;
  fs::create_directories("results", ec);
  std::ofstream csv("results/heuristic.csv");
  if (csv) {
    csv << "matrix,chosen,oracle_best,chosen_ms,best_ms,hit,slowdown\n";
    for (const auto& r : results) {
      const std::string chosen =
          (r.chosen_ms < 0) ? "n/a" : spmvbench::kernel_name(r.choice.kernel);
      csv << r.name << "," << chosen << "," << r.best_name << "," << r.chosen_ms
          << "," << r.best_ms << "," << (r.hit ? "1" : "0") << "," << r.slowdown
          << "\n";
    }
    std::printf("wrote results/heuristic.csv\n");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------
struct TimeStat {
  double mean = 0.0, min = 0.0, max = 0.0;
};

template <typename F>
TimeStat time_loop(F&& fn, int iters, int warmup, int repeats = 1) {
  for (int i = 0; i < warmup; ++i)
    fn();
  spmvbench::CpuTimer t;
  TimeStat s;
  s.min = 1e300;
  double best_round_mean = 1e300;
  double best_round_min = 1e300;
  double best_round_max = 0.0;
  for (int r = 0; r < repeats; ++r) {
    double sum = 0.0, rmin = 1e300, rmax = 0.0;
    for (int i = 0; i < iters; ++i) {
      t.start();
      fn();
      t.stop();
      const double ms = t.ms();
      sum += ms;
      rmin = std::min(rmin, ms);
      rmax = std::max(rmax, ms);
    }
    const double rmean = sum / std::max(1, iters);
    if (rmean < best_round_mean) {
      best_round_mean = rmean;
      best_round_min = rmin;
      best_round_max = rmax;
    }
  }
  s.mean = best_round_mean;
  s.min = best_round_min;
  s.max = best_round_max;
  return s;
}

struct BenchRow {
  std::string matrix, precision, kernel;
  double mean_ms = 0, min_ms = 0, max_ms = 0;
  double gflops = 0;
  double bw_lo = 0, bw_hi = 0;  // GB/s, compulsory / no-reuse
  double speedup = 0;           // vs single-thread CPU baseline
};

double gbs(double bytes, double ms) {
  return ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0;
}

template <typename T, typename Acc>
std::vector<BenchRow> bench_type(const std::string& name,
                                 const spmvbench::CsrMatrix<T>& A,
                                 const std::vector<T>& x, int iters, int warmup,
                                 const std::string& prec) {
  const std::int64_t nnz = A.nnz;
  const int m = A.m, n = A.n;
  std::vector<Acc> y(static_cast<std::size_t>(m));
  std::vector<BenchRow> rows;
  const double xbytes = sizeof(T);

  // Derived metrics use the minimum time (best case) so they are comparable
  // across runs on a noise-prone laptop CPU.
  auto add = [&](const std::string& kernel, const TimeStat& s, double flops,
                 double bytes_lo, double bytes_hi) {
    BenchRow r;
    r.matrix = name;
    r.precision = prec;
    r.kernel = kernel;
    r.mean_ms = s.mean;
    r.min_ms = s.min;
    r.max_ms = s.max;
    r.gflops = s.min > 0 ? flops / (s.min * 1e-3) / 1e9 : 0.0;
    r.bw_lo = gbs(bytes_lo, s.min);
    r.bw_hi = gbs(bytes_hi, s.min);
    rows.push_back(r);
  };

  // Compulsory/noreuse byte models for CSR.
  const double csr_v = static_cast<double>(nnz) * sizeof(T);
  const double csr_c = static_cast<double>(nnz) * sizeof(Index);
  const double csr_r = static_cast<double>(m + 1) * sizeof(Index);
  const double csr_lo = csr_v + csr_c + csr_r + n * xbytes + m * sizeof(Acc);
  const double csr_hi = csr_v + csr_c + csr_r + nnz * xbytes + m * sizeof(Acc);

  // Baseline: single-threaded scalar CSR.
  const TimeStat base = time_loop(
      [&] { spmvbench::spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data()); },
      iters, warmup);
  add("csr_scalar", base, 2.0 * nnz, csr_lo, csr_hi);

  const TimeStat vec =
      time_loop([&] { spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y.data()); },
                iters, warmup);
  add("csr_vector", vec, 2.0 * nnz, csr_lo, csr_hi);

  const TimeStat mer = time_loop(
      [&] { spmvbench::spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data()); },
      iters, warmup);
  add("csr_merge", mer, 2.0 * nnz, csr_lo, csr_hi);

  {
    const auto ell = spmvbench::csr_to_ell(A);
    const double pn = static_cast<double>(ell.padded_nnz);
    const double v = pn * sizeof(T), c = pn * sizeof(Index);
    const double lo = v + c + n * xbytes + m * sizeof(Acc);
    const double hi = v + c + pn * xbytes + m * sizeof(Acc);
    const TimeStat s = time_loop(
        [&] { spmvbench::spmv_ell_cpu<T, Acc>(ell, x.data(), y.data()); },
        iters, warmup);
    add("ell", s, 2.0 * nnz, lo, hi);
  }
  {
    const auto b = spmvbench::csr_to_bcsr<T, 4, 4>(A);
    const double bn = static_cast<double>(b.block_nnz);
    const double v = bn * 16 * sizeof(T), c = bn * sizeof(Index);
    const double r = static_cast<double>(b.mb + 1) * sizeof(Index);
    const double lo = v + c + r + n * xbytes + m * sizeof(Acc);
    const double hi = v + c + r + bn * 16 * xbytes + m * sizeof(Acc);
    const TimeStat s = time_loop(
        [&] { spmvbench::spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data()); }, iters,
        warmup);
    add("bcsr", s, 2.0 * nnz, lo, hi);
  }
  // SpMM with k = 32.
  {
    const int k = 32;
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<T> B(static_cast<std::size_t>(n) * k);
    for (auto& v : B)
      v = static_cast<T>(d(rng));
    std::vector<Acc> Y(static_cast<std::size_t>(m) * k);
    const double bbytes = static_cast<double>(n) * k * sizeof(T);
    const double ybytes = static_cast<double>(m) * k * sizeof(Acc);
    const double lo = csr_v + csr_c + csr_r + bbytes + ybytes;
    const double hi =
        csr_v + csr_c + csr_r + static_cast<double>(nnz) * k * xbytes + ybytes;
    const TimeStat s = time_loop(
        [&] { spmvbench::spmm_csr_cpu<T, Acc>(A, B.data(), Y.data(), k); },
        iters, warmup);
    add("spmm_csr_k32", s, 2.0 * nnz * k, lo, hi);
  }

  // Speedup vs the single-thread baseline (minimum-time based).
  for (auto& r : rows)
    r.speedup = r.min_ms > 0 ? base.min / r.min_ms : 0.0;
  return rows;
}

int cmd_bench(const std::vector<std::string>& args) {
  int iters = 50, warmup = 5;
  std::string only;
  for (std::size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--iters" && i + 1 < args.size())
      iters = std::stoi(args[++i]);
    else if (args[i] == "--warmup" && i + 1 < args.size())
      warmup = std::stoi(args[++i]);
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

  std::vector<BenchRow> all;
  for (const auto& p : targets) {
    const std::string name = base_name(p);
    std::fprintf(stderr, "[bench] %s ...\n", name.c_str());
    spmvbench::CsrMatrix<double> Ad = spmvbench::read_matrix_market<double>(p);
    const std::vector<double> xd = make_x_deterministic(Ad.n, 999u);

    // fp64
    for (auto& r :
         bench_type<double, double>(name, Ad, xd, iters, warmup, "fp64"))
      all.push_back(r);
    // fp32
    spmvbench::CsrMatrix<float> Af = spmvbench::read_matrix_market<float>(p);
    std::vector<float> xf(xd.begin(), xd.end());
    for (auto& r :
         bench_type<float, float>(name, Af, xf, iters, warmup, "fp32"))
      all.push_back(r);
  }

  fs::create_directories("results");
  std::ofstream csv("results/bench.csv");
  csv << "matrix,precision,kernel,mean_ms,min_ms,max_ms,gflops,bw_lo_gbs,"
         "bw_hi_gbs,speedup_vs_st\n";
  for (const auto& r : all) {
    csv << r.matrix << "," << r.precision << "," << r.kernel << "," << r.mean_ms
        << "," << r.min_ms << "," << r.max_ms << "," << r.gflops << ","
        << r.bw_lo << "," << r.bw_hi << "," << r.speedup << "\n";
  }

  std::ofstream md("results/bench.md");
  md << "# SpMVBench benchmark results\n\n"
     << "Baseline for speedup is the single-threaded CPU `csr_scalar` kernel "
        "(no GPU on this machine).\n\n"
     << "| matrix | prec | kernel | mean ms | min ms | GFLOP/s | BW lo GB/s | "
        "BW hi GB/s | speedup |\n"
     << "|--------|------|--------|--------:|-------:|--------:|-----------:|"
        "----------:|--------:|\n";
  for (const auto& r : all) {
    md << "| " << r.matrix << " | " << r.precision << " | " << r.kernel << " | "
       << r.mean_ms << " | " << r.min_ms << " | " << r.gflops << " | "
       << r.bw_lo << " | " << r.bw_hi << " | " << r.speedup << " |\n";
  }

  std::printf(
      "[bench] wrote results/bench.csv (%zu rows) and results/bench.md\n",
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
