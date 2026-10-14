// SpMVBench command-line driver.
//
// Subcommands (see --help):
//   --stats    <matrix|--all>   load a .mtx and print nonzero-distribution stats
//   --auto     <matrix>         print stats, heuristic choice, and oracle table
//   --validate                  numerical validation vs the SciPy reference
//   --bench                     benchmark harness (GFLOP/s, bandwidth, speedups)
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

namespace fs = std::filesystem;

namespace {

using spmvbench::Index;
using spmvbench::KernelId;

void print_help(const char* argv0) {
  std::printf(
      "SpMVBench - sparse linear algebra kernels, benchmarks and error analysis\n"
      "\n"
      "Usage:\n"
      "  %s --stats <matrix.mtx>        print nonzero-distribution statistics\n"
      "  %s --stats --all               stats for every matrix in data/\n"
      "  %s --auto  <matrix.mtx>        heuristic selection vs measured oracle\n"
      "  %s --validate                  validate every kernel vs SciPy reference\n"
      "  %s --bench                     run the benchmark harness\n"
      "  %s --help                      print this message\n"
      "\n"
      "Matrices are Matrix Market (.mtx) files, typically under data/.\n",
      argv0, argv0, argv0, argv0, argv0, argv0);
}

std::string data_dir() {
  const char* env = std::getenv("SPMVBENCH_DATA");
  if (env && *env) return env;
  return "data";
}

std::vector<std::string> list_matrices() {
  std::vector<std::string> out;
  const std::string dir = data_dir();
  if (!fs::exists(dir)) return out;
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
    if (a == "--all") all = true;
    else path = a;
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
    std::printf("=== %s ===\n%s\n\n", base_name(p).c_str(), s.to_string().c_str());
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
                                        int warmup) {
  const std::int64_t nnz = A.nnz;
  std::vector<Acc> y(A.m);
  std::vector<KernelTiming> out;
  auto gflops = [&](double ms) {
    return ms > 0 ? 2.0 * static_cast<double>(nnz) / (ms * 1e-3) / 1e9 : 0.0;
  };
  auto timeit = [&](auto&& fn) {
    for (int i = 0; i < warmup; ++i) fn();
    spmvbench::CpuTimer t;
    double best = 1e300;
    t.start();
    for (int i = 0; i < iters; ++i) fn();
    t.stop();
    best = t.ms() / std::max(1, iters);
    return best;
  };
  auto push = [&](KernelId id, const char* name, double ms) {
    out.push_back({id, name, ms, ms, gflops(ms), true});
  };

  // Serial scalar.
  push(KernelId::CsrScalar, "csr_scalar",
       timeit([&] { spmvbench::spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data()); }));
  // Row-parallel CSR (the vector-group analogue on CPU).
  push(KernelId::CsrVector, "csr_vector",
       timeit([&] { spmvbench::spmv_csr_cpu<T, Acc>(A, x.data(), y.data()); }));
  // Load-balanced CSR.
  push(KernelId::CsrMerge, "csr_merge",
       timeit([&] { spmvbench::spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data()); }));
  // ELL.
  {
    const auto ell = spmvbench::csr_to_ell(A);
    push(KernelId::Ell, "ell",
         timeit([&] { spmvbench::spmv_ell_cpu<T, Acc>(ell, x.data(), y.data()); }));
  }
  // BCSR 4x4.
  {
    const auto b = spmvbench::csr_to_bcsr<T, 4, 4>(A);
    push(KernelId::Bcsr, "bcsr",
         timeit([&] { spmvbench::spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data()); }));
  }
  return out;
}

std::vector<double> make_x_deterministic(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) x[i] = d(rng);
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
    if (t.id == r.choice.kernel) r.chosen_ms = t.mean_ms;
  }
  if (r.chosen_ms > 0) {
    r.slowdown = r.chosen_ms / r.best_ms;
    r.hit = (r.chosen_ms <= r.best_ms * (1.0 + 1e-6));
  }
  return r;
}

void print_auto_detail(const AutoResult& r) {
  std::printf("=== %s ===\n%s\n\n", r.name.c_str(), r.stats.to_string().c_str());
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
    std::printf("heuristic chose %s which has no CPU implementation in this build\n",
                spmvbench::kernel_name(r.choice.kernel));
  } else if (r.hit) {
    std::printf("heuristic picked the fastest kernel (hit)\n");
  } else {
    std::printf("heuristic MISS: chose %s at %.4f ms, %.2fx slower than best (%s)\n",
                spmvbench::kernel_name(r.choice.kernel), r.chosen_ms, r.slowdown,
                r.best_name.c_str());
  }
}

int cmd_auto(const std::vector<std::string>& args) {
  bool all = false;
  std::string path;
  for (const auto& a : args) {
    if (a == "--all") all = true;
    else path = a;
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

  const int iters = 50, warmup = 5;
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
    for (const auto& r : results) print_auto_detail(r);
  }

  std::printf("%-24s %-12s %-12s %10s %10s %6s %10s\n", "matrix", "chosen",
              "oracle_best", "chosen_ms", "best_ms", "hit", "slowdown");
  int hits = 0;
  double worst = 1.0;
  for (const auto& r : results) {
    const std::string chosen = (r.chosen_ms < 0)
                                   ? "n/a"
                                   : spmvbench::kernel_name(r.choice.kernel);
    std::printf("%-24s %-12s %-12s %10.4f %10.4f %6s %10.2f\n", r.name.c_str(),
                chosen.c_str(), r.best_name.c_str(), r.chosen_ms, r.best_ms,
                r.hit ? "yes" : "no", r.slowdown);
    if (r.hit) hits++;
    worst = std::max(worst, r.slowdown);
  }
  std::printf("\nheuristic-vs-oracle: %d/%zu matrices picked the fastest kernel; "
              "worst slowdown %.2fx\n",
              hits, results.size(), worst);

  // Persist a traceable artifact.
  std::error_code ec;
  fs::create_directories("results", ec);
  std::ofstream csv("results/heuristic.csv");
  if (csv) {
    csv << "matrix,chosen,oracle_best,chosen_ms,best_ms,hit,slowdown\n";
    for (const auto& r : results) {
      const std::string chosen = (r.chosen_ms < 0)
                                     ? "n/a"
                                     : spmvbench::kernel_name(r.choice.kernel);
      csv << r.name << "," << chosen << "," << r.best_name << ","
          << r.chosen_ms << "," << r.best_ms << ","
          << (r.hit ? "1" : "0") << "," << r.slowdown << "\n";
    }
    std::printf("wrote results/heuristic.csv\n");
  }
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

  if (cmd == "--stats") return cmd_stats(args);
  if (cmd == "--auto") return cmd_auto(args);
  if (cmd == "--validate") {
    std::fprintf(stderr, "--validate is not available yet in this build\n");
    return 2;
  }
  if (cmd == "--bench") {
    std::fprintf(stderr, "--bench is not available yet in this build\n");
    return 2;
  }

  std::fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());
  print_help(argv[0]);
  return 1;
}
