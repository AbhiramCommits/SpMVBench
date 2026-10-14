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
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "spmvbench/csr.hpp"
#include "spmvbench/dtypes.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/select.hpp"
#include "spmvbench/stats.hpp"

namespace fs = std::filesystem;

namespace {

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
  // Prefer ./data relative to the current working directory.
  if (fs::exists("data")) return "data";
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
  if (cmd == "--auto") {
    std::fprintf(stderr, "--auto is not available yet in this build\n");
    return 2;
  }
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
