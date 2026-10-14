#pragma once

// Numerical validation entry point: compares every CPU kernel against the
// SciPy float64 reference in three accumulation modes and writes accuracy.csv.

#include <string>

namespace spmvbench {

struct ValidationOptions {
  std::string data_dir = "data";
  std::string ref_dir = "data/ref";
  std::string out_csv = "results/accuracy.csv";
};

// Returns 0 on success.
int run_validation(const ValidationOptions& opts);

}  // namespace spmvbench
