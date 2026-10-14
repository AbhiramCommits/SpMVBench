#pragma once

// Nonzero-distribution-driven kernel selection.

#include <string>

#include "spmvbench/stats.hpp"

namespace spmvbench {

enum class KernelId : int {
  CsrScalar = 0,  // one thread per row, minimal orchestration
  CsrVector,      // sub-warp / vector width per row, shfl reduction
  CsrWarp,        // one full warp per row
  CsrMerge,       // load-balanced over nnz chunks (skewed rows)
  Ell,            // padded ELL
  Bcsr,           // blocked CSR
  Count
};

const char* kernel_name(KernelId id);

struct KernelChoice {
  KernelId kernel = KernelId::CsrVector;
  int lanes = 4;       // for CsrVector: sub-warp lanes per row
  int block_r = 1;     // for Bcsr
  int block_c = 1;     // for Bcsr
  std::string reason;  // human-readable justification
};

// Choose a SpMV kernel from row-distribution statistics using the explicit
// rules documented in src/select.cpp.
KernelChoice choose_spmv(const RowStats& stats);

}  // namespace spmvbench
