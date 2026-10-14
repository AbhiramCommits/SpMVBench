// Kernel-selection heuristic.
//
// The rules below are deliberately explicit and ordered. The first matching
// rule wins; the order encodes the relative importance of the signals:
//
//   (0) nnz < 50000
//         -> CsrScalar. The problem is so small that block/thread orchestration
//            and format padding dominate the actual arithmetic; a plain scalar
//            sweep is fastest (and this is true on both CPU and GPU).
//
//   (1) mean_row_nnz < 4
//         -> CsrScalar. Rows are so short that per-row orchestration (a warp
//            or vector group per row) is pure overhead; one thread per row
//            with no cross-lane reduction is fastest.
//
//   (2) frac_dense_blocks_4x4 > 0.6
//         -> Bcsr (4x4). More than 60% of all possible 4x4 tiles are > 50%
//            populated, so the matrix is locally dense and blocked storage
//            amortizes index overhead and reuses loads across the block.
//
//   (3) cv > 1.5 || max/mean > 50
//         -> CsrMerge. Row lengths are highly non-uniform: a "one thread /
//            warp per row" decomposition leaves most of the machine idle
//            waiting on a few very long rows, so we bin work by nnz instead.
//
//   (4) 4 <= mean_row_nnz < 32 && cv < 0.5
//         -> CsrVector with lanes = next_pow2(ceil(mean_row_nnz)). Medium,
//            uniform rows map cleanly onto a sub-warp sized vector group and
//            avoid padding; this is checked before ELL.
//
//   (5) ell_padding_overhead < 1.2
//         -> Ell. Rows are too long for a single vector group but still nearly
//            uniform, so padding wastes < 20% of storage and the fully
//            coalesced column-major layout wins.
//
//   (6) otherwise
//         -> CsrVector with lanes = 4 (safe default for medium rows).
//
// Note: `mean_row_nnz < 4` and the dense-block rule are checked first because
// they describe *storage format* decisions that dominate any tiling choice.

#include "spmvbench/select.hpp"

#include <cmath>
#include <string>

namespace spmvbench {
namespace {

int next_pow2(int v) {
  if (v <= 1)
    return 1;
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

std::string fmt(double v) {
  std::string s = std::to_string(v);
  // Trim trailing zeros for readability in the reason string.
  if (s.find('.') != std::string::npos) {
    while (!s.empty() && s.back() == '0')
      s.pop_back();
    if (!s.empty() && s.back() == '.')
      s.pop_back();
  }
  return s;
}

}  // namespace

const char* kernel_name(KernelId id) {
  switch (id) {
    case KernelId::CsrScalar:
      return "csr_scalar";
    case KernelId::CsrVector:
      return "csr_vector";
    case KernelId::CsrWarp:
      return "csr_warp";
    case KernelId::CsrMerge:
      return "csr_merge";
    case KernelId::Ell:
      return "ell";
    case KernelId::Bcsr:
      return "bcsr";
    case KernelId::Count:
      break;
  }
  return "unknown";
}

KernelChoice choose_spmv(const RowStats& s) {
  KernelChoice c;

  // (0) Tiny problems: orchestration dominates.
  if (s.nnz < 50000) {
    c.kernel = KernelId::CsrScalar;
    c.lanes = 1;
    c.reason =
        "nnz=" + std::to_string(s.nnz) +
        " < 50000: problem too small for parallel/format overhead to pay "
        "off, use scalar";
    return c;
  }

  // (1) Very short rows.
  if (s.mean_row_nnz < 4.0) {
    c.kernel = KernelId::CsrScalar;
    c.lanes = 1;
    c.reason = "mean_row_nnz=" + fmt(s.mean_row_nnz) +
               " < 4: short rows, scalar (one thread/row) is cheapest";
    return c;
  }

  // (2) Locally dense 4x4 structure.
  if (s.frac_dense_blocks_4x4 > 0.6) {
    c.kernel = KernelId::Bcsr;
    c.block_r = 4;
    c.block_c = 4;
    c.reason = "frac_dense_blocks_4x4=" + fmt(s.frac_dense_blocks_4x4) +
               " > 0.6: locally dense, use 4x4 block storage";
    return c;
  }

  // (3) Skewed row lengths -> load-balanced merge path.
  if (s.cv_row_nnz > 1.5 || s.max_over_mean > 50.0) {
    c.kernel = KernelId::CsrMerge;
    c.lanes = 1;
    c.reason = "cv=" + fmt(s.cv_row_nnz) +
               " or max/mean=" + fmt(s.max_over_mean) +
               " indicates skew: use nnz-load-balanced csr_merge";
    return c;
  }

  // (4) Medium uniform rows -> sub-warp vector group (checked before ELL:
  //     a vector group covers these rows with no padding at all).
  if (s.mean_row_nnz >= 4.0 && s.mean_row_nnz < 32.0 && s.cv_row_nnz < 0.5) {
    c.kernel = KernelId::CsrVector;
    const int m = static_cast<int>(std::ceil(s.mean_row_nnz));
    c.lanes = next_pow2(m);
    c.reason =
        "4 <= mean_row_nnz=" + fmt(s.mean_row_nnz) +
        " < 32 and cv=" + fmt(s.cv_row_nnz) +
        " < 0.5: sub-warp vector group with lanes=" + std::to_string(c.lanes);
    return c;
  }

  // (5) Low padding overhead -> ELL.
  if (s.ell_padding_overhead < 1.2) {
    c.kernel = KernelId::Ell;
    c.lanes = 1;
    c.reason = "ell_padding_overhead=" + fmt(s.ell_padding_overhead) +
               " < 1.2: long but nearly uniform rows, coalesced ELL wins";
    return c;
  }

  // (6) Default.
  c.kernel = KernelId::CsrVector;
  c.lanes = 4;
  c.reason = "no special signal (mean_row_nnz=" + fmt(s.mean_row_nnz) +
             ", cv=" + fmt(s.cv_row_nnz) +
             "): default sub-warp vector group with lanes=4";
  return c;
}

}  // namespace spmvbench
