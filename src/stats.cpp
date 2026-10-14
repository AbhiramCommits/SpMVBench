// Nonzero-distribution statistics.

#include "spmvbench/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <vector>

namespace spmvbench {

RowStats compute_row_stats(const Index* rowptr, const Index* colind, int m,
                           int n, std::int64_t nnz) {
  RowStats s;
  s.nnz = nnz;
  s.rows = m;
  s.cols = n;
  s.density = (static_cast<double>(m) * n > 0)
                  ? static_cast<double>(nnz) / (static_cast<double>(m) * n)
                  : 0.0;

  if (m <= 0) {
    s.max_row_nnz = 0;
    s.min_row_nnz = 0;
    s.row_nnz_histogram.assign(1, 0);
    s.ell_padding_overhead = 1.0;
    s.frac_dense_blocks_4x4 = 0.0;
    return s;
  }

  std::vector<int> row_nnz(m);
  long long sum = 0;
  int mn = 0;
  int mx = 0;
  for (int i = 0; i < m; ++i) {
    const int r = static_cast<int>(rowptr[i + 1] - rowptr[i]);
    row_nnz[i] = r;
    sum += r;
    if (i == 0 || r < mn)
      mn = r;
    if (i == 0 || r > mx)
      mx = r;
  }
  s.min_row_nnz = mn;
  s.max_row_nnz = mx;
  s.mean_row_nnz = static_cast<double>(sum) / m;

  // Median.
  std::vector<int> sorted = row_nnz;
  std::sort(sorted.begin(), sorted.end());
  if (m % 2 == 1) {
    s.median_row_nnz = sorted[m / 2];
  } else {
    s.median_row_nnz = 0.5 * (sorted[m / 2 - 1] + sorted[m / 2]);
  }

  // Population standard deviation and coefficient of variation.
  double var = 0.0;
  for (int i = 0; i < m; ++i) {
    const double d = row_nnz[i] - s.mean_row_nnz;
    var += d * d;
  }
  var /= m;
  s.stddev_row_nnz = std::sqrt(var);
  s.cv_row_nnz = (s.mean_row_nnz > 0) ? s.stddev_row_nnz / s.mean_row_nnz : 0.0;
  s.max_over_mean = (s.mean_row_nnz > 0) ? s.max_row_nnz / s.mean_row_nnz : 0.0;

  // Histogram.
  s.row_nnz_histogram.assign(static_cast<std::size_t>(mx) + 1, 0);
  for (int i = 0; i < m; ++i)
    s.row_nnz_histogram[row_nnz[i]]++;

  // ELL padding overhead with natural max-length padding.
  s.ell_padding_overhead =
      (nnz > 0) ? static_cast<double>(m) * mx / static_cast<double>(nnz) : 1.0;

  // Fraction of *all possible* 4x4 blocks that are > 50% populated.
  {
    const int R = 4, C = 4;
    const int mb = (m + R - 1) / R;
    const int nb = (n + C - 1) / C;
    std::int64_t dense_blocks = 0;
    if (colind != nullptr && nnz > 0) {
      std::vector<Index> bcols;
      std::vector<int> counts;
      for (int bi = 0; bi < mb; ++bi) {
        bcols.clear();
        counts.clear();
        for (int r = 0; r < R; ++r) {
          const int i = bi * R + r;
          if (i >= m)
            break;
          for (Index p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            const Index bj = colind[p] / C;
            int idx = -1;
            for (std::size_t t = 0; t < bcols.size(); ++t) {
              if (bcols[t] == bj) {
                idx = static_cast<int>(t);
                break;
              }
            }
            if (idx < 0) {
              bcols.push_back(bj);
              counts.push_back(1);
            } else {
              counts[static_cast<std::size_t>(idx)]++;
            }
          }
        }
        for (int c : counts) {
          if (c > (R * C) / 2)
            dense_blocks++;
        }
      }
    }
    const std::int64_t possible = static_cast<std::int64_t>(mb) * nb;
    s.frac_dense_blocks_4x4 =
        (possible > 0) ? static_cast<double>(dense_blocks) / possible : 0.0;
  }

  return s;
}

std::string RowStats::to_string() const {
  std::ostringstream os;
  os << "rows=" << rows << " cols=" << cols << " nnz=" << nnz
     << " density=" << density << "\n";
  os << "row_nnz: mean=" << mean_row_nnz << " median=" << median_row_nnz
     << " min=" << min_row_nnz << " max=" << max_row_nnz
     << " stddev=" << stddev_row_nnz << " cv=" << cv_row_nnz
     << " max/mean=" << max_over_mean << "\n";
  os << "ell_padding_overhead=" << ell_padding_overhead
     << " frac_dense_blocks_4x4=" << frac_dense_blocks_4x4;
  return os.str();
}

}  // namespace spmvbench
