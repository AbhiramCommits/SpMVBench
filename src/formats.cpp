// Format conversions: CSR -> ELL and CSR -> BCSR.

#include "spmvbench/bcsr.hpp"
#include "spmvbench/ell.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <vector>

namespace spmvbench {

template <typename T>
EllMatrix<T> csr_to_ell(const CsrMatrix<T>& csr) {
  EllMatrix<T> ell;
  ell.m = csr.m;
  ell.n = csr.n;
  ell.nnz = csr.nnz;

  int max_row = 0;
  for (int i = 0; i < csr.m; ++i) {
    max_row = std::max(max_row, static_cast<int>(csr.rowptr[i + 1] - csr.rowptr[i]));
  }
  ell.max_row_nnz = max_row;
  ell.padded_nnz = static_cast<std::int64_t>(csr.m) * max_row;
  ell.colind.assign(static_cast<std::size_t>(ell.padded_nnz), 0);
  ell.values.assign(static_cast<std::size_t>(ell.padded_nnz), T(0));

  for (int i = 0; i < csr.m; ++i) {
    int j = 0;
    for (Index p = csr.rowptr[i]; p < csr.rowptr[i + 1]; ++p, ++j) {
      // Column-major layout: entry j of row i lives at j * m + i.
      ell.colind[static_cast<std::size_t>(j) * csr.m + i] = csr.colind[p];
      ell.values[static_cast<std::size_t>(j) * csr.m + i] = csr.values[p];
    }
  }
  return ell;
}

template <typename T, int R, int C>
BcsrMatrix<T> csr_to_bcsr(const CsrMatrix<T>& csr) {
  static_assert(R >= 1 && C >= 1, "block dimensions must be positive");
  BcsrMatrix<T> B;
  B.m = csr.m;
  B.n = csr.n;
  B.R = R;
  B.C = C;
  B.mb = (csr.m + R - 1) / R;
  B.nb = (csr.n + C - 1) / C;
  B.nnz = csr.nnz;
  B.rowptr.assign(B.mb + 1, 0);

  using Block = std::array<T, R * C>;
  std::vector<Index> bcols;
  std::vector<Block> blocks;

  for (int bi = 0; bi < B.mb; ++bi) {
    bcols.clear();
    blocks.clear();
    for (int r = 0; r < R; ++r) {
      const int i = bi * R + r;
      if (i >= csr.m) break;
      for (Index p = csr.rowptr[i]; p < csr.rowptr[i + 1]; ++p) {
        const Index col = csr.colind[p];
        const Index bj = col / C;
        const int cc = col % C;
        int idx = -1;
        for (std::size_t t = 0; t < bcols.size(); ++t) {
          if (bcols[t] == bj) {
            idx = static_cast<int>(t);
            break;
          }
        }
        if (idx < 0) {
          bcols.push_back(bj);
          blocks.emplace_back();
          blocks.back().fill(T(0));
          idx = static_cast<int>(bcols.size()) - 1;
        }
        blocks[static_cast<std::size_t>(idx)][r * C + cc] = csr.values[p];
      }
    }

    // Emit blocks in ascending block-column order (canonical BCSR).
    std::vector<int> order(bcols.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return bcols[a] < bcols[b]; });
    for (int oi : order) {
      B.colind.push_back(bcols[oi]);
      const Block& blk = blocks[oi];
      B.values.insert(B.values.end(), blk.begin(), blk.end());
    }
    B.rowptr[bi + 1] = B.rowptr[bi] + static_cast<Index>(bcols.size());
  }

  B.block_nnz = static_cast<std::int64_t>(B.colind.size());
  return B;
}

// Explicit instantiations.
template EllMatrix<float> csr_to_ell<float>(const CsrMatrix<float>&);
template EllMatrix<double> csr_to_ell<double>(const CsrMatrix<double>&);

template BcsrMatrix<float> csr_to_bcsr<float, 2, 2>(const CsrMatrix<float>&);
template BcsrMatrix<float> csr_to_bcsr<float, 4, 4>(const CsrMatrix<float>&);
template BcsrMatrix<double> csr_to_bcsr<double, 2, 2>(const CsrMatrix<double>&);
template BcsrMatrix<double> csr_to_bcsr<double, 4, 4>(const CsrMatrix<double>&);

}  // namespace spmvbench
