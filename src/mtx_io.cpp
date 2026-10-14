// Matrix Market reader implementation.
//
// Handles:  %%MatrixMarket matrix coordinate <real|integer|pattern> <general|symmetric>
// Rejects:  complex values, array format, non-matrix objects, skew-symmetric.

#include "spmvbench/mtx_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spmvbench {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

struct Triple {
  Index i;  // 0-based row
  Index j;  // 0-based column
  double v;
};

std::runtime_error mm_error(const std::string& path, const std::string& msg) {
  return std::runtime_error("Matrix Market: " + path + ": " + msg);
}

}  // namespace

template <typename T>
CsrMatrix<T> read_matrix_market(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw mm_error(path, "cannot open file");

  std::string line;
  if (!std::getline(in, line)) throw mm_error(path, "empty file");

  // --- Banner -------------------------------------------------------------
  bool pattern = false;
  bool symmetric = false;
  {
    std::istringstream hs(line);
    std::string banner, object, format, field, symmetry;
    hs >> banner >> object >> format >> field >> symmetry;
    if (lower(banner) != "%%matrixmarket")
      throw mm_error(path, "missing %%MatrixMarket banner");
    if (lower(object) != "matrix")
      throw mm_error(path,
                     "only 'matrix' objects are supported, got '" + object + "'");
    if (lower(format) != "coordinate")
      throw mm_error(path, "only 'coordinate' format is supported, got '" +
                               format + "'");
    const std::string f = lower(field);
    if (f == "complex") throw mm_error(path, "complex matrices are not supported");
    if (f != "real" && f != "integer" && f != "pattern")
      throw mm_error(path, "unsupported field type '" + field + "'");
    const std::string sym = lower(symmetry);
    if (sym == "skew-symmetric")
      throw mm_error(path, "skew-symmetric matrices are not supported");
    if (sym != "general" && sym != "symmetric")
      throw mm_error(path, "unsupported symmetry '" + symmetry + "'");
    pattern = (f == "pattern");
    symmetric = (sym == "symmetric");
  }

  // --- Size line (skipping comments / blanks) ------------------------------
  long long rows = 0, cols = 0, entries = 0;
  bool got_size = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '%') continue;
    std::istringstream ss(line);
    if (!(ss >> rows >> cols >> entries))
      throw mm_error(path, "malformed size line: '" + line + "'");
    got_size = true;
    break;
  }
  if (!got_size) throw mm_error(path, "missing size line");
  if (rows < 0 || cols < 0 || entries < 0)
    throw mm_error(path, "negative dimensions in size line");
  if (rows > 2147483647LL || cols > 2147483647LL)
    throw mm_error(path, "dimensions exceed int32 index range");

  // --- Entries -------------------------------------------------------------
  std::vector<Triple> tris;
  tris.reserve(static_cast<std::size_t>(entries) * (symmetric ? 2 : 1));

  long long count = 0;
  while (count < entries && std::getline(in, line)) {
    if (line.empty() || line[0] == '%') continue;
    std::istringstream ss(line);
    long long i, j;
    double v = 1.0;
    if (!(ss >> i >> j))
      throw mm_error(path, "malformed entry line: '" + line + "'");
    if (!pattern) {
      if (!(ss >> v))
        throw mm_error(path, "expected a value on entry line: '" + line + "'");
    }
    if (i < 1 || i > rows || j < 1 || j > cols)
      throw mm_error(path, "index out of range on entry line: '" + line + "'");
    tris.push_back({static_cast<Index>(i - 1), static_cast<Index>(j - 1), v});
    if (symmetric && i != j)
      tris.push_back({static_cast<Index>(j - 1), static_cast<Index>(i - 1), v});
    ++count;
  }
  if (count != entries)
    throw mm_error(path, "expected " + std::to_string(entries) +
                             " entries but found " + std::to_string(count));

  // --- Assemble CSR --------------------------------------------------------
  CsrMatrix<T> csr;
  csr.m = static_cast<int>(rows);
  csr.n = static_cast<int>(cols);
  csr.nnz = static_cast<std::int64_t>(tris.size());
  csr.rowptr.assign(csr.m + 1, 0);
  csr.colind.resize(tris.size());
  csr.values.resize(tris.size());
  if (csr.m == 0 && csr.nnz == 0) {
    csr.rowptr.assign(1, 0);
    return csr;
  }

  for (const auto& t : tris) csr.rowptr[t.i + 1]++;

  std::vector<Index> cursor(csr.m);
  for (int i = 0; i < csr.m; ++i) {
    csr.rowptr[i + 1] += csr.rowptr[i];
    cursor[i] = csr.rowptr[i];
  }
  for (const auto& t : tris) {
    const Index pos = cursor[t.i]++;
    csr.colind[pos] = t.j;
    csr.values[pos] = static_cast<T>(t.v);
  }

  // Canonicalize: sort each row by column index (stable, keeps duplicates).
  std::vector<std::pair<Index, T>> row;
  for (int i = 0; i < csr.m; ++i) {
    const Index s = csr.rowptr[i];
    const Index e = csr.rowptr[i + 1];
    if (e - s <= 1) continue;
    row.clear();
    row.reserve(e - s);
    for (Index p = s; p < e; ++p) row.emplace_back(csr.colind[p], csr.values[p]);
    std::sort(row.begin(), row.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (Index p = s; p < e; ++p) {
      csr.colind[p] = row[p - s].first;
      csr.values[p] = row[p - s].second;
    }
  }

  return csr;
}

// Explicit instantiations.
template CsrMatrix<float> read_matrix_market<float>(const std::string&);
template CsrMatrix<double> read_matrix_market<double>(const std::string&);

}  // namespace spmvbench
