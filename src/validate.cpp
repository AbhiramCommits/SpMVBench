// Numerical validation: every CPU SpMV/SpMM kernel vs the SciPy float64
// reference, in three accumulation modes:
//   (a) fp32 values, fp32 accumulate
//   (b) fp32 values, fp64 accumulate
//   (c) fp64 values, fp64 accumulate
// plus a Kahan-compensated CSR variant as an extra accuracy datapoint.

#include "spmvbench/validate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "spmvbench/bcsr.hpp"
#include "spmvbench/csr.hpp"
#include "spmvbench/ell.hpp"
#include "spmvbench/mtx_io.hpp"
#include "spmvbench/spmv_cpu.hpp"
#include "spmvbench/stats.hpp"

namespace fs = std::filesystem;

namespace spmvbench {
namespace {

struct RefData {
  int m = 0, n = 0, k = 0;
  std::int64_t nnz = 0;
  std::vector<double> x, y, B, Y;
};

// --- tiny flat JSON reader (our own files only) ----------------------------
std::map<std::string, std::string> read_flat_json(const std::string& path) {
  std::map<std::string, std::string> kv;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    const auto q1 = line.find('"');
    if (q1 == std::string::npos) continue;
    const auto q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) continue;
    const std::string key = line.substr(q1 + 1, q2 - q1 - 1);
    const auto colon = line.find(':', q2);
    if (colon == std::string::npos) continue;
    std::string val = line.substr(colon + 1);
    // trim whitespace and trailing comma
    while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
    while (!val.empty() && (val.back() == ',' || val.back() == '\r' || val.back() == ' ')) val.pop_back();
    if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
      val = val.substr(1, val.size() - 2);
    kv[key] = val;
  }
  return kv;
}

std::vector<double> read_f64_bin(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  std::vector<double> out(static_cast<std::size_t>(bytes / sizeof(double)));
  in.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

RefData load_ref(const std::string& ref_dir, const std::string& name) {
  const auto kv = read_flat_json(ref_dir + "/" + name + ".json");
  RefData r;
  r.m = std::stoi(kv.at("m"));
  r.n = std::stoi(kv.at("n"));
  r.k = std::stoi(kv.at("k"));
  r.nnz = std::stoll(kv.at("nnz"));
  r.x = read_f64_bin(ref_dir + "/" + kv.at("x"));
  r.y = read_f64_bin(ref_dir + "/" + kv.at("y"));
  r.B = read_f64_bin(ref_dir + "/" + kv.at("bmat"));
  r.Y = read_f64_bin(ref_dir + "/" + kv.at("ymat"));
  return r;
}

struct ErrStats {
  double max_rel = 0.0;
  double l2_rel = 0.0;
  double max_abs = 0.0;
};

ErrStats compare(const std::vector<double>& yhat,
                 const std::vector<double>& ref) {
  ErrStats e;
  if (yhat.size() != ref.size()) throw std::runtime_error("size mismatch");
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(yhat[i] - ref[i]);
    const double r = std::max(std::fabs(ref[i]), 1e-30);
    e.max_rel = std::max(e.max_rel, d / r);
    e.max_abs = std::max(e.max_abs, d);
    num += d * d;
    den += ref[i] * ref[i];
  }
  e.l2_rel = (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
  return e;
}

template <typename T>
CsrMatrix<T> cast_csr(const CsrMatrix<double>& A) {
  CsrMatrix<T> B;
  B.m = A.m;
  B.n = A.n;
  B.nnz = A.nnz;
  B.rowptr = A.rowptr;
  B.colind = A.colind;
  B.values.resize(static_cast<std::size_t>(A.nnz));
  for (std::int64_t i = 0; i < A.nnz; ++i)
    B.values[static_cast<std::size_t>(i)] = static_cast<T>(A.values[static_cast<std::size_t>(i)]);
  return B;
}

void emit(std::ofstream& csv, const std::string& name, const std::string& kernel,
          const std::string& mode, const ErrStats& e) {
  std::ostringstream os;
  os << name << "," << kernel << "," << mode << "," << e.max_rel << ","
     << e.l2_rel << "," << e.max_abs << "\n";
  csv << os.str();
}

template <typename T, typename Acc>
void validate_type(const std::string& name, const CsrMatrix<double>& Ad,
                   const RefData& ref, const char* mode, std::ofstream& csv) {
  const CsrMatrix<T> A = cast_csr<T>(Ad);
  const std::vector<T> x(ref.x.begin(), ref.x.end());
  std::vector<Acc> y(static_cast<std::size_t>(A.m));

  auto to_double = [](const std::vector<Acc>& v) {
    std::vector<double> d(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = static_cast<double>(v[i]);
    return d;
  };

  spmv_csr_scalar_cpu<T, Acc>(A, x.data(), y.data());
  emit(csv, name, "csr_scalar", mode, compare(to_double(y), ref.y));

  spmv_csr_cpu<T, Acc>(A, x.data(), y.data());
  emit(csv, name, "csr_vector", mode, compare(to_double(y), ref.y));

  spmv_csr_merge_cpu<T, Acc>(A, x.data(), y.data());
  emit(csv, name, "csr_merge", mode, compare(to_double(y), ref.y));

  {
    const EllMatrix<T> ell = csr_to_ell(A);
    spmv_ell_cpu<T, Acc>(ell, x.data(), y.data());
    emit(csv, name, "ell", mode, compare(to_double(y), ref.y));
  }
  {
    const BcsrMatrix<T> b = csr_to_bcsr<T, 4, 4>(A);
    spmv_bcsr_cpu<T, Acc>(b, x.data(), y.data());
    emit(csv, name, "bcsr", mode, compare(to_double(y), ref.y));
  }
  if constexpr (std::is_same<T, Acc>::value) {
    spmv_csr_kahan_cpu<T, Acc>(A, x.data(), y.data());
    emit(csv, name, "csr_kahan", mode, compare(to_double(y), ref.y));
  }

  // SpMM (uses the Y reference).
  {
    const std::vector<T> B(ref.B.begin(), ref.B.end());
    std::vector<Acc> Y(static_cast<std::size_t>(A.m) * ref.k);
    spmm_csr_cpu<T, Acc>(A, B.data(), Y.data(), ref.k);
    emit(csv, name, "spmm_csr", mode, compare(to_double(Y), ref.Y));
  }
}

}  // namespace

int run_validation(const ValidationOptions& opts) {
  std::vector<std::string> mtxs;
  if (fs::exists(opts.data_dir)) {
    for (const auto& e : fs::directory_iterator(opts.data_dir)) {
      if (e.is_regular_file() && e.path().extension() == ".mtx")
        mtxs.push_back(e.path().string());
    }
  }
  std::sort(mtxs.begin(), mtxs.end());
  if (mtxs.empty()) {
    std::fprintf(stderr, "no .mtx files in %s\n", opts.data_dir.c_str());
    return 1;
  }

  fs::create_directories(fs::path(opts.out_csv).parent_path());
  std::ofstream csv(opts.out_csv);
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", opts.out_csv.c_str());
    return 1;
  }
  csv << "matrix,kernel,mode,max_rel_err,l2_rel_err,max_abs_err\n";

  int worst_ok = 0, total = 0;
  for (const auto& p : mtxs) {
    const std::string name = fs::path(p).stem().string();
    RefData ref;
    try {
      ref = load_ref(opts.ref_dir, name);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "skip %s: %s\n", name.c_str(), e.what());
      continue;
    }
    CsrMatrix<double> A = read_matrix_market<double>(p);

    // (a) fp32 values, fp32 accumulate
    validate_type<float, float>(name, A, ref, "fp32/fp32", csv);
    // (b) fp32 values, fp64 accumulate
    validate_type<float, double>(name, A, ref, "fp32/fp64", csv);
    // (c) fp64 values, fp64 accumulate
    validate_type<double, double>(name, A, ref, "fp64/fp64", csv);

    // Sanity: fp64/fp64 CSR must match SciPy to < 1e-12 (recompute here).
    {
      std::vector<double> x(ref.x.begin(), ref.x.end());
      std::vector<double> y(static_cast<std::size_t>(A.m));
      spmv_csr_cpu<double, double>(A, x.data(), y.data());
      const ErrStats e = compare(y, ref.y);
      total++;
      if (e.max_rel < 1e-12) worst_ok++;
      std::printf("[validate] %-24s fp64 csr max_rel=%.3e l2_rel=%.3e max_abs=%.3e\n",
                  name.c_str(), e.max_rel, e.l2_rel, e.max_abs);
    }
  }

  std::printf("[validate] fp64 CSR within 1e-12 relative vs SciPy: %d/%d\n",
              worst_ok, total);
  std::printf("[validate] wrote %s\n", opts.out_csv.c_str());
  return 0;
}

}  // namespace spmvbench
