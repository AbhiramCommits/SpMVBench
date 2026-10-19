// CUDA tiled SpMM: Y (m x k) = A (m x n, CSR) * B (n x k, row-major dense).
//
// Parallel decomposition: one warp per sparse row. The k dimension is tiled
// into TILE_K-wide tiles held in shared memory accumulators; lanes stride the
// tile so dense-B reads are coalesced across the warp. A load-balanced variant
// bins rows by nnz into short/medium/long buckets and launches one grid per
// bucket so heavy rows do not serialize a single grid.

#include "spmvbench/spmm_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace spmvbench {
namespace {

constexpr int kWarpsPerBlock = 8;

template <typename Ptr>
void cuda_malloc(Ptr** ptr, std::size_t bytes) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr),
                        std::max<std::size_t>(bytes, 1)));
}

template <typename T, typename Acc, int TileK>
__global__ void spmm_tiled_kernel(DeviceCsr<T> A, const T* __restrict__ B,
                                  Acc* __restrict__ Y, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= A.m)
    return;
  __shared__ Acc s_acc[kWarpsPerBlock][TileK];

  for (int t0 = 0; t0 < k; t0 += TileK) {
    const int rem = min(TileK, k - t0);
    for (int j = lane; j < rem; j += 32)
      s_acc[warp][j] = Acc(0);
    __syncwarp();
    for (Index p = A.rowptr[row]; p < A.rowptr[row + 1]; ++p) {
      const Index col = A.colind[p];
      const Acc a = static_cast<Acc>(A.values[p]);
      const T* brow = B + static_cast<std::size_t>(col) * k + t0;
      for (int j = lane; j < rem; j += 32) {
        s_acc[warp][j] += a * static_cast<Acc>(brow[j]);
      }
    }
    __syncwarp();
    for (int j = lane; j < rem; j += 32) {
      Y[static_cast<std::size_t>(row) * k + t0 + j] = s_acc[warp][j];
    }
    __syncwarp();
  }
}

// Bucketed variant: `row_ids` lists the rows assigned to this grid.
template <typename T, typename Acc, int TileK>
__global__ void spmm_tiled_bucketed_kernel(DeviceCsr<T> A,
                                           const Index* __restrict__ row_ids,
                                           int n_bucket_rows,
                                           const T* __restrict__ B,
                                           Acc* __restrict__ Y, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = blockIdx.x * kWarpsPerBlock + warp;
  if (slot >= n_bucket_rows)
    return;
  const int row = row_ids[slot];
  __shared__ Acc s_acc[kWarpsPerBlock][TileK];

  for (int t0 = 0; t0 < k; t0 += TileK) {
    const int rem = min(TileK, k - t0);
    for (int j = lane; j < rem; j += 32)
      s_acc[warp][j] = Acc(0);
    __syncwarp();
    for (Index p = A.rowptr[row]; p < A.rowptr[row + 1]; ++p) {
      const Index col = A.colind[p];
      const Acc a = static_cast<Acc>(A.values[p]);
      const T* brow = B + static_cast<std::size_t>(col) * k + t0;
      for (int j = lane; j < rem; j += 32) {
        s_acc[warp][j] += a * static_cast<Acc>(brow[j]);
      }
    }
    __syncwarp();
    for (int j = lane; j < rem; j += 32) {
      Y[static_cast<std::size_t>(row) * k + t0 + j] = s_acc[warp][j];
    }
    __syncwarp();
  }
}

template <typename T, typename Acc>
void launch_tiled(DeviceCsr<T> A, const T* B, Acc* Y, int k, int tile,
                  cudaStream_t stream) {
  const int grid = std::max(1, (A.m + kWarpsPerBlock - 1) / kWarpsPerBlock);
  switch (tile) {
    case 16:
      spmm_tiled_kernel<T, Acc, 16>
          <<<grid, kWarpsPerBlock * 32, 0, stream>>>(A, B, Y, k);
      break;
    case 64:
      spmm_tiled_kernel<T, Acc, 64>
          <<<grid, kWarpsPerBlock * 32, 0, stream>>>(A, B, Y, k);
      break;
    default:
      spmm_tiled_kernel<T, Acc, 32>
          <<<grid, kWarpsPerBlock * 32, 0, stream>>>(A, B, Y, k);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename Acc>
void launch_bucketed(DeviceCsr<T> A, const std::vector<Index>& short_ids,
                     const std::vector<Index>& med_ids,
                     const std::vector<Index>& long_ids, const T* B, Acc* Y,
                     int k, int tile, cudaStream_t stream) {
  auto launch_one = [&](const std::vector<Index>& ids) {
    if (ids.empty())
      return;
    Index* d_ids = nullptr;
    cuda_malloc(&d_ids, ids.size() * sizeof(Index));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(Index),
                          cudaMemcpyHostToDevice));
    const int grid =
        std::max(1, (static_cast<int>(ids.size()) + kWarpsPerBlock - 1) /
                        kWarpsPerBlock);
    switch (tile) {
      case 16:
        spmm_tiled_bucketed_kernel<T, Acc, 16>
            <<<grid, kWarpsPerBlock * 32, 0, stream>>>(
                A, d_ids, (int)ids.size(), B, Y, k);
        break;
      case 64:
        spmm_tiled_bucketed_kernel<T, Acc, 64>
            <<<grid, kWarpsPerBlock * 32, 0, stream>>>(
                A, d_ids, (int)ids.size(), B, Y, k);
        break;
      default:
        spmm_tiled_bucketed_kernel<T, Acc, 32>
            <<<grid, kWarpsPerBlock * 32, 0, stream>>>(
                A, d_ids, (int)ids.size(), B, Y, k);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    // Keep the allocation alive until the kernel completes.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_ids);
  };
  launch_one(short_ids);
  launch_one(med_ids);
  launch_one(long_ids);
}

template <typename T, typename Acc>
SpmmResult run_tiled(const CsrMatrix<T>& A, const std::vector<T>& B, int k,
                     int tile, bool balanced, int iters, int warmup,
                     std::vector<T>& Y) {
  const int m = A.m;
  const int n = A.n;

  Index* d_rowptr = nullptr;
  Index* d_colind = nullptr;
  T* d_values = nullptr;
  T* d_B = nullptr;
  Acc* d_Y = nullptr;

  cuda_malloc(&d_rowptr, static_cast<std::size_t>(m + 1) * sizeof(Index));
  cuda_malloc(&d_colind, static_cast<std::size_t>(A.nnz) * sizeof(Index));
  cuda_malloc(&d_values, static_cast<std::size_t>(A.nnz) * sizeof(T));
  cuda_malloc(&d_B, static_cast<std::size_t>(n) * k * sizeof(T));
  cuda_malloc(&d_Y, static_cast<std::size_t>(m) * k * sizeof(Acc));

  CUDA_CHECK(cudaMemcpy(d_rowptr, A.rowptr.data(), (m + 1) * sizeof(Index),
                        cudaMemcpyHostToDevice));
  if (A.nnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_colind, A.colind.data(), A.nnz * sizeof(Index),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.nnz * sizeof(T),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_B, B.data(), (std::size_t)n * k * sizeof(T),
                        cudaMemcpyHostToDevice));

  DeviceCsr<T> dA{d_rowptr, d_colind, d_values, m, n, A.nnz};

  // Build buckets once (short / medium / long by row nnz).
  std::vector<Index> short_ids, med_ids, long_ids;
  if (balanced) {
    for (int i = 0; i < m; ++i) {
      const int len = A.rowptr[i + 1] - A.rowptr[i];
      if (len <= 8)
        short_ids.push_back(i);
      else if (len <= 64)
        med_ids.push_back(i);
      else
        long_ids.push_back(i);
    }
  }

  auto launch = [&]() {
    if (balanced) {
      launch_bucketed(dA, short_ids, med_ids, long_ids, d_B, d_Y, k, tile, 0);
    } else {
      launch_tiled(dA, d_B, d_Y, k, tile, 0);
    }
  };

  for (int i = 0; i < warmup; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  double sum = 0.0, mn = 1e300, mx = 0.0;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    sum += ms;
    mn = std::min(mn, (double)ms);
    mx = std::max(mx, (double)ms);
  }

  Y.assign(static_cast<std::size_t>(m) * k, T(0));
  std::vector<Acc> yacc(static_cast<std::size_t>(m) * k);
  CUDA_CHECK(cudaMemcpy(yacc.data(), d_Y, (std::size_t)m * k * sizeof(Acc),
                        cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < yacc.size(); ++i)
    Y[i] = static_cast<T>(yacc[i]);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  cudaFree(d_rowptr);
  cudaFree(d_colind);
  cudaFree(d_values);
  cudaFree(d_B);
  cudaFree(d_Y);

  SpmmResult r;
  r.iters = iters;
  r.mean_ms = sum / std::max(1, iters);
  r.min_ms = mn;
  r.max_ms = mx;
  r.kernel = "spmm_csr_tiled";
  return r;
}

}  // namespace

SpmmResult spmm_csr_tiled_f32(const CsrMatrix<float>& A,
                              const std::vector<float>& B, int k, int tile,
                              bool balanced, int iters, int warmup,
                              std::vector<float>& Y) {
  return run_tiled<float, float>(A, B, k, tile, balanced, iters, warmup, Y);
}

SpmmResult spmm_csr_tiled_f64(const CsrMatrix<double>& A,
                              const std::vector<double>& B, int k, int tile,
                              bool balanced, int iters, int warmup,
                              std::vector<double>& Y) {
  return run_tiled<double, double>(A, B, k, tile, balanced, iters, warmup, Y);
}

}  // namespace spmvbench
