// CUDA SpMV kernels and the host-side SpmvRunner.
//
// Compiled only when SPMVBENCH_ENABLE_CUDA is defined (see CMakeLists.txt).
// Every kernel is templated on the storage type T and the accumulator type
// Acc so fp32/fp64 mixed accumulation is expressible, matching the CPU path.

#include "spmvbench/spmv_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "spmvbench/bcsr.hpp"
#include "spmvbench/ell.hpp"

namespace spmvbench {

namespace {

constexpr int kWarp = 32;

template <typename Acc>
__device__ __forceinline__ Acc warp_reduce_sum(Acc v, unsigned mask) {
#pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v += __shfl_down_sync(mask, v, off);
  }
  return v;
}

// --- one thread per row ----------------------------------------------------
template <typename T, typename Acc>
__global__ void csr_scalar_kernel(DeviceCsr<T> A, const T* __restrict__ x,
                                  Acc* __restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= A.m)
    return;
  Acc sum = Acc(0);
  for (Index p = A.rowptr[row]; p < A.rowptr[row + 1]; ++p) {
    sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
  }
  y[row] = sum;
}

// --- sub-warp (Lanes) per row, shuffle reduction ---------------------------
template <typename T, typename Acc, int Lanes>
__global__ void csr_vector_kernel(DeviceCsr<T> A, const T* __restrict__ x,
                                  Acc* __restrict__ y) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
  constexpr int kRowsPerWarp = kWarp / Lanes;
  const int row = warp * kRowsPerWarp + lane / Lanes;
  const int sublane = lane % Lanes;
  const bool valid = row < A.m;
  const unsigned mask = __ballot_sync(0xffffffffu, valid);

  Acc sum = Acc(0);
  if (valid) {
    for (Index p = A.rowptr[row] + sublane; p < A.rowptr[row + 1]; p += Lanes) {
      sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
    }
  }
#pragma unroll
  for (int off = Lanes / 2; off > 0; off >>= 1) {
    sum += __shfl_down_sync(mask, sum, off, Lanes);
  }
  if (valid && sublane == 0)
    y[row] = sum;
}

// --- one full warp per row, coalesced --------------------------------------
template <typename T, typename Acc>
__global__ void csr_warp_kernel(DeviceCsr<T> A, const T* __restrict__ x,
                                Acc* __restrict__ y) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
  if (warp >= A.m)
    return;  // whole warp exits together, no divergence
  Acc sum = Acc(0);
  for (Index p = A.rowptr[warp] + lane; p < A.rowptr[warp + 1]; p += kWarp) {
    sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
  }
  sum = warp_reduce_sum(sum, 0xffffffffu);
  if (lane == 0)
    y[warp] = sum;
}

// --- load-balanced: dynamic row chunks from a global atomic counter --------
template <typename T, typename Acc, int RowsPerBlock>
__global__ void csr_merge_kernel(DeviceCsr<T> A, const T* __restrict__ x,
                                 Acc* __restrict__ y,
                                 int* __restrict__ counter) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
  __shared__ int s_start;
  for (;;) {
    if (threadIdx.x == 0)
      s_start = atomicAdd(counter, RowsPerBlock);
    __syncthreads();
    const int row = s_start + warp;
    if (s_start >= A.m)
      break;
    if (row < A.m) {
      Acc sum = Acc(0);
      for (Index p = A.rowptr[row] + lane; p < A.rowptr[row + 1]; p += kWarp) {
        sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
      }
      sum = warp_reduce_sum(sum, 0xffffffffu);
      if (lane == 0)
        y[row] = sum;
    }
    __syncthreads();
  }
}

// --- ELL -------------------------------------------------------------------
template <typename T, typename Acc>
__global__ void ell_kernel(DeviceEll<T> A, const T* __restrict__ x,
                           Acc* __restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= A.m)
    return;
  Acc sum = Acc(0);
  for (int j = 0; j < A.max_row_nnz; ++j) {
    const Index p = static_cast<Index>(j) * A.m + row;  // column-major
    sum += static_cast<Acc>(A.values[p]) * static_cast<Acc>(x[A.colind[p]]);
  }
  y[row] = sum;
}

// --- BCSR: one warp per block row, register-blocked R x C inner product -----
template <typename T, typename Acc, int R, int C>
__global__ void bcsr_kernel(DeviceBcsr<T> A, const T* __restrict__ x,
                            Acc* __restrict__ y) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
  if (warp >= A.mb)
    return;
  Acc acc = Acc(0);
  if (lane < R) {
    for (Index bp = A.rowptr[warp]; bp < A.rowptr[warp + 1]; ++bp) {
      const Index bj = A.colind[bp];
      const T* blk = &A.values[static_cast<std::size_t>(bp) * R * C];
#pragma unroll
      for (int c = 0; c < C; ++c) {
        const int gc = bj * C + c;
        if (gc < A.n)
          acc += static_cast<Acc>(blk[lane * C + c]) * static_cast<Acc>(x[gc]);
      }
    }
  }
  if (lane < R) {
    const int gi = warp * R + lane;
    if (gi < A.m)
      y[gi] = acc;
  }
}

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src) {
  if (!src.empty())
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
}

template <typename Ptr>
void cuda_malloc(Ptr** ptr, std::size_t bytes) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr),
                        std::max<std::size_t>(bytes, 1)));
}

}  // namespace

// ---------------------------------------------------------------------------
// SpmvRunner
// ---------------------------------------------------------------------------
template <typename T, typename Acc>
SpmvRunner<T, Acc>::SpmvRunner(const CsrMatrix<T>& A) : A_(A) {
  upload_matrix();
}

template <typename T, typename Acc>
SpmvRunner<T, Acc>::~SpmvRunner() {
  free_buffers();
}

template <typename T, typename Acc>
void SpmvRunner<T, Acc>::upload_matrix() {
  const int m = A_.m;
  const int n = A_.n;

  cuda_malloc(&d_rowptr_, static_cast<std::size_t>(m + 1) * sizeof(Index));
  cuda_malloc(&d_colind_, static_cast<std::size_t>(A_.nnz) * sizeof(Index));
  cuda_malloc(&d_values_, static_cast<std::size_t>(A_.nnz) * sizeof(T));
  copy_to_device(d_rowptr_, A_.rowptr);
  copy_to_device(d_colind_, A_.colind);
  copy_to_device(d_values_, A_.values);

  d_csr_ = DeviceCsr<T>{d_rowptr_, d_colind_, d_values_, m, n, A_.nnz};

  // ELL
  const EllMatrix<T> ell = csr_to_ell(A_);
  cuda_malloc(&d_ell_values_,
              static_cast<std::size_t>(ell.padded_nnz) * sizeof(T));
  cuda_malloc(&d_ell_colind_,
              static_cast<std::size_t>(ell.padded_nnz) * sizeof(Index));
  copy_to_device(d_ell_values_, ell.values);
  copy_to_device(d_ell_colind_, ell.colind);
  d_ell_ = DeviceEll<T>{d_ell_colind_, d_ell_values_, m, n, ell.max_row_nnz};

  // BCSR 2x2 and 4x4
  auto upload_bcsr = [&](auto tag, T** v, Index** c, Index** r,
                         DeviceBcsr<T>* out) {
    constexpr int R = decltype(tag)::R;
    constexpr int C = decltype(tag)::C;
    const BcsrMatrix<T> b = csr_to_bcsr<T, R, C>(A_);
    cuda_malloc(r, static_cast<std::size_t>(b.mb + 1) * sizeof(Index));
    cuda_malloc(c, static_cast<std::size_t>(b.block_nnz) * sizeof(Index));
    cuda_malloc(v, static_cast<std::size_t>(b.block_nnz) * R * C * sizeof(T));
    copy_to_device(*r, b.rowptr);
    copy_to_device(*c, b.colind);
    copy_to_device(*v, b.values);
    *out = DeviceBcsr<T>{*r, *c, *v, m, n, R, C, b.mb, b.block_nnz};
  };
  struct Tag22 {
    static constexpr int R = 2, C = 2;
  };
  struct Tag44 {
    static constexpr int R = 4, C = 4;
  };
  upload_bcsr(Tag22{}, &d_b2_values_, &d_b2_colind_, &d_b2_rowptr_, &d_bcsr2_);
  upload_bcsr(Tag44{}, &d_b4_values_, &d_b4_colind_, &d_b4_rowptr_, &d_bcsr4_);

  cuda_malloc(&d_x_, static_cast<std::size_t>(n) * sizeof(T));
  cuda_malloc(&d_y_, static_cast<std::size_t>(m) * sizeof(Acc));
  cuda_malloc(&d_counter_, sizeof(int));
}

template <typename T, typename Acc>
void SpmvRunner<T, Acc>::free_buffers() {
  cudaFree(d_rowptr_);
  cudaFree(d_colind_);
  cudaFree(d_values_);
  cudaFree(d_ell_values_);
  cudaFree(d_ell_colind_);
  cudaFree(d_b2_values_);
  cudaFree(d_b2_colind_);
  cudaFree(d_b2_rowptr_);
  cudaFree(d_b4_values_);
  cudaFree(d_b4_colind_);
  cudaFree(d_b4_rowptr_);
  cudaFree(d_x_);
  cudaFree(d_y_);
  cudaFree(d_counter_);
}

template <typename T, typename Acc>
void SpmvRunner<T, Acc>::set_x(const std::vector<T>& x) {
  copy_to_device(d_x_, x);
}

template <typename T, typename Acc>
std::vector<Acc> SpmvRunner<T, Acc>::result() const {
  std::vector<Acc> y(A_.m);
  if (A_.m > 0)
    CUDA_CHECK(
        cudaMemcpy(y.data(), d_y_, A_.m * sizeof(Acc), cudaMemcpyDeviceToHost));
  return y;
}

template <typename T, typename Acc>
SpmvResult SpmvRunner<T, Acc>::run(KernelId id, int iters, int lanes, int block,
                                   int warmup) {
  const int m = A_.m;
  const int threads = 256;

  auto grid_for_rows = [&](int rpb) {
    return static_cast<unsigned>(std::max(1, (m + rpb - 1) / rpb));
  };

  auto launch = [&](int lanes_sel) {
    switch (id) {
      case KernelId::CsrScalar: {
        csr_scalar_kernel<T, Acc>
            <<<grid_for_rows(threads), threads>>>(d_csr_, d_x_, d_y_);
        break;
      }
      case KernelId::CsrVector: {
        const int rows_per_block = (threads / kWarp) * (kWarp / lanes_sel);
        switch (lanes_sel) {
          case 2:
            csr_vector_kernel<T, Acc, 2>
                <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_,
                                                             d_y_);
            break;
          case 4:
            csr_vector_kernel<T, Acc, 4>
                <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_,
                                                             d_y_);
            break;
          case 8:
            csr_vector_kernel<T, Acc, 8>
                <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_,
                                                             d_y_);
            break;
          case 16:
            csr_vector_kernel<T, Acc, 16>
                <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_,
                                                             d_y_);
            break;
          default:
            csr_vector_kernel<T, Acc, 32>
                <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_,
                                                             d_y_);
            break;
        }
        break;
      }
      case KernelId::CsrWarp: {
        const int rows_per_block = threads / kWarp;
        csr_warp_kernel<T, Acc>
            <<<grid_for_rows(rows_per_block), threads>>>(d_csr_, d_x_, d_y_);
        break;
      }
      case KernelId::CsrMerge: {
        constexpr int kRowsPerBlock = 8;  // == warps per block
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        const int blocks = std::min((m + kRowsPerBlock - 1) / kRowsPerBlock,
                                    prop.multiProcessorCount * 32);
        CUDA_CHECK(cudaMemset(d_counter_, 0, sizeof(int)));
        csr_merge_kernel<T, Acc, kRowsPerBlock>
            <<<std::max(1, blocks), threads>>>(d_csr_, d_x_, d_y_, d_counter_);
        break;
      }
      case KernelId::Ell: {
        ell_kernel<T, Acc>
            <<<grid_for_rows(threads), threads>>>(d_ell_, d_x_, d_y_);
        break;
      }
      case KernelId::Bcsr: {
        const int rows_per_block = threads / kWarp;
        if (block <= 2) {
          const int grid =
              std::max(1, (d_bcsr2_.mb + rows_per_block - 1) / rows_per_block);
          bcsr_kernel<T, Acc, 2, 2><<<grid, threads>>>(d_bcsr2_, d_x_, d_y_);
        } else {
          const int grid =
              std::max(1, (d_bcsr4_.mb + rows_per_block - 1) / rows_per_block);
          bcsr_kernel<T, Acc, 4, 4><<<grid, threads>>>(d_bcsr4_, d_x_, d_y_);
        }
        break;
      }
      case KernelId::Count:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  };

  for (int i = 0; i < warmup; ++i)
    launch(lanes);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double sum = 0.0, mn = 1e300, mx = 0.0;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(lanes);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    sum += ms;
    mn = std::min(mn, static_cast<double>(ms));
    mx = std::max(mx, static_cast<double>(ms));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  SpmvResult r;
  r.iters = iters;
  r.mean_ms = sum / std::max(1, iters);
  r.min_ms = mn;
  r.max_ms = mx;
  r.kernel = kernel_name(id);
  return r;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class SpmvRunner<float, float>;
template class SpmvRunner<float, double>;
template class SpmvRunner<double, double>;

}  // namespace spmvbench
