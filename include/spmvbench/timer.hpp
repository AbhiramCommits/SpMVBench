#pragma once

// Wall-clock timer used by the CPU benchmark harness. The CUDA path uses
// cudaEvents instead (see spmv_cuda.cuh).

#include <chrono>

namespace spmvbench {

class CpuTimer {
 public:
  void start() {
    t0_ = clock::now();
  }
  void stop() {
    t1_ = clock::now();
  }

  // Elapsed milliseconds between the last start() and stop().
  double ms() const {
    return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
  }

 private:
  using clock = std::chrono::steady_clock;
  clock::time_point t0_{};
  clock::time_point t1_{};
};

}  // namespace spmvbench
