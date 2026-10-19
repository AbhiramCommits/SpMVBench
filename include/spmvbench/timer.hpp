#pragma once

// Wall-clock timer and timing summary used by the benchmark harness. The CUDA
// path uses CUDA events but stores its samples in the same summary struct.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

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

struct TimingSummary {
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double median_ms = 0.0;
  double max_ms = 0.0;
  int iters = 0;
};

// Summarize per-iteration millisecond samples. Median is the primary statistic
// for the benchmark tables (min is reported alongside it); mean/max are kept
// for CSV context.
inline TimingSummary summarize(std::vector<double> samples) {
  TimingSummary s;
  if (samples.empty())
    return s;
  std::sort(samples.begin(), samples.end());
  s.iters = static_cast<int>(samples.size());
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  double sum = 0.0;
  for (double v : samples)
    sum += v;
  s.mean_ms = sum / static_cast<double>(samples.size());
  const std::size_t n = samples.size();
  s.median_ms = (n % 2 == 1) ? samples[n / 2]
                             : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
  return s;
}

}  // namespace spmvbench
