# Nsight Compute profile: powerlaw_skew and uniform_random

Device: **Tesla T4** (driver 580.82.07, CUDA 13.0.88, compute capability 7.5).
Each kernel was profiled with one matching launch:

```bash
ncu --set full --kernel-name-base function \
    --kernel-name "regex:<kernel>_kernel" --launch-count 1 \
    -o results/ncu/<matrix>__<kernel> \
    build/bench --bench --gpu-only --iters 1 --warmup 0 data/<matrix>.mtx
```

Raw reports: `results/ncu/<matrix>__<kernel>.ncu-rep`; imported details:
`results/ncu/<matrix>__<kernel>.details.txt`; ncu console logs:
`results/ncu/<matrix>__<kernel>.log`.

## Metrics

| matrix | kernel | duration (ms) | achieved occupancy (%) | memory throughput (% peak) | warp exec. efficiency (%) | main stall |
|--------|--------|--------------:|-----------------------:|---------------------------:|--------------------------:|------------|
| powerlaw_skew | `csr_vector` | 0.393 | 39.45 | 15.10 | 32.2 | Long Scoreboard (L1TEX), 73.7% |
| powerlaw_skew | `csr_merge` | 0.242 | 98.52 | 23.03 | 78.8 | Long Scoreboard (L1TEX), 48.7% |
| powerlaw_skew | `ell` | **13.140** | 72.64 | **96.67** | 99.9 | Long Scoreboard (L1TEX), 86.8% |
| uniform_random | `csr_vector` | 0.115 | 85.64 | 54.93 | 96.2 | LG throttle (L1TEX instruction queue), 53.7% |
| uniform_random | `csr_merge` | 0.295 | 98.90 | 27.50 | 86.4 | Long Scoreboard (L1TEX), 55.4% |
| uniform_random | `ell` | 0.105 | 73.63 | 53.83 | 100.0 | LG throttle (L1TEX instruction queue), 81.4% |

Definitions:

* **Achieved occupancy** is ncu's `Achieved Occupancy` (active warps as a
  percentage of the maximum resident warps).
* **Memory throughput** is the ncu Speed-of-Light `Memory Throughput` /
  `Max Bandwidth` percentage (DRAM). For `powerlaw_skew/ell` the absolute value
  is 307.38 GB/s (96.67% of the clock-scaled DRAM peak).
* **Warp execution efficiency** is derived from ncu's
  `Avg. Active Threads Per Warp / 32`; the CUDA 13 details page reports the
  former rather than the older named metric.
* **Main stall** is the dominant reason in the Warp State Statistics OPT text,
  with its share of the average warp cycles between two issued instructions.

## Slowest kernel and highest-impact optimization

**`ell` on `powerlaw_skew` is the slowest by a wide margin: 13.14 ms**, versus
0.24–0.39 ms for the CSR variants on the same matrix. It is already at
**96.67% of DRAM peak** with **99.9% warp execution efficiency** and 72.6%
occupancy: the kernel is not compute-, occupancy-, or divergence-limited. It is
bandwidth-bound on **padded zeros**. `powerlaw_skew` has max row length 8000 and
mean 27.9, so ELL stores 30000 × 8000 = 240M entries for 837k real nonzeros
(**286.6× padding**). The 86.8% Long Scoreboard share is the symptom of waiting
on those padded value/column loads.

**Highest-impact optimization: replace full-width ELL padding with a hybrid
ELL(K) + CSR-overflow layout** (or, as an immediate dispatch-only mitigation,
route high-CV matrices to `csr_warp`, which is 0.22 ms here — about 60× faster).

The hybrid keeps the first K entries of each row (e.g. K = 8/16/32) in the
coalesced column-major ELL arrays and stores the remaining entries in a CSR
overflow structure. For `powerlaw_skew`, that removes essentially all of the
286.6× padded traffic: the ELL part covers the common short rows, and only the
genuinely long rows pay the CSR index cost. The trade-off is a second code path
and slightly less regular control flow for the overflow rows, but the measured
96.67% DRAM utilization shows there is no headroom left in the current layout —
reducing bytes moved is the only high-impact lever.

Not implemented, per the request.
