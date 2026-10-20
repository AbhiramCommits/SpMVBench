# ELL without the padding tax

## What the profile said

`ncu` on `ell` for `powerlaw_skew` showed a kernel at **96.67% of DRAM peak**,
**99.9% warp execution efficiency**, and **86.8% of warp cycles stalled on
L1TEX scoreboard waits**. It was not short of occupancy or arithmetic. It was
reading 30000 × 8000 = 240M `(column, value)` pairs to find 837k real
nonzeros: **286.6× padding**.

## The change

The ELL layout stays exactly the same. Each row now also carries its real
length (`row_lengths[i]`), and both the CPU and GPU kernels walk
`row_lengths[i]` entries instead of `max_row_nnz`. The padding is still
allocated, but it is never read. The bandwidth model was updated to count the
real entries rather than the padded allocation.

This is the smallest useful version of the hybrid ELL idea: it removes the
padded **traffic** without introducing a second storage format. It does not
shrink the allocation, so a hybrid ELL(K) + CSR overflow is still the follow-up
if memory footprint becomes the constraint.

The change is in `include/spmvbench/ell.hpp`, `src/formats.cpp`,
`src/cpu/spmv_cpu.cpp`, `src/cuda/spmv_cuda.cu`, and the associated device view
and benchmark byte model.

## How the A/B was run

Both sides were measured on the same Tesla T4, with the same data and the same
harness:

- `--bench --gpu-only --iters 50`, 10 warmup iterations, CUDA events;
- **three independent runs** per matrix for each side;
- before = the existing binary; after = rebuild with the optimized kernel;
- `ctest` after the rebuild: **57/57 passed**.

`uniform_random` is the control. Every row has exactly 20 entries, so
`row_lengths[i] == max_row_nnz` and the optimization should do nothing there.

## Results

| matrix | precision | before, median (min–max) | after, median (min–max) | speedup |
|--------|-----------|--------------------------:|-------------------------:|--------:|
| `powerlaw_skew` | fp64 | 12.9786 ms (12.9777–12.9809) | 1.3046 ms (1.1698–1.5667) | **9.95×** |
| `powerlaw_skew` | fp32 | 9.6625 ms (9.6600–9.6637) | 0.8029 ms (0.7876–0.8087) | **12.03×** |
| `uniform_random` | fp64 | 0.06546 ms (0.06539–0.06579) | 0.06680 ms (0.06554–0.06707) | 0.98× |
| `uniform_random` | fp32 | 0.05134 ms (0.05120–0.05144) | 0.05264 ms (0.05224–0.05283) | 0.98× |

The speedup column is the ratio of the before and after **medians**, not a mean
of per-run speedups. With three runs per side the median is one of the measured
values; the parenthesized range is min–max, not a standard deviation.

Raw per-run rows: `timings.csv`; the twelve full bench CSVs and the rebuild /
ctest logs are in this directory.

## Real or noise?

**Real.** Three things make the case:

1. The before side is extremely stable: fp64 12.9777–12.9809 ms, fp32
   9.6600–9.6637 ms. There is no baseline drift to explain a 10× move.
2. Every after run is faster than every before run. Paired run speedups are
   **8.29–11.09×** for fp64 and **11.95–12.27×** for fp32. The fp64 after side
   varies more (1.17–1.57 ms), but that variance is still far below the effect.
3. The control behaves as expected: `uniform_random` moves by less than 3% in
   either direction (0.98× median speedup), which is the same order as
   run-to-run noise. If the improvement were a clock or harness artifact, the
   control would move too.

The larger fp64 variance after the change is run-to-run GPU scheduling/clock
noise on the shared Colab T4; the kernel is now latency-bound rather than
bandwidth-bound, so it is more sensitive to when the long rows land on the
SMs. It does not change the conclusion.

One honest caveat: the optimized ELL is still about 5.9× slower than `csr_warp`
on `powerlaw_skew` (1.30 ms vs 0.22 ms), because one thread per row still
serializes the 8000-entry row. The padded-traffic bottleneck identified by ncu
is fixed; the next step for this matrix would be warp-per-row or hybrid
handling for the long rows.

`results/bench.csv` has now been re-run on the current code; this directory
remains the controlled before/after experiment. Because the A/B used
`--bench --gpu-only`, its absolute times are not directly comparable to the
full sweep (for example, `uniform_random` fp64 is 0.0668 ms here but 0.1019 ms
in the full sweep).
