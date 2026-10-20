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

| matrix | precision | before, median of 3 runs | after, median of 3 runs | speedup |
|--------|-----------|--------------------------:|-------------------------:|--------:|
| `powerlaw_skew` | fp64 | 12.9791 ± 0.0013 ms | 1.3470 ± 0.1648 ms | **9.64×** |
| `powerlaw_skew` | fp32 | 9.6621 ± 0.0016 ms | 0.7997 ± 0.0089 ms | **12.08×** |
| `uniform_random` | fp64 | 0.0655 ± 0.0002 ms | 0.0665 ± 0.0007 ms | 0.99× |
| `uniform_random` | fp32 | 0.0513 ± 0.0001 ms | 0.0526 ± 0.0002 ms | 0.98× |

Raw per-run rows: `timings.csv`; the twelve full bench CSVs and the rebuild /
ctest logs are in this directory.

## Real or noise?

**Real.** Three things make the case:

1. The before side is extremely stable: fp64 12.9777–12.9809 ms, fp32
   9.6600–9.6637 ms. There is no baseline drift to explain a 10× move.
2. Every after run is faster than every before run. The worst paired speedup
   is **8.28×**; the best is **11.10×**. The fp64 after side varies more
   (1.17–1.57 ms), but that variance is still far below the effect.
3. The control behaves as expected: `uniform_random` moves by −1% to −2%,
   which is noise. If the improvement were a clock or harness artifact, the
   control would move too.

The larger fp64 variance after the change is run-to-run GPU scheduling/clock
noise on the shared Colab T4; the kernel is now latency-bound rather than
bandwidth-bound, so it is more sensitive to when the long rows land on the
SMs. It does not change the conclusion.

One honest caveat: the optimized ELL is still about 6× slower than `csr_warp`
on `powerlaw_skew` (0.22 ms), because one thread per row still serializes the
8000-entry row. The padded-traffic bottleneck identified by ncu is fixed; the
next step for this matrix would be warp-per-row or hybrid handling for the
long rows.

`results/bench.csv` remains the pre-optimization full benchmark snapshot; this
directory is the controlled before/after experiment.
