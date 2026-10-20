# SpMVBench

**Hardware.** NVIDIA Tesla T4 (15360 MiB); Intel Xeon @ 2.00 GHz, 2 vCPU.

## What it is

SpMVBench is a hand-written sparse linear algebra library: CSR, ELL and blocked-CSR SpMV plus a tiled sparse×dense SpMM, for CUDA and OpenMP.
It includes a nonzero-distribution heuristic that picks a format/kernel, and benchmarks every kernel against cuSPARSE on GPU and a multithreaded OpenMP CSR baseline on CPU.
A SciPy-referenced validation checks fp32/fp64 accumulation; every number below comes from a file in `results/`.

## Results

Medians, fp64, from `results/bench.csv` (the exact table is `results/readme_table.csv`). "GPU best" excludes the cuSPARSE baseline; "CPU best" excludes the single-thread context row.

| matrix | GPU best | GPU ms | vs cuSPARSE | CPU best (parallel) | CPU ms | vs OpenMP |
|--------|----------|-------:|------------:|---------------------|-------:|----------:|
| amazon0302 | `ell` | 0.0951 | 1.91× | `csr_merge_omp` | 2.615 | 0.79× |
| banded_100k | `ell` | 0.0357 | 1.28× | `ell_omp` | 0.430 | 0.96× |
| cage10 | `csr_vector_l4` | 0.00990 | 1.80× | `csr_merge_omp` | 0.133 | 0.93× |
| cfd1 | `ell` | 0.1255 | 1.66× | `csr_merge_omp` | 1.444 | 0.85× |
| diag_dominant_small | `csr_vector_l4` | 0.00618 | 2.56× | `ell_omp` | 0.0121 | 0.93× |
| powerlaw_skew | `csr_warp` | 0.2203 | 0.55× | `csr_merge_omp` | 0.898 | 0.91× |
| uniform_random | `ell` | 0.1063 | 1.37× | `csr_merge_omp` | 1.131 | 0.89× |
| wiki-Vote | `csr_warp` | 0.0505 | 0.70× | `csr_merge_omp` | 0.131 | 1.01× |

The OpenMP baseline is strong: the best parallel CPU kernel is often slower than it. On the GPU, `ell` wins four of eight matrices, `csr_warp` wins the two skewed ones, and the remaining two go to `csr_vector_l4`.

The full benchmark predates the ELL traversal fix described below; the ELL note has the current measured numbers.

## Heuristic accuracy

`choose_spmv` picked the fastest GPU kernel on **0/9** matrices. Worst miss: **2.31×** on `amazon0302` (`csr_vector` 0.221 ms vs `ell` 0.096 ms). Next: `wiki-Vote` **1.84×** (`csr_merge` vs `csr_vector`) and `uniform_random` **1.56×** (`csr_vector` vs `ell`). The rule order is the problem: the vector rule is checked before ELL, and high CV is sent to merge; on the T4, ELL and the warp/vector kernels won those cases. Full table: `results/heuristic.csv`.

## How to run

```bash
cmake -S . -B build -DSPMVBENCH_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build
ctest --test-dir build

python3 tools/gen_matrices.py --suitesparse-set
python3 tools/reference.py
./build/bench --validate
./build/bench --bench --iters 50
./build/bench --auto --all
python3 tools/report.py
bash scripts/profile.sh
```

For a CPU-only build, omit the CUDA flags. `--suitesparse-set` needs network; the synthetic matrices still work without it.

## Design notes

- **CSR scalar** — one thread per row, no cross-lane work. The OpenMP version is the CPU baseline; the serial version is context only.
- **CSR vector** — 2/4/8/16/32 lanes per row with a shuffle reduction; good for medium uniform rows. The benchmark uses lanes=4.
- **CSR warp** — one full warp per row, coalesced loads. It won `powerlaw_skew` and `wiki-Vote` on the T4.
- **CSR merge** — dynamic row chunks. It fills the SMs (98.5–98.9% occupancy) but reaches only 23–27.5% memory throughput and did not win.
- **ELL** — column-major padded. It now walks only `row_lengths` and skips the padded tail, which is best on four of eight GPU matrices. On `powerlaw_skew` the fix took it from 12.98 ms to 1.35 ms (9.6×). Storage is still padded; hybrid ELL+CSR is the next step.
- **BCSR** — 4×4 blocks. Only pays off with dense blocks, which this matrix set does not have.
- **SpMM** — one warp per sparse row, k-tiled shared-memory accumulators, plus a bucketed load-balanced variant. cuSPARSE is the baseline; details in `results/bench.md`.

## What I learned

- I built the heuristic on CPU intuition, and the GPU did not agree with it. `choose_spmv` went 0/9 on the T4: it kept picking vector where ELL was faster, and merge for the skewed matrices where warp/vector won. The 2.31× miss on `amazon0302` was the tell.
- The ncu run was more useful than adding another kernel variant. ELL on `powerlaw_skew` was at 96.7% of DRAM and 99.9% warp efficiency; the problem was the 286× padding, not occupancy. Reading only real row entries took it from 12.98 ms to 1.35 ms (9.6×), while the uniform control did not move.
- The baseline changes the story. Against a single thread everything looked like a win; against OpenMP CSR, the best parallel CPU kernel is 0.79–1.01× (often slower), and `powerlaw_skew`/`wiki-Vote` are 0.55×/0.70× vs cuSPARSE. I trust those numbers more.

## License

MIT.
