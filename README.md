# SpMVBench

**Hardware.** NVIDIA Tesla T4 (15360 MiB); Intel Xeon @ 2.00 GHz, 2 vCPU.

## What it is

SpMVBench is a hand-written sparse linear algebra library: CSR, ELL and blocked-CSR SpMV plus a tiled sparse×dense SpMM, for CUDA and OpenMP.
It includes a nonzero-distribution heuristic that picks a format/kernel, and benchmarks every kernel against cuSPARSE on GPU and a multithreaded OpenMP CSR baseline on CPU.
A SciPy-referenced validation checks fp32/fp64 accumulation; every number below comes from a file in `results/`.

## Results

Medians, fp64, from `results/bench.csv` (the exact table is `results/readme_table.csv`). "GPU best" excludes the cuSPARSE baseline; "CPU best" excludes the single-thread context row.

The CPU timings come from a **2-core Colab VM** (Intel Xeon @ 2.00 GHz, 2 vCPU), not a scaling study. Read the 0.73–0.95× ratios as a statement about that 2-vCPU allocation, not as a general claim about OpenMP.

| matrix | GPU best | GPU ms | vs cuSPARSE | CPU best (parallel) | CPU ms | vs OpenMP |
|--------|----------|-------:|------------:|---------------------|-------:|----------:|
| amazon0302 | `ell` | 0.101 | 1.70× | `csr_merge_omp` | 2.406 | 0.86× |
| banded_100k | `ell` | 0.03765 | 1.31× | `ell_omp` | 0.497 | 0.80× |
| cage10 | `csr_vector_l4` | 0.009952 | 1.85× | `csr_merge_omp` | 0.1794 | 0.90× |
| cfd1 | `ell` | 0.1088 | 1.92× | `csr_merge_omp` | 1.72 | 0.74× |
| diag_dominant_small | `csr_vector_l4` | 0.006624 | 2.55× | `ell_omp` | 0.0163 | 0.73× |
| powerlaw_skew | `csr_warp` | 0.2205 | 0.55× | `csr_merge_omp` | 0.917 | 0.87× |
| uniform_random | `ell` | 0.1019 | 1.43× | `csr_merge_omp` | 1.153 | 0.84× |
| wiki-Vote | `csr_warp` | 0.02109 | 0.82× | `csr_merge_omp` | 0.1456 | 0.95× |

The OpenMP baseline is strong: the best parallel CPU kernel is slower than it on every matrix (0.73–0.95×). On the GPU, `ell` wins four of eight matrices, `csr_warp` wins the two skewed ones, and the remaining two go to `csr_vector_l4`.

The table above is the post-fix full sweep. The controlled A/B in `results/ell_optimization/` uses `--bench --gpu-only`, so its absolute ELL times are not directly comparable to this full sweep (for example, `uniform_random` fp64 is 0.0668 ms there vs 0.1019 ms here); both are current-code measurements.

## Heuristic accuracy

`choose_spmv` picked the fastest GPU kernel on **0/9** matrices. Worst miss: **1.84×** on `wiki-Vote` (`csr_merge` 0.0264 ms vs `csr_vector` 0.0144 ms). Next: `uniform_random` **1.56×** (`csr_vector` vs `ell`) and `cage10` **1.45×** (`csr_vector` vs `ell`). The rule order is the problem: the vector rule is checked before ELL, and high CV is sent to merge; on the T4, ELL and the warp/vector kernels won those cases. Full table: `results/heuristic.csv`.

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
- **ELL** — column-major padded. It now walks only `row_lengths` and skips the padded tail, which is best on four of eight GPU matrices. On `powerlaw_skew` the controlled A/B took it from 12.98 ms to 1.30 ms (9.95×); the current full sweep measures 1.10 ms. Storage is still padded; hybrid ELL+CSR is the next step.
- **BCSR** — 4×4 blocks. Only pays off with dense blocks, which this matrix set does not have.
- **SpMM** — one warp per sparse row, k-tiled shared-memory accumulators, plus a bucketed load-balanced variant. cuSPARSE is the baseline; details in `results/bench.md`.

## What I learned

- I built the heuristic on CPU intuition, and the GPU did not agree with it. `choose_spmv` went 0/9 on the T4: it kept picking vector where ELL was faster, and merge for the skewed matrices where warp/vector won. The 1.84× miss on `wiki-Vote` (and 1.56× on `uniform_random`) was the tell.
- The ncu run was more useful than adding another kernel variant. ELL on `powerlaw_skew` was at 96.7% of DRAM and 99.9% warp efficiency; the problem was the 286× padding, not occupancy. Reading only real row entries took it from 12.98 ms to 1.30 ms in the controlled A/B (9.95×), while the uniform control did not move.
- The baseline changes the story. Against a single thread everything looked like a win; against OpenMP CSR on the 2-vCPU Colab VM, the best parallel CPU kernel is 0.73–0.95× (slower on every matrix), and `powerlaw_skew`/`wiki-Vote` are 0.55×/0.82× vs cuSPARSE. The CPU numbers are specific to that 2-core allocation; they are not a general OpenMP scaling result.

## License

MIT.
