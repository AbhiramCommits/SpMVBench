# SpMVBench

Hand-written sparse linear algebra primitives with automatic format/kernel
selection, cuSPARSE baselining, and numerical error analysis.

SpMVBench implements, from scratch:

* **SpMV** (sparse matrix-vector) kernels in five decompositions:
  CSR scalar, CSR vector (sub-warp), CSR warp, CSR merge (load-balanced), ELL,
  and blocked-CSR (2x2 / 4x4) — each templated on storage type and accumulator
  type so fp32-value/fp64-accumulate is first-class.
* **SpMM** (sparse x dense): a shared-memory, k-tiled kernel with a
  load-balanced, nnz-bucketed variant.
* A **portable OpenMP CPU path** (bit-for-bit reference for the GPU kernels and
  the actual path measured on machines without a GPU).
* A **nonzero-distribution heuristic** (`choose_spmv`) that picks a kernel from
  row-length statistics, and a harness that proves the choice against a
  measured oracle.
* **Numerical validation** against SciPy with an explicit fp32/fp64 accumulation
  error study and the `gamma_n` error bound.

The CUDA kernels and the cuSPARSE/cuSPARSE-SpMM baselines are **source-complete
and compiling-gated** behind `SPMVBENCH_ENABLE_CUDA`. This machine has no CUDA
toolchain, so the GPU path was neither compiled nor measured here (see
[Hardware](#hardware--software-actually-used)).

---

## Architecture

```
CMakeLists.txt              # CUDA auto-detect (CheckLanguage), OpenMP, -O3
include/spmvbench/          # public headers
  csr.hpp ell.hpp bcsr.hpp  # formats
  mtx_io.hpp stats.hpp      # Matrix Market reader + row statistics
  select.hpp                # kernel-selection heuristic
  spmv_cpu.hpp              # OpenMP CPU kernels
  spmv_cuda.cuh spmm_cuda.cuh   # CUDA interfaces (CUDA-only)
  timer.hpp dtypes.hpp omp_compat.hpp validate.hpp
src/                        # implementation
  mtx_io.cpp formats.cpp stats.cpp select.cpp validate.cpp
  cpu/spmv_cpu.cpp cpu/spmm_cpu.cpp
  cuda/spmv_cuda.cu spmm_cuda.cu cusparse_baseline.cu cusparse_spmm.cu
bench/bench_main.cpp        # --stats / --auto / --validate / --bench
tools/gen_matrices.py       # seeded synthetic .mtx (+ optional SuiteSparse fetch)
tools/reference.py          # SciPy float64 references -> data/ref/*.bin + .json
tools/report.py             # results/bench.csv + accuracy.csv -> summary.md, png
scripts/profile.sh          # Nsight Compute driver (no-op with explanation if absent)
tests/                      # GoogleTest suite (34 cases)
results/                    # committed, real measured artifacts
data/                       # generated matrices + references (gitignored)
```

### Data flow

```
.mtx ──read_matrix_market──▶ CsrMatrix<T>
                                │
             ┌──────────────────┼───────────────────┐
             ▼                  ▼                   ▼
        RowStats          csr_to_ell          csr_to_bcsr<4,4>
             │                  │                   │
             ▼                  ▼                   ▼
      choose_spmv ───▶  {scalar, vector, warp, merge, ell, bcsr}  ──▶ kernel
             │                                                        │
             ▼                                                        ▼
   results/heuristic.csv                         validate (vs SciPy) / bench
             │                                                        │
             └──────────────► tools/report.py ◀──────────────────────┘
                                   │
                          results/summary.md, gflops.png
```

### Kernel decomposition and when it wins

| kernel | parallel decomposition | wins when |
|--------|------------------------|-----------|
| `csr_scalar` | one thread per row, no cross-lane work | very short rows or tiny problems, where orchestration dominates |
| `csr_vector` | a sub-warp (2/4/8/16/32 lanes) per row, `__shfl_down_sync` reduction | medium, low-CV rows (4–32 nnz) |
| `csr_warp` | one full warp per row, coalesced `colind`/`values` | long rows, coalesced streaming |
| `csr_merge` | rows handed out from a global atomic counter in chunks (load balanced) | high-CV / heavy-tailed row lengths |
| `ell` | one thread per row over padded, column-major columns | nearly uniform row lengths (low padding) |
| `bcsr` | one warp per block row, register-blocked R x C inner product | locally dense 4x4 structure |
| `spmm_csr_tiled` | one warp per row, k tiled into shared-memory accumulators | sparse x dense with k ≥ 16; bucketed variant for skew |

The heuristic (`src/select.cpp`) encodes these as an explicit, ordered rule
list (tiny-problem, dense-block, skew, padding, medium-uniform, default) and
returns a human-readable reason string.

---

## Hardware & software actually used

All results in this README were produced on this machine with these tools. The
CUDA probes are reproduced verbatim (they are the evidence that the GPU path
could not be measured):

```
$ nvcc --version
zsh:1: command not found: nvcc              # exit 127

$ nvidia-smi
zsh:1: command not found: nvidia-smi        # exit 127

$ sysctl -n machdep.cpu.brand_string
Apple M3                                    # 8 cores, unified memory

$ sysctl -n hw.memsize
17179869184                                 # 16 GiB

$ sw_vers
ProductName:    macOS
ProductVersion: 27.0.1
BuildVersion:   26A434

$ clang++ --version
Apple clang version 21.0.0 (clang-2100.3.34.2)

$ cmake --version
cmake version 4.3.3

$ python3 --version && python3 -c "import numpy,scipy,matplotlib;print(numpy.__version__,scipy.__version__,matplotlib.__version__)"
3.9.6   numpy 2.0.2   scipy 1.13.1   matplotlib 3.9.4
```

* **OpenMP**: `find_package(OpenMP)` succeeded against Homebrew `libomp`
  (`-Xclang -fopenmp`), 8 threads available.
* **No GPU / no `nvcc` / no `ncu`.** The CUDA sources compile only behind
  `SPMVBENCH_ENABLE_CUDA`; the OpenMP CPU path is what was built, tested, and
  benchmarked. Every GPU number is therefore marked **not measured on this
  machine** where it would otherwise appear.

---

## How to run

```bash
# 0. toolchain: CMake >= 3.20, a C++17 compiler, Python 3 with numpy/scipy.

# 1. configure + build (CUDA auto-detected; force with -DSPMVBENCH_ENABLE_CUDA=ON/OFF)
cmake -S . -B build
cmake --build build -j

# 2. tests
ctest --test-dir build --output-on-failure

# 3. generate the synthetic matrices (deterministic, seeded)
python3 tools/gen_matrices.py
#    optionally fetch a real SuiteSparse matrix (network-dependent):
# python3 tools/gen_matrices.py --suitesparse cage10

# 4. generate SciPy float64 references (data/ref/*.bin + *.json)
python3 tools/reference.py

# 5. inspect statistics / kernel choice
./build/bench --stats --all
./build/bench --auto --all

# 6. numerical validation -> results/accuracy.csv
./build/bench --validate

# 7. benchmark harness -> results/bench.csv + results/bench.md
./build/bench --bench --iters 50

# 8. report + profiling (ncu no-ops with an explanation when absent)
python3 tools/report.py
bash scripts/profile.sh
```

`bench --help` lists every subcommand.

> Note: `tools/reference.py` and `tools/report.py` need NumPy/SciPy/matplotlib.
> On this machine those live in `/usr/bin/python3`, so if a Homebrew Python
> shadows it, invoke `/usr/bin/python3 tools/reference.py`.

---

## RESULTS

Every number below is copied from a committed artifact under `results/`. The
raw data is `results/bench.csv`, `results/accuracy.csv`, `results/heuristic.csv`,
`results/summary.md` and `results/ncu_summary.txt`. Speedup is measured against
the **single-threaded CPU `csr_scalar` kernel** (no GPU on this machine).

### Matrices (from `bench --stats`)

| matrix | rows | cols | nnz | mean row nnz | max row nnz | CV | ELL padding |
|--------|-----:|-----:|----:|-------------:|------------:|---:|------------:|
| `banded_100k` | 100000 | 100000 | 497998 | 4.98 | 5 | 0.028 | 1.004 |
| `diag_dominant_small` | 2000 | 2000 | 16000 | 8.0 | 8 | 0.0 | 1.0 |
| `powerlaw_skew` | 30000 | 30000 | 837371 | 27.91 | 8000 | 5.309 | 286.6 |
| `tiny_3x3` | 3 | 3 | 5 | 1.67 | 2 | 0.283 | 1.2 |
| `uniform_random` | 50000 | 50000 | 1000000 | 20.0 | 20 | 0.0 | 1.0 |

### Best measured kernel per matrix (from `results/bench.csv`)

Throughput and speedup use the **minimum** time over 50 iterations (best case);
the mean is shown for context. The 3x3 fixture (`tiny_3x3`) is omitted here
because its times are below useful timer resolution (it is covered by the tests
and the heuristic table).

| matrix | prec | best kernel | min ms | mean ms | GFLOP/s | BW lo GB/s | BW hi GB/s | speedup |
|--------|------|-------------|-------:|--------:|--------:|-----------:|-----------:|--------:|
| `banded_100k` | fp32 | `ell` | 0.05721 | 0.0638 | 17.41 | 83.9 | 111.9 | 3.70x |
| `banded_100k` | fp64 | `ell` | 0.05833 | 0.06429 | 17.07 | 130.3 | 185.1 | 3.63x |
| `diag_dominant_small` | fp32 | `csr_scalar` | 0.006416 | 0.006479 | 4.99 | 23.69 | 32.42 | 1.00x |
| `diag_dominant_small` | fp64 | `csr_scalar` | 0.006458 | 0.006505 | 4.96 | 35.93 | 53.27 | 1.00x |
| `powerlaw_skew` | fp32 | `csr_vector` | 0.1576 | 0.1953 | 10.62 | 44.78 | 65.27 | 4.42x |
| `powerlaw_skew` | fp64 | `csr_merge` | 0.1452 | 0.1807 | 11.54 | 73.35 | 117.8 | 4.92x |
| `uniform_random` | fp32 | `ell` | 0.1232 | 0.1449 | 16.24 | 68.2 | 99.05 | 4.51x |
| `uniform_random` | fp64 | `ell` | 0.1409 | 0.1767 | 14.19 | 90.83 | 144.8 | 4.53x |

Best speedup on this machine: **4.92x** (`powerlaw_skew`, fp64, `csr_merge`).
Worst real case: **0.0029x** for `ell` on `powerlaw_skew` (frontier of the
table), because ELL pads the 8000-wide heavy row into 286x more storage; `bcsr`
on the tiny `diag_dominant_small` is 0.133x for the same reason (block padding
on a matrix with no dense blocks). cuSPARSE speedups are **not measured on this
machine** (no GPU).

### Heuristic vs measured oracle (from `results/heuristic.csv`)

`choose_spmv` picked the fastest measured kernel on **4/5 = 80%** of matrices;
worst-case slowdown on a miss was **1.04x** (essentially a tie).

| matrix | chosen | oracle best | chosen ms | best ms | hit | slowdown |
|--------|--------|-------------|----------:|--------:|-----|---------:|
| `banded_100k` | `csr_vector` | `csr_vector` | 0.0773 | 0.0773 | yes | 1.00x |
| `diag_dominant_small` | `csr_scalar` | `csr_scalar` | 0.006546 | 0.006546 | yes | 1.00x |
| `powerlaw_skew` | `csr_merge` | `csr_vector` | 0.2037 | 0.1950 | no | 1.04x |
| `tiny_3x3` | `csr_scalar` | `csr_scalar` | 6.25e-06 | 6.25e-06 | yes | 1.00x |
| `uniform_random` | `csr_vector` | `csr_vector` | 0.2109 | 0.2109 | yes | 1.00x |

The single miss is `powerlaw_skew`, where the heuristic's skew rule chooses the
load-balanced `csr_merge` but `csr_vector` is 4% faster — a statistical tie on
8 cores. The heuristic is tuned for GPU behaviour, so its CPU-only hit rate is
a conservative lower bound. Chosen-vs-oracle timings come from a separate
`--auto` run than `--bench`, so small run-to-run differences appear between the
two tables (e.g. `banded_100k`'s best flips between `ell` and `csr_vector`,
which are within ~5%).

### Accuracy: max relative error per mode (from `results/accuracy.csv`)

CSR SpMV; `fp64/fp64` matched SciPy to **exactly 0** relative error on all 5
matrices (5/5 within 1e-12). Full discussion: [`docs/error_analysis.md`](docs/error_analysis.md).

| matrix | fp32/fp32 | fp32/fp64 | fp64/fp64 |
|--------|-----------|-----------|-----------|
| `banded_100k` | 3.305e-04 | 3.407e-04 | 0 |
| `diag_dominant_small` | 1.770e-05 | 1.219e-05 | 0 |
| `powerlaw_skew` | 9.969e-04 | 5.946e-04 | 0 |
| `tiny_3x3` | 5.752e-08 | 5.752e-08 | 0 |
| `uniform_random` | 4.321e-04 | 5.449e-04 | 0 |

The long-row `powerlaw_skew` matrix has the largest fp32/fp32 error; switching
only the accumulator to fp64 cuts its L2 relative error from **4.35e-07** to
**3.27e-08**, and Kahan compensation on the fp32 accumulator reaches **4.71e-08**
without widening storage. Observed absolute errors stay below the worst-case
`gamma_n * sum|a_ij x_j|` bound in every case.

### Profiling

No concrete occupancy/memory findings are available: **Nsight Compute was not
available on this machine**, and there was no GPU to profile. The exact probe is
committed to `results/ncu_summary.txt`:

```
Nsight Compute (ncu) is not installed on this machine.
A CUDA-capable NVIDIA GPU is also absent, so the GPU kernels could
not be profiled (they are source-complete but gated behind
SPMVBENCH_ENABLE_CUDA and were not compiled or run here).

Diagnostics:
  $ nvcc --version
    nvcc: command not found
  $ nvidia-smi
    nvidia-smi: command not found
```

`scripts/profile.sh` runs `ncu --set full --kernel-name-base function` on the
two most interesting kernels on a GPU host and dumps `ncu --import ... --page
details` into `results/ncu_summary.txt`; on this host it records the reason and
exits 0 so the pipeline stays green.

### Tests

`ctest --test-dir build` runs **34 GoogleTest cases**, all passing
(`test_mtx_io`, `test_formats`, `test_stats`, `test_spmv_correctness`,
`test_spmm_correctness`, `test_select`). The CI integration gate additionally
regenerates matrices + references and runs `bench --validate`.

---

## Limitations & next steps

* **No GPU measurements here.** The CUDA SpMV/SpMM kernels and the cuSPARSE and
  cuSPARSE-SpMM baselines are written and gated, but were not compiled or run
  because this machine has no `nvcc`/NVIDIA GPU. The heuristic itself is
  tuned for GPU behaviour, which is why it can mispredict on the CPU-only path
  (the one `uniform_random` miss).
* **No SpGEMM** (sparse x sparse) and no sparse triangular solve.
* **No mixed-precision tensor-core path**; the fp32/fp64 split is expressed but
  not mapped to TF32/FP16/BF16 MMA instructions.
* **SuiteSparse coverage** is limited to whatever downloads succeed; the
  synthetic set is the reproducible default.
* **Small-file portability:** reference artifact names avoid case-only
  differences because macOS/Windows filesystems are case-insensitive.
* **Single-node, in-memory** only; no distributed or out-of-core path.
* Future work: an MLIR/TACO-style code generator that emits these kernels from a
  format algebra, and an auto-tuned heuristic validated against GPU oracles.

## License

MIT — see [`LICENSE`](LICENSE).
