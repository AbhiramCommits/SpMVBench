#!/usr/bin/env bash
# Profile two representative SpMV kernels with Nsight Compute.
#
# On a machine without NVIDIA hardware/CUDA (e.g. the Apple M3 used to develop
# this repo) `ncu` is absent; in that case we record why and exit 0 so the
# pipeline still succeeds.
#
# The profile is deliberately bounded: one representative matrix, one matching
# launch per kernel, so `--set full` cannot expand into thousands of launches.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/results"
SUMMARY="$RESULTS_DIR/ncu_summary.txt"
BENCH="$ROOT/build/bench"
MATRIX="${SPMVBENCH_PROFILE_MATRIX:-banded_100k}"

mkdir -p "$RESULTS_DIR"

if ! command -v ncu >/dev/null 2>&1; then
  {
    echo "Nsight Compute (ncu) is not installed on this machine."
    echo "A CUDA-capable NVIDIA GPU is also absent, so the GPU kernels could"
    echo "not be profiled (they are source-complete but gated behind"
    echo "SPMVBENCH_ENABLE_CUDA)."
    echo
    echo "Diagnostics:"
    echo "  \$ nvcc --version"
    nvcc --version 2>&1 | sed 's/^/    /' || echo "    nvcc: command not found"
    echo "  \$ nvidia-smi"
    nvidia-smi 2>&1 | sed 's/^/    /' || echo "    nvidia-smi: command not found"
  } > "$SUMMARY"
  echo "ncu (Nsight Compute) not found -> GPU profiling unavailable on this host."
  echo "Wrote explanation to $SUMMARY"
  exit 0
fi

if [ ! -x "$BENCH" ]; then
  echo "bench binary not found at $BENCH; build first with 'cmake --build build'."
  exit 0
fi

if [ ! -f "$ROOT/data/$MATRIX.mtx" ]; then
  echo "data/$MATRIX.mtx not found; run tools/gen_matrices.py first."
  exit 0
fi

# Fresh summary: do not append to a stale file from another machine.
: > "$SUMMARY"
echo "Nsight Compute profile on $MATRIX (one launch per kernel)" >> "$SUMMARY"

run_profile() {
  local tag="$1"
  local regex="$2"
  local report="$RESULTS_DIR/ncu_$tag"
  echo "== ncu: $tag ($regex) ==" | tee -a "$SUMMARY"
  ncu --set full --kernel-name-base function --kernel-name "regex:$regex" \
      --launch-count 1 -o "$report" \
      "$BENCH" --bench --gpu-only --iters 1 --warmup 0 \
      "$ROOT/data/$MATRIX.mtx" 2>&1 | tee -a "$SUMMARY"
  if [ -f "${report}.ncu-rep" ]; then
    ncu --import "${report}.ncu-rep" --page details >> "$SUMMARY" 2>&1 || true
  fi
}

run_profile csr_scalar csr_scalar_kernel
run_profile ell ell_kernel

echo "Wrote $SUMMARY"
