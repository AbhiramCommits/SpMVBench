#!/usr/bin/env bash
# Profile the two most interesting kernels with Nsight Compute.
#
# On a machine without NVIDIA hardware (like the Apple M3 used to build this
# repository) `ncu` is absent; in that case we record why and exit 0 so the
# end-to-end pipeline still succeeds.

set -u

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/ncu_summary.txt"

if ! command -v ncu >/dev/null 2>&1; then
  {
    echo "Nsight Compute (ncu) is not installed on this machine."
    echo "A CUDA-capable NVIDIA GPU is also absent, so the GPU kernels could"
    echo "not be profiled (they are source-complete but gated behind"
    echo "SPMVBENCH_ENABLE_CUDA and were not compiled or run here)."
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

BENCH="${RESULTS_DIR%/results}/build/bench"
if [ ! -x "$BENCH" ]; then
  echo "bench binary not found at $BENCH; build first with 'cmake --build build'."
  exit 0
fi

KERNELS="csr_vector ell"
for k in $KERNELS; do
  echo "== ncu: $k =="
  ncu --set full --kernel-name-base function \
      -o "$RESULTS_DIR/ncu_$k" \
      "$BENCH" --bench --iters 1 --warmup 0 2>&1 | tee -a "$SUMMARY"
  ncu --import "$RESULTS_DIR/ncu_$k.ncu-rep" --page details \
      >> "$SUMMARY" 2>&1 || true
done

echo "Wrote $SUMMARY"
