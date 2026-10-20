#!/usr/bin/env bash
# Profile selected SpMV kernels with Nsight Compute.
#
# Defaults match the SpMVBench profiling request:
#   matrices: powerlaw_skew uniform_random
#   kernels:  csr_vector csr_merge ell
#
# Raw reports and imported details are written under results/ncu/:
#   <matrix>__<kernel>.ncu-rep       raw Nsight Compute report
#   <matrix>__<kernel>.details.txt   ncu --import --page details text
#   <matrix>__<kernel>.log           ncu console output
#   summary.txt                      combined device info + details
#
# On a machine without ncu (e.g. the Apple M3 used to develop this repo) the
# script records why and exits 0 so the pipeline still succeeds.
#
# Override the set with:
#   SPMVBENCH_PROFILE_MATRICES="powerlaw_skew uniform_random" \
#   SPMVBENCH_PROFILE_KERNELS="csr_vector csr_merge ell" \
#   bash scripts/profile.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/results"
OUT_DIR="$RESULTS_DIR/ncu"
SUMMARY="$OUT_DIR/summary.txt"
BENCH="$ROOT/build/bench"
MATRICES="${SPMVBENCH_PROFILE_MATRICES:-powerlaw_skew uniform_random}"
KERNELS="${SPMVBENCH_PROFILE_KERNELS:-csr_vector csr_merge ell}"

mkdir -p "$OUT_DIR"

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
  cp "$SUMMARY" "$RESULTS_DIR/ncu_summary.txt" 2>/dev/null || true
  echo "ncu (Nsight Compute) not found -> GPU profiling unavailable on this host."
  echo "Wrote explanation to $SUMMARY"
  exit 0
fi

if [ ! -x "$BENCH" ]; then
  echo "bench binary not found at $BENCH; build first with 'cmake --build build'."
  exit 0
fi

# Fresh summary: never append to a stale file from another machine/run.
: > "$SUMMARY"
{
  echo "Nsight Compute profiling"
  echo "Matrices: $MATRICES"
  echo "Kernels:  $KERNELS"
  echo
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || true
  nvcc --version 2>&1 || true
  echo
} >> "$SUMMARY"

run_profile() {
  local matrix="$1"
  local kernel="$2"
  local tag="${matrix}__${kernel}"
  local report="$OUT_DIR/$tag"
  local log="$OUT_DIR/$tag.log"
  local details="$OUT_DIR/$tag.details.txt"

  echo "== ncu: $tag ==" | tee -a "$SUMMARY"

  # One matching launch per kernel. --set full gives all SOL/occupancy/warp
  # sections; --kernel-name-base function makes the regex match the C++ name.
  ncu --set full \
      --kernel-name-base function \
      --kernel-name "regex:${kernel}_kernel" \
      --launch-count 1 \
      -o "$report" \
      "$BENCH" --bench --gpu-only --iters 1 --warmup 0 \
      "$ROOT/data/$matrix.mtx" > "$log" 2>&1 || true

  if [ -f "${report}.ncu-rep" ]; then
    ncu --import "${report}.ncu-rep" --page details > "$details" 2>&1 || true
    {
      echo "---- $tag (details) ----"
      cat "$details"
      echo
    } >> "$SUMMARY"
  else
    echo "WARNING: no report produced for $tag (see $log)" | tee -a "$SUMMARY"
  fi
}

for matrix in $MATRICES; do
  if [ ! -f "$ROOT/data/$matrix.mtx" ]; then
    echo "data/$matrix.mtx not found; run tools/gen_matrices.py first." | tee -a "$SUMMARY"
    continue
  fi
  for kernel in $KERNELS; do
    run_profile "$matrix" "$kernel"
  done
done

# Keep the legacy aggregate path populated for the rest of the pipeline.
cp "$SUMMARY" "$RESULTS_DIR/ncu_summary.txt" 2>/dev/null || true
echo "Wrote $SUMMARY"
echo "Raw reports under $OUT_DIR/"
