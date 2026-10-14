#pragma once

// Tiny OpenMP compatibility shim so the same sources build with or without
// OpenMP (and with Apple clang + libomp, GCC, or MSVC).

#if defined(_OPENMP)
#include <omp.h>
#define SPMVBENCH_HAS_OPENMP 1
#define SPMVBENCH_SET_THREADS(n)      \
  do {                                \
    if ((n) > 0) omp_set_num_threads((n)); \
  } while (0)
#else
#define SPMVBENCH_HAS_OPENMP 0
#define SPMVBENCH_SET_THREADS(n) \
  do {                           \
    (void)(n);                   \
  } while (0)
#endif
