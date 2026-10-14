#pragma once

// Common value-type traits and aliases for SpMVBench.
//
// The library is templated on two independent type parameters everywhere it
// matters:
//   * `T`   - the *storage* type of the sparse matrix values and the input
//             vectors (fp32 vs fp64).
//   * `Acc` - the *accumulator* type used inside the inner product.
//
// Keeping these separate is what lets us express the "fp32 values, fp64
// accumulate" mixed-precision mode that the error analysis studies.

#include <cstdint>
#include <type_traits>

namespace spmvbench {

using Index = std::int32_t;

// Is `U` a value type we are willing to instantiate kernels for?
template <typename U>
constexpr bool is_value_type_v =
    std::is_same_v<U, float> || std::is_same_v<U, double>;

// Human readable name for a value type, used in reports.
template <typename U>
constexpr const char* dtype_name() {
  static_assert(is_value_type_v<U>, "unsupported value type");
  if constexpr (std::is_same_v<U, float>)
    return "fp32";
  if constexpr (std::is_same_v<U, double>)
    return "fp64";
  return "unknown";
}

}  // namespace spmvbench
