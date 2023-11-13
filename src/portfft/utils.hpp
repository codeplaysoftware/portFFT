/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's portFFT
 *
 **************************************************************************/

#ifndef PORTFFT_UTILS_HPP
#define PORTFFT_UTILS_HPP

#include <sycl/sycl.hpp>

#include <limits>
#include <vector>

#include "defines.hpp"
#include "enums.hpp"

namespace portfft {
namespace detail {
/**
 * Get kernel ids for the implementation used.
 *
 * @tparam kernel which base template for kernel to use
 * @tparam SubgroupSize size of the subgroup
 * @return vector of kernel ids
 */
template <template <typename, domain, direction, detail::memory, detail::layout, detail::layout, Idx> class Kernel,
          typename Scalar, domain Domain, Idx SubgroupSize>
std::vector<sycl::kernel_id> get_ids() {
  std::vector<sycl::kernel_id> ids;
#define PORTFFT_GET_ID(DIRECTION, MEMORY, LAYOUT_IN, LAYOUT_OUT)                                                \
  try {                                                                                                         \
    ids.push_back(                                                                                              \
        sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, LAYOUT_IN, LAYOUT_OUT, SubgroupSize>>()); \
  } catch (...) {                                                                                               \
  }

#define INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, LAYOUT_IN) \
  PORTFFT_GET_ID(DIR, MEM, LAYOUT_IN, layout::BATCH_INTERLEAVED)   \
  PORTFFT_GET_ID(DIR, MEM, LAYOUT_IN, layout::PACKED)

#define INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, MEM)                          \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, layout::BATCH_INTERLEAVED) \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, layout::PACKED)

#define INSTANTIATE_DIRECTION_MEM_LAYOUTS(DIR)        \
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, memory::USM) \
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, memory::BUFFER)

  INSTANTIATE_DIRECTION_MEM_LAYOUTS(direction::FORWARD)
  INSTANTIATE_DIRECTION_MEM_LAYOUTS(direction::BACKWARD)
#undef PORTFFT_GET_ID
#undef INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS
#undef INSTANTIATE_MEM_LAYOUTS_MODIFIERS
#undef INSTANTIATE_DIRECTION_MEM_LAYOUTS
  return ids;
}

/**
 * Utility function to check if a value can be casted safely.
 * @tparam InputType Input Type
 * @tparam OutputType Type to be casted to
 * @param x value to be casted
 * @return bool, true if its safe to cast
 */
template <typename InputType, typename OutputType>
constexpr bool can_cast_safely(const InputType& x) {
  static_assert(std::is_signed_v<InputType> && std::is_signed_v<OutputType>);
  if constexpr (sizeof(OutputType) > sizeof(InputType)) {
    return true;
  }
  OutputType x_converted = static_cast<OutputType>(x);
  return (static_cast<InputType>(x_converted) == x);
}

template <typename F>
IdxGlobal factorize_input_impl(IdxGlobal factor_size, F&& check_and_select_target_level, bool transposed) {
  IdxGlobal fact_1 = factor_size;
  if (check_and_select_target_level(fact_1, transposed)) {
    std::cout << fact_1 << std::endl;
    return fact_1;
  }
  if ((detail::factorize(fact_1) == 1)) {
    throw unsupported_configuration("Large prime sized factors are not supported at the moment");
  }
  do {
    fact_1 = detail::factorize(fact_1);
  } while (!check_and_select_target_level(fact_1));
  std::cout << fact_1 << std::endl;
  return fact_1;
}

template <typename F>
void factorize_input(IdxGlobal input_size, F&& check_and_select_target_level) {
  if (detail::factorize(input_size) == 1) {
    throw unsupported_configuration("Large Prime sized FFTs are currently not supported");
  }
  IdxGlobal temp = 1;
  while (true) {
    temp *= factorize_input_impl(input_size / temp, check_and_select_target_level, true);
    if (input_size / temp == 1) {
      break;
    }
  }
}

}  // namespace detail
}  // namespace portfft
#endif
