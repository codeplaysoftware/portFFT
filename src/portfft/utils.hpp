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

#include "common/logging.hpp"
#include "defines.hpp"
#include "enums.hpp"

namespace portfft {
namespace detail {
template <typename Scalar, detail::memory>
class transpose_kernel;

/**
 * Get kernel ids for the implementation used.
 *
 * @tparam kernel which base template for kernel to use
 * @tparam SubgroupSize size of the subgroup
 * @return vector of kernel ids
 */
template <template <typename, domain, detail::memory, detail::layout, detail::layout, Idx> class Kernel,
          typename Scalar, domain Domain, Idx SubgroupSize>
std::vector<sycl::kernel_id> get_ids() {
  PORTFFT_LOG_FUNCTION_ENTRY();
  std::vector<sycl::kernel_id> ids;
#define PORTFFT_GET_ID(MEMORY, LAYOUT_IN, LAYOUT_OUT)                                                          \
  try {                                                                                                        \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, MEMORY, LAYOUT_IN, LAYOUT_OUT, SubgroupSize>>()); \
  } catch (...) {                                                                                              \
  }

#define INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(MEM, LAYOUT_IN) \
  PORTFFT_GET_ID(MEM, LAYOUT_IN, layout::BATCH_INTERLEAVED)   \
  PORTFFT_GET_ID(MEM, LAYOUT_IN, layout::PACKED)

#define INSTANTIATE_MEM_LAYOUTS_MODIFIERS(MEM)                          \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(MEM, layout::BATCH_INTERLEAVED) \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(MEM, layout::PACKED)

#ifdef PORTFFT_ENABLE_BUFFER_BUILDS
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(memory::USM)
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(memory::BUFFER)
#else
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(memory::USM)
#endif

#undef PORTFFT_GET_ID
#undef INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS
#undef INSTANTIATE_MEM_LAYOUTS_MODIFIERS
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

/**
 * Function which handles factorizing a size till it can be dispatched to one of the existing implementations
 * @tparam F Decltype of function being passed
 * @param factor_size Length of the factor
 * @param check_and_select_target_level Function which checks whether the factor can fit in one of the existing
 * implementations
 * The function should accept factor size and whether it would be have a BATCH_INTERLEAVED layout or not as an input,
 * and should return a boolean indicating whether or not the factor size can fit in any of the implementation.
 * @param transposed whether or not the factor will be computed in a BATCH_INTERLEAVED format
 * @param encountered_prime_factor A flag to be set if a prime factor which cannot be dispatched to workitem
 * implementation
 * @return A factor of the committed size which can be dispatched to either workitem or subgroup implementation
 */
template <typename F>
IdxGlobal factorize_input_impl(IdxGlobal factor_size, F&& check_and_select_target_level, bool transposed,
                               bool& encountered_prime_factor) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  IdxGlobal fact_1 = factor_size;
  if (check_and_select_target_level(fact_1, transposed)) {
    return fact_1;
  }
  if ((detail::factorize(fact_1) == 1)) {
    encountered_prime_factor = true;
    return fact_1;
  }
  do {
    fact_1 = detail::factorize(fact_1);
    if (fact_1 == 1) {
      encountered_prime_factor = true;
      return fact_1;
    }
  } while (!check_and_select_target_level(fact_1));
  return fact_1;
}

/**
 * Driver function to factorize large inputs for global implementation
 * @tparam F Decltype of the function being passed
 * @param input_size committed_size
 * @param check_and_select_target_level Function which checks whether the factor can fit in one of the existing
 * implementations. The function should accept factor size and whether it would be have a BATCH_INTERLEAVED layout or
 * not as an input, and should return a boolean indicating whether or not the factor size can fit in any of the
 * implementation.
 * @param Whether or not the factorization was successful
 */
template <typename F>
bool factorize_input(IdxGlobal input_size, F&& check_and_select_target_level) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  if (detail::factorize(input_size) == 1) {
    return false;
  }
  IdxGlobal temp = 1;
  bool encountered_prime = false;
  while (input_size / temp != 1) {
    temp *= factorize_input_impl(input_size / temp, check_and_select_target_level, true, encountered_prime);
  }
  return encountered_prime;
}

/**
 * Obtains kernel ids for transpose kernels
 * @tparam Scalar Scalar type
 * @return vector containing sycl::kernel_ids
 */
template <typename Scalar>
std::vector<sycl::kernel_id> get_transpose_kernel_ids() {
  PORTFFT_LOG_FUNCTION_ENTRY();
  std::vector<sycl::kernel_id> ids;
#define PORTFFT_GET_TRANSPOSE_KERNEL_ID(MEMORY)                               \
  try {                                                                       \
    ids.push_back(sycl::get_kernel_id<transpose_kernel<Scalar, (MEMORY)>>()); \
  } catch (...) {                                                             \
  }

  PORTFFT_GET_TRANSPOSE_KERNEL_ID(detail::memory::USM)
  PORTFFT_GET_TRANSPOSE_KERNEL_ID(detail::memory::BUFFER)
#undef PORTFFT_GET_TRANSPOSE_KERNEL_ID
  return ids;
}

/**
 * Utility function to create a shared pointer, with memory allocated on device
 * @tparam T Type of the memory being allocated
 * @param size Number of elements to allocate.
 * @param queue Associated queue
 * @return std::shared_ptr<T>
 */
template <typename T>
inline std::shared_ptr<T> make_shared(std::size_t size, sycl::queue& queue) {
  T* ptr = sycl::malloc_device<T>(size, queue);
  if (ptr != nullptr) {
    return std::shared_ptr<T>(sycl::malloc_device<T>(size, queue), [captured_queue = queue](T* ptr) {
      if (ptr != nullptr) {
        sycl::free(ptr, captured_queue);
      }
    });
  }
  throw internal_error("Could not allocate usm memory of size: ", size * sizeof(T), " bytes");
}

/**
 * Function to get the scale specialization constant.
 * @tparam Scalar Scalar type associated with the committed descriptor
 * @return sycl::specialization_id
 */
template <typename Scalar>
PORTFFT_INLINE constexpr const sycl::specialization_id<Scalar>& get_spec_constant_scale() {
  if constexpr (std::is_same_v<Scalar, float>) {
    return detail::SpecConstScaleFactorFloat;
  } else {
    return detail::SpecConstScaleFactorDouble;
  }
}

/**
 * Checks usm_device_allocations support on device
 * @param dev sycl::device for which USM support is to be checked
 */
void check_usm_support(sycl::device& dev) {
  if (!dev.has(sycl::aspect::usm_device_allocations)) {
    throw unsupported_device("The selected device does not support device USM allocations");
  }
}

}  // namespace detail
}  // namespace portfft
#endif
