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

#include "common/memory_views.hpp"
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

/**
 * Function which handles factorizing a size till it can be dispatched to one of the existing implementations
 * @tparam F Decltype of function being passed
 * @param factor_size Length of the factor
 * @param check_and_select_target_level Function which checks whether the factor can fit in one of the existing
 * implementations
 * The function should accept factor size and whether the data layout in the local memory would be in a batch
 * interleaved format, and whether or not load modifers be used. and should return a boolean indicating whether or not
 * the factor size can fit in any of the implementation.
 * @param transposed whether or not the factor will be computed in a BATCH_INTERLEAVED format
 * @param encountered_prime whether or not a large prime was encountered during factorization
 * @param requires_load_modifier whether or not load modifier will be required.
 * @return Largest factor that was possible to fit in either or workitem/subgroup level FFTs
 */
template <typename F>
IdxGlobal factorize_input_impl(IdxGlobal factor_size, F&& check_and_select_target_level, bool transposed,
                               bool& encountered_prime, bool requires_load_modifier) {
  IdxGlobal fact_1 = factor_size;
  if (check_and_select_target_level(fact_1, transposed, requires_load_modifier)) {
    return fact_1;
  }
  if ((detail::factorize(fact_1) == 1)) {
    encountered_prime = true;
    return factor_size;
  }
  do {
    fact_1 = detail::factorize(fact_1);
    if (fact_1 == 1) {
      encountered_prime = true;
      return factor_size;
    }
  } while (!check_and_select_target_level(fact_1, transposed, requires_load_modifier));
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
 * @param requires_load_modifier whether or not load modifier will be required.
 * @return whether or not a large prime was encounterd during factorization.
 */
template <typename F>
bool factorize_input(IdxGlobal input_size, F&& check_and_select_target_level, bool requires_load_modifier = false) {
  bool encountered_prime = false;
  if (detail::factorize(input_size) == 1) {
    encountered_prime = true;
    return encountered_prime;
  }
  IdxGlobal temp = 1;
  while (input_size / temp != 1) {
    if (encountered_prime) {
      return encountered_prime;
    }
    temp *= factorize_input_impl(input_size / temp, check_and_select_target_level, true, encountered_prime,
                                 requires_load_modifier);
    requires_load_modifier = false;
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
  return std::shared_ptr<T>(sycl::malloc_device<T>(size, queue), [captured_queue = queue](T* ptr) {
    if (ptr != nullptr) {
      sycl::free(ptr, captured_queue);
    }
  });
}

/**
 * Calculate the number of groups or bank lines of PORTFFT_N_LOCAL_BANKS between each padding in local memory,
 * specifically for reducing bank conflicts when reading values from the columns of a 2D data layout. e.g. If there are
 * 64 complex elements in a row, then the consecutive values in the same column are 128 floats apart. There are 32
 * banks, each the size of a float, so we only want a padding float every 128/32=4 bank lines to read along the column
 * without bank conflicts.
 *
 * @tparam T Input type to the function
 * @param row_size the size in bytes of the row. 32 std::complex<float> values would probably have a size of 256 bytes.
 * @return the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
template <typename T>
constexpr T bank_lines_per_pad_wg(T row_size) {
  constexpr T BankLineSize = sizeof(float) * PORTFFT_N_LOCAL_BANKS;
  if (row_size % BankLineSize == 0) {
    return row_size / BankLineSize;
  }
  // There is room for improvement here. E.G if row_size was half of BankLineSize then maybe you would still want 1
  // pad every bank group.
  return 1;
}

/**
 * @brief Gets the cumulative local memory usage for a particular level
 * @tparam Scalar Scalar type
 * @param level level to get the cumulative local memory usage for
 * @param factor_size Factor size
 * @param is_batch_interleaved Will the data be in a batch interleaved format in local memory
 * @param is_load_modifier_applied Is load modifier applied
 * @param is_store_modifier_applied Is store modifier applied
 * @param workgroup_size workgroup size with which the kernel will be launched
 * @return cumulative local memory usage in terms on number of scalars in local memory.
 */
inline Idx get_local_memory_usage(detail::level level, Idx factor_size, bool is_batch_interleaved,
                                  bool is_load_modifier_applied, bool is_store_modifier_applied, Idx subgroup_size,
                                  Idx workgroup_size) {
  Idx local_memory_usage = 0;
  switch (level) {
    case detail::level::WORKITEM: {
      // This will use local memory for load / store modifiers in the future.
      if (!is_batch_interleaved) {
        local_memory_usage += detail::pad_local(2 * factor_size * workgroup_size, 1);
      }
    } break;
    case detail::level::SUBGROUP: {
      local_memory_usage += 2 * factor_size;
      Idx fact_sg = factorize_sg(factor_size, subgroup_size);
      Idx num_ffts_in_sg = subgroup_size / fact_sg;
      Idx num_ffts_in_local_mem =
          is_batch_interleaved ? workgroup_size / 2 : num_ffts_in_sg * (workgroup_size / subgroup_size);
      local_memory_usage += detail::pad_local(2 * num_ffts_in_local_mem * factor_size, 1);
      if (is_load_modifier_applied) {
        local_memory_usage += detail::pad_local(2 * num_ffts_in_local_mem * factor_size, 1);
      }
      if (is_store_modifier_applied) {
        local_memory_usage += detail::pad_local(2 * num_ffts_in_local_mem * factor_size, 1);
      }
    } break;
    case detail::level::WORKGROUP: {
      Idx n = detail::factorize(factor_size);
      Idx m = factor_size / n;
      Idx num_ffts_in_local_mem = is_batch_interleaved ? workgroup_size / 2 : 1;
      local_memory_usage += detail::pad_local(2 * factor_size * num_ffts_in_local_mem, bank_lines_per_pad_wg(m));
    } break;
    default:
      break;
  }
  return local_memory_usage;
}

}  // namespace detail
}  // namespace portfft
#endif
