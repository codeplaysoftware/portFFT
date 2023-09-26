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

#include <defines.hpp>
#include <descriptor.hpp>
#include <enums.hpp>
#include <vector>

namespace portfft {

namespace detail {

/**
 * Get kernel ids for the implementation used.
 *
 * @tparam kernel which base template for kernel to use
 * @tparam SubgroupSize size of the subgroup
 * @param ids vector of kernel ids
 */
template <
    template <typename, domain, direction, detail::memory, detail::transpose, detail::transpose, bool, bool, bool, int>
    class Kernel,
    typename Scalar, domain Domain, int SubgroupSize>
std::vector<sycl::kernel_id> get_ids() {
#define PORTFFT_GET_ID(DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, SCALE_FACTOR) \
  try {                                                                                                             \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT,        \
                                             LOAD_MODIFIER, STORE_MODIFIER, SCALE_FACTOR, SubgroupSize>>());        \
  } catch (...) {                                                                                                   \
  }

#define GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER) \
  PORTFFT_GET_ID(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, true)   \
  PORTFFT_GET_ID(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, false)

#define INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER) \
  GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, true)                     \
  GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, false)

#define INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT)   \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, true) \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, false)

#define INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, TRANSPOSE_IN)         \
  INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, transpose::TRANSPOSED) \
  INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, transpose::NOT_TRANSPOSED)

#define INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, MEM)                         \
  INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, transpose::TRANSPOSED) \
  INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, transpose::NOT_TRANSPOSED)

#define INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(DIR) \
  INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, memory::USM)    \
  INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, memory::BUFFER)

  std::vector<sycl::kernel_id> ids;
  INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(direction::FORWARD)
  INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(direction::BACKWARD)
#undef PORTFFT_GET_ID
#undef GENERATE_KERNELS
#undef INSTANTITATE_LOAD_MODIFIER_MODIFIERS
#undef INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS
#undef INSTANTIATE_MEM_TRANSPOSES_MODIFIERS
#undef INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS
  return ids;
}

template <typename Scalar, domain Domain, int SubgroupSize>
void get_transpose_kernel_ids(std::vector<sycl::kernel_id>& ids) {
  ids.push_back(sycl::get_kernel_id<transpose_kernel<Scalar, Domain, memory::USM, SubgroupSize>>());
  ids.push_back(sycl::get_kernel_id<transpose_kernel<Scalar, Domain, memory::BUFFER, SubgroupSize>>());
}

template <typename F, typename G>
struct factorize_input_struct {
  static void execute(std::size_t input_size, F fits_in_target_level, G select_impl) {
    std::size_t fact_1 = input_size;
    if (fits_in_target_level(input_size)) {
      select_impl(input_size);
      return;
    }
    if ((detail::factorize(fact_1) == 1)) {
      throw std::runtime_error("Large prime sized factors are not supported at the moment");
    }
    do {
      fact_1 = detail::factorize(fact_1);
    } while (!fits_in_target_level(fact_1));
    select_impl(fact_1);
    factorize_input_struct<F, G>::execute(input_size / fact_1, fits_in_target_level, select_impl);
  }
};

}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_UTILS_HPP
