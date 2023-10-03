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

#include <enums.hpp>
#include <kernels.hpp>

#include <sycl/sycl.hpp>

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
template <template <typename, domain, direction, detail::memory, detail::layout, detail::layout,
                    detail::apply_load_modifier, detail::apply_store_modifier, detail::apply_scale_factor, int>
          class Kernel,
          typename Scalar, domain Domain, int SubgroupSize>
std::vector<sycl::kernel_id> get_ids() {
  std::vector<sycl::kernel_id> ids;
#define PORTFFT_GET_ID(DIRECTION, MEMORY, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, STORE_MODIFIER, SCALE_FACTOR)         \
  try {                                                                                                               \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, \
                                             STORE_MODIFIER, SCALE_FACTOR, SubgroupSize>>());                         \
  } catch (...) {                                                                                                     \
  }

#define GENERATE_KERNELS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, STORE_MODIFIER)                      \
  PORTFFT_GET_ID(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, STORE_MODIFIER, apply_scale_factor::APPLIED) \
  PORTFFT_GET_ID(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, STORE_MODIFIER, apply_scale_factor::NOT_APPLIED)

#define INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER)      \
  GENERATE_KERNELS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, apply_store_modifier::APPLIED) \
  GENERATE_KERNELS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, LOAD_MODIFIER, apply_store_modifier::NOT_APPLIED)

#define INSTANTIATE_LAYOUTOUT_MODIFIERS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT)                              \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, apply_load_modifier::APPLIED) \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, LAYOUT_IN, LAYOUT_OUT, apply_load_modifier::NOT_APPLIED)

#define INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, LAYOUT_IN)                \
  INSTANTIATE_LAYOUTOUT_MODIFIERS(DIR, MEM, LAYOUT_IN, layout::BATCH_INTERLEAVED) \
  INSTANTIATE_LAYOUTOUT_MODIFIERS(DIR, MEM, LAYOUT_IN, layout::PACKED)

#define INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, MEM)                          \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, layout::BATCH_INTERLEAVED) \
  INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS(DIR, MEM, layout::PACKED)

#define INSTANTIATE_DIRECTION_MEM_LAYOUTS_MODIFIERS(DIR) \
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, memory::USM)    \
  INSTANTIATE_MEM_LAYOUTS_MODIFIERS(DIR, memory::BUFFER)

  INSTANTIATE_DIRECTION_MEM_LAYOUTS_MODIFIERS(direction::FORWARD)
  INSTANTIATE_DIRECTION_MEM_LAYOUTS_MODIFIERS(direction::BACKWARD)
#undef PORTFFT_GET_ID
#undef GENERATE_KERNELS
#undef INSTANTITATE_LOAD_MODIFIER_MODIFIERS
#undef INSTANTIATE_LAYOUTIN_LAYOUT_MODIFIERS
#undef INSTANTIATE_MEM_LAYOUTS_MODIFIERS
#undef INSTANTIATE_DIRECTION_MEM_LAYOUTS_MODIFIERS
  return ids;
}

template <typename Scalar, domain Domain, int SubgroupSize>
void get_transpose_kernel_ids(std::vector<sycl::kernel_id>& ids) {
#define PORTFFT_GET_TRANSPOSE_ID(MEMORY)                                                          \
  try {                                                                                           \
    ids.push_back(sycl::get_kernel_id<transpose_kernel<Scalar, Domain, MEMORY, SubgroupSize>>()); \
  } catch (...) {                                                                                 \
  }
  PORTFFT_GET_TRANSPOSE_ID(memory::USM);
  PORTFFT_GET_TRANSPOSE_ID(memory::BUFFER);
#undef PORTFFT_GET_TRANSPOSE_ID
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
#endif
