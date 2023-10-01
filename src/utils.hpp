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
template <template <typename, domain, direction, detail::memory, detail::transpose, detail::transpose,
                    detail::apply_load_modifier, detail::apply_store_modifier, detail::apply_scale_factor, int>
          class Kernel,
          typename Scalar, domain Domain, int SubgroupSize>
void get_ids(std::vector<sycl::kernel_id>& ids) {
#define PORTFFT_GET_ID(DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, SCALE_FACTOR) \
  try {                                                                                                             \
    ids.push_back(sycl::get_kernel_id<Kernel<Scalar, Domain, DIRECTION, MEMORY, TRANSPOSE_IN, TRANSPOSE_OUT,        \
                                             LOAD_MODIFIER, STORE_MODIFIER, SCALE_FACTOR, SubgroupSize>>());        \
  } catch (...) {                                                                                                   \
  }

#define GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER)                      \
  PORTFFT_GET_ID(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, apply_scale_factor::APPLIED) \
  PORTFFT_GET_ID(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, STORE_MODIFIER, apply_scale_factor::NOT_APPLIED)

#define INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER)      \
  GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, apply_store_modifier::APPLIED) \
  GENERATE_KERNELS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, LOAD_MODIFIER, apply_store_modifier::NOT_APPLIED)

#define INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT)                           \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, apply_load_modifier::APPLIED) \
  INSTANTITATE_LOAD_MODIFIER_MODIFIERS(DIR, MEM, TRANSPOSE_IN, TRANSPOSE_OUT, apply_load_modifier::NOT_APPLIED)

#define INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, TRANSPOSE_IN)         \
  INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, transpose::TRANSPOSED) \
  INSTANTIATE_TRANSPOSEOUT_MODIFIERS(DIR, MEM, TRANSPOSE_IN, transpose::NOT_TRANSPOSED)

#define INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, MEM)                         \
  INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, transpose::TRANSPOSED) \
  INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS(DIR, MEM, transpose::NOT_TRANSPOSED)

#define INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(DIR) \
  INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, memory::USM)    \
  INSTANTIATE_MEM_TRANSPOSES_MODIFIERS(DIR, memory::BUFFER)

  INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(direction::FORWARD)
  INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS(direction::BACKWARD)
#undef PORTFFT_GET_ID
#undef GENERATE_KERNELS
#undef INSTANTITATE_LOAD_MODIFIER_MODIFIERS
#undef INSTANTIATE_TRANSPOSEIN_TRANSPOSE_MODIFIERS
#undef INSTANTIATE_MEM_TRANSPOSES_MODIFIERS
#undef INSTANTIATE_DIRECTION_MEM_TRANSPOSES_MODIFIERS
}
}  // namespace detail
}  // namespace portfft
#endif