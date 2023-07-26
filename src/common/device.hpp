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

#ifndef PORTFFT_COMMON_DEVICE_HPP
#define PORTFFT_COMMON_DEVICE_HPP

#include <common/helpers.hpp>
#include <common/transfers.hpp>

namespace portfft {
namespace detail {
// TODO: Try using SYCL-Graphs instead of manual fusion. In that case, twiddle multiplication would be a separate kernel
// This function will be used in both multi factor FFTs as well as to fuse bluestein's pointwise multiply kernel
/**
 * Generic pointwise multiplication function, which multiplies inplace with values from another array.
 *
 * @tparam T Type of scalar
 *
 * @param priv array which will be multiplied
 * @param scales values with which it will be multiplied with
 * @param priv_index Index for private array
 * @param scale_index index for the array which will hold multiplicative values
 */
template <typename T>
__attribute__((always_inline)) inline void pointwise_multiply(T* priv,  T* scales, std::size_t priv_index,
                                                              std::size_t scale_index) {
  using T_vec = sycl::vec<T, 2>;  // Assmuing complex inputs for now
  const T_vec complex_scale_value = reinterpret_cast<T_vec*>(scales)[scale_index];
  T tmp_real = priv[priv_index];
  priv[priv_index] = tmp_real * complex_scale_value[0] - priv[priv_index + 1] * complex_scale_value[1];
  priv[priv_index] = tmp_real * complex_scale_value[1] + priv[priv_index] * complex_scale_value[0];
}
}  // namespace detail
}  // namespace portfft

#endif