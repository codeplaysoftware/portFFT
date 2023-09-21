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

#ifndef PORTFFT_COMMON_TWIDDLE_CALC_HPP
#define PORTFFT_COMMON_TWIDDLE_CALC_HPP

#include <cmath>
#include <complex>

namespace portfft {
namespace detail {

/**
 * Calculates a twiddle factor.
 * @tparam T floating point type to use
 * @tparam TIndex Index type
 * @param n which twiddle factor to calculate
 * @param total total number of twiddles
 */
template <typename T, typename TIndex>
std::complex<T> calculate_twiddle(TIndex n, TIndex total) {
  T theta = static_cast<T>(-2) * static_cast<T>(n) / static_cast<T>(total);
  return {sycl::cospi(theta), sycl::sinpi(theta)};
}

}  // namespace detail
}  // namespace portfft

#endif
