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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_COMMON_HELPERS_HPP
#define SYCL_FFT_COMMON_HELPERS_HPP

#include <sycl/sycl.hpp>
#include <type_traits>

namespace sycl_fft::detail {

/**
 * Implements a loop that will be fully unrolled.
 * @tparam Start starting value of loop counter
 * @tparam Stop loop counter value before which the loop terminates
 * @tparam Step Increment of the loop counter
 * @tparam Functor type of the callable
 * @param funct functor containing body of the loop. Should accept one value - the loop counter. Should have
 * __attribute__((always_inline)).
 */
template <auto Start, auto Stop, auto Step, typename Functor>
void __attribute__((always_inline)) unrolled_loop(Functor&& funct) {
  if constexpr (Start < Stop) {
    funct(Start);
    unrolled_loop<Start + Step, Stop, Step>(funct);
  }
}

/**
 * Divides the value and rounds the result up.
 * @tparam T type of the inputs and the result
 * @param dividend dividend
 * @param divisor divisor
 * @return rounded-up quotient
 */
template <typename T>
inline T divideCeil(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

/**
 * Rounds the value up, so it is divisible by factor.
 * @tparam T type of the inputs and the result
 * @param value value to round up
 * @param factor factor that divides the result
 * @return rounded-up value
 */
template <typename T>
inline T roundUpToMultiple(T value, T factor) {
  return divideCeil(value, factor) * factor;
}

template <typename T>
inline auto get_global_multi_ptr(T ptr) {
  return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::legacy>(ptr);
}

template <typename T>
inline auto get_local_multi_ptr(T ptr) {
  return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::legacy>(ptr);
}

};  // namespace sycl_fft::detail

#endif
