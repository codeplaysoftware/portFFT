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

#ifndef PORTFFT_COMMON_HELPERS_HPP
#define PORTFFT_COMMON_HELPERS_HPP

#include <sycl/sycl.hpp>

#include <type_traits>

#include "portfft/defines.hpp"

namespace portfft::detail {

/**
 * Implements a loop that will be fully unrolled.
 * @tparam Start starting value of loop counter
 * @tparam Stop loop counter value before which the loop terminates
 * @tparam Step Increment of the loop counter
 * @tparam Functor type of the callable
 * @param funct functor containing body of the loop. Should accept one value - the loop counter. Should have
 * __attribute__((always_inline)).
 */
template <Idx Start, Idx Stop, Idx Step, typename Functor>
PORTFFT_INLINE void unrolled_loop(Functor&& funct) {
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
inline T divide_ceil(T dividend, T divisor) {
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
inline T round_up_to_multiple(T value, T factor) {
  return divide_ceil(value, factor) * factor;
}

/**
 * Cast a raw pointer to a global sycl::multi_ptr.
 * The multi_ptr is using the legacy decoration for now as this is better supported.
 *
 * @tparam T Pointer type
 * @param ptr Raw pointer to cast to multi_ptr
 * @return sycl::multi_ptr
 */
template <typename T>
inline auto get_global_multi_ptr(T ptr) {
  return sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::legacy>(ptr);
}

/**
 * Cast a raw pointer to a local sycl::multi_ptr.
 * The multi_ptr is using the legacy decoration for now as this is better supported.
 *
 * @tparam T Pointer type
 * @param ptr Raw pointer to cast to multi_ptr
 * @return sycl::multi_ptr
 */
template <typename T>
inline auto get_local_multi_ptr(T ptr) {
  return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::legacy>(ptr);
}

template <typename T>
T* get_access(T* ptr, sycl::handler&) {
  return ptr;
}

template <typename T, std::enable_if_t<std::is_const<T>::value>* = nullptr>
auto get_access(sycl::buffer<T, 1> buf, sycl::handler& cgh) {
  return buf.template get_access<sycl::access::mode::read>(cgh);
}

template <typename T, std::enable_if_t<!std::is_const<T>::value>* = nullptr>
auto get_access(sycl::buffer<T, 1> buf, sycl::handler& cgh) {
  return buf.template get_access<sycl::access::mode::write>(cgh);
}

template <typename T, typename TSrc>
T* reinterpret(TSrc* ptr) {
  return reinterpret_cast<T*>(ptr);
}

template <typename T, typename TSrc>
auto reinterpret(const sycl::buffer<TSrc, 1>& buf) {
  static_assert(sizeof(TSrc) % sizeof(T) == 0, "Can only reinterpret from a type, size of which is a multiple of size of the target type!");
  return buf.template reinterpret<T, 1>(sizeof(TSrc) / sizeof(T) * buf.size());
}

/**
 * Multiplies 2 complex numbers
 *
 * @tparam T Scalar Type
 * @param input_real Input real part
 * @param input_imag Input imag part
 * @param multiplier_real Multiplier real part
 * @param multiplier_imag Multiplier imag part
 * @param output_real output real part
 * @param output_imag output imag part
 */
template <typename T>
PORTFFT_INLINE void multiply_complex(const T input_real, const T input_imag, const T multiplier_real,
                                     const T multiplier_imag, T& output_real, T& output_imag) {
  output_real = input_real * multiplier_real - input_imag * multiplier_imag;
  output_imag = input_real * multiplier_imag + input_imag * multiplier_real;
}

/**
 * Log2 of an Idx
 * @param x The value to get log2 of. Must be more than 0.
 */
PORTFFT_INLINE constexpr Idx int_log2(Idx x) {
  Idx y = 0;
  for (; x > 1; x /= 2) {
    ++y;
  }
  return y;
}
}  // namespace portfft::detail

#endif
