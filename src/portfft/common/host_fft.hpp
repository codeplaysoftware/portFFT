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

#ifndef PORTFFT_COMMON_HOST_FFT_HPP
#define PORTFFT_COMMON_HOST_FFT_HPP

#include "portfft/defines.hpp"
#include <complex>

namespace portfft {
namespace detail {

/**
 * Host Naive DFT. Works OOP only
 * @tparam T Scalar Type
 * @param input input pointer
 * @param output output pointer
 * @param fft_size fft size
 */
template <typename T>
void host_naive_dft(std::complex<T>* input, std::complex<T>* output, std::size_t fft_size) {
  using complex_t = std::complex<T>;
  for (std::size_t i = 0; i < fft_size; i++) {
    complex_t temp = complex_t(0, 0);
    for (std::size_t j = 0; j < fft_size; j++) {
      complex_t multiplier =
          complex_t(static_cast<T>(std::cos((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size))),
                    static_cast<T>(std::sin((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size))));
      temp += input[j] * multiplier;
    }
    output[i] = temp;
  }
}
}  // namespace detail
}  // namespace portfft

#endif
