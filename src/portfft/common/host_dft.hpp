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

#ifndef PORTFFT_COMMON_HOST_DFT_HPP
#define PORTFFT_COMMON_HOST_DFT_HPP

#include "portfft/common/helpers.hpp"
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
      // Not using sycl::cospi / sycl::sinpi as std::cos/std::sin provides better accuracy in float and double tests
      double theta = -2 * M_PI * static_cast<double>(i * j) / static_cast<double>(fft_size);
      complex_t multiplier = complex_t(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
      temp += input[j] * multiplier;
    }
    output[i] = temp;
  }
}

/**
 * Host implementation of the cooley tukey algorithm. Handles prime values using the naive implementation
 * @tparam T Scalar type for std::complex
 * @param input pointer of type std::complex<T> containing the input values
 * @param output otuput pointer of type std::complex<T> containing the output values
 * @param fft_size DFT size
 */
template <typename T>
void host_cooley_tukey(std::complex<T>* input, std::complex<T>* output, std::size_t fft_size) {
  std::size_t n = detail::factorize(fft_size);
  if (n == 1 || fft_size <= 8) {
    host_naive_dft(input, output, fft_size);
    return;
  }

  std::size_t m = fft_size / n;
  std::size_t scratch_size = n > m ? n : m;
  std::vector<std::complex<T>> scratch_space(scratch_size);
  std::vector<std::complex<T>> scratch_space2(scratch_size);
  std::vector<std::complex<T>> output_buffer(fft_size);

  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      scratch_space[j] = input[j * m + i];
    }
    host_cooley_tukey(scratch_space.data(), scratch_space2.data(), n);
    for (std::size_t j = 0; j < n; j++) {
      output[j * m + i] = scratch_space2[j];
    }
  }

  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < m; j++) {
      // Not using sycl::cospi / sycl::sinpi as std::cos/std::sin provides better accuracy in float and double tests
      double theta = -2 * M_PI * static_cast<double>(i * j) / static_cast<double>(n * m);
      output[i * m + j] *= std::complex<T>(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
    }
  }

  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < m; j++) {
      scratch_space[j] = output[i * m + j];
    }
    host_cooley_tukey(scratch_space.data(), scratch_space2.data(), m);
    for (std::size_t j = 0; j < m; j++) {
      output_buffer[i * m + j] = scratch_space2[j];
    }
  }

  for (std::size_t i = 0; i < fft_size; i++) {
    std::size_t j = i / n;
    std::size_t k = i % n;
    output[i] = output_buffer[k * m + j];
  }
}
}  // namespace detail
}  // namespace portfft

#endif
