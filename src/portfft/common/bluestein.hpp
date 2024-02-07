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

#ifndef PORTFFT_COMMON_BLUESTEIN_HPP
#define PORTFFT_COMMON_BLUESTEIN_HPP

#include "portfft/common/host_fft.hpp"
#include "portfft/defines.hpp"

#include <cmath>
#include <complex>
#include <sycl/sycl.hpp>

namespace portfft {
namespace detail {
/**
 * Utility function to get chirp signal and fft
 * @tparam T Scalar Type
 * @param ptr Host Pointer containing the load/store modifiers.
 * @param committed_size original problem size
 * @param dimension_size padded size
 */
template <typename T>
void get_fft_chirp_signal(T* ptr, std::size_t committed_size, std::size_t dimension_size) {
  using ctype = std::complex<T>;
  ctype* chirp_signal = (ctype*)calloc(dimension_size, sizeof(ctype));
  ctype* chirp_fft = (ctype*)malloc(dimension_size * sizeof(ctype));
  for (std::size_t i = 0; i < committed_size; i++) {
    double theta = M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    chirp_signal[i] = ctype(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
  }
  std::size_t num_zeros = dimension_size - 2 * committed_size + 1;
  for (std::size_t i = 0; i < committed_size; i++) {
    chirp_signal[committed_size + num_zeros + i - 1] = chirp_signal[committed_size - i];
  }
  host_naive_dft(chirp_signal, chirp_fft, dimension_size);
  std::memcpy(ptr, reinterpret_cast<T*>(&chirp_fft[0]), 2 * dimension_size * sizeof(T));
  free(chirp_signal);
  free(chirp_fft);
}

/**
 * Populates input modifiers required for bluestein
 * @tparam T Scalar Type
 * @param ptr Host Pointer containing the load/store modifiers.
 * @param committed_size original problem size
 * @param dimension_size padded size
 */
template <typename T>
void populate_bluestein_input_modifiers(T* ptr, std::size_t committed_size, std::size_t dimension_size) {
  using ctype = std::complex<T>;
  ctype* scratch = (ctype*)calloc(dimension_size, sizeof(ctype));
  for (std::size_t i = 0; i < committed_size; i++) {
    double theta = -M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    scratch[i] = ctype(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
  }
  std::memcpy(ptr, reinterpret_cast<T*>(&scratch[0]), 2 * dimension_size * sizeof(T));
  free(scratch);
}
}  // namespace detail
}  // namespace portfft

#endif
