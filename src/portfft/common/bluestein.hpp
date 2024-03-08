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

#include "portfft/common/host_dft.hpp"
#include "portfft/defines.hpp"

#include <complex>
#include <sycl/sycl.hpp>

namespace portfft {
namespace detail {
/**
 * Utility function to get the dft transform of the chirp signal
 * @tparam T Scalar Type
 * @param ptr Host Pointer containing the load/store modifiers.
 * @param committed_size original problem size
 * @param dimension_size padded size
 */
template <typename T>
void populate_fft_chirp_signal(T* ptr, std::size_t committed_size, std::size_t dimension_size) {
  std::cout << "committed_size = " << committed_size << " padded size = " << dimension_size << std::endl;
  using complex_t = std::complex<T>;
  std::vector<complex_t> chirp_signal(dimension_size, 0);
  std::vector<complex_t> chirp_fft(dimension_size, 0);
  for (std::size_t i = 0; i < committed_size; i++) {
    double theta = M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    chirp_signal[i] = complex_t(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
  }
  std::size_t num_zeros = dimension_size - 2 * committed_size + 1;
  for (std::size_t i = 1; i < committed_size; i++) {
    chirp_signal[committed_size + num_zeros + i - 1] = chirp_signal[committed_size - i];
  }
  host_cooley_tukey(chirp_signal.data(), chirp_fft.data(), dimension_size);
  std::memcpy(ptr, reinterpret_cast<T*>(chirp_fft.data()), 2 * dimension_size * sizeof(T));
}

/**
 * Populates input modifiers required for bluestein
 * @tparam T Scalar Type
 * @param ptr Host Pointer containing the load/store modifiers.
 * @param committed_size committed problem length
 * @param dimension_size padded dft length
 */
template <typename T>
void populate_bluestein_input_modifiers(T* ptr, std::size_t committed_size, std::size_t dimension_size) {
  std::cout << "committed_size = " << committed_size << " padded size = " << dimension_size << std::endl;
  using complex_t = std::complex<T>;
  std::vector<complex_t> scratch(dimension_size, 0);
  for (std::size_t i = 0; i < committed_size; i++) {
    double theta = -M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    scratch[i] = complex_t(static_cast<T>(std::cos(theta)), static_cast<T>(std::sin(theta)));
  }
  std::memcpy(ptr, reinterpret_cast<T*>(scratch.data()), 2 * dimension_size * sizeof(T));
}
}  // namespace detail
}  // namespace portfft

#endif
