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
 * @param ptr Device Pointer containing the load/store modifiers.
 * @param committed_size original problem size
 * @param dimension_size padded size
 * @param queue queue with the committed descriptor
 */
template <typename T>
void get_fft_chirp_signal(T* ptr, IdxGlobal committed_size, IdxGlobal dimension_size, sycl::queue& queue) {
  using ctype = std::complex<T>;
  ctype* chirp_signal = (ctype*)calloc(dimension_size, sizeof(ctype));
  ctype* chirp_fft = (ctype*)malloc(dimension_size * sizeof(ctype));
  for (IdxGlobal i = 0; i < committed_size; i++) {
    double theta = M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    chirp_signal[i] = ctype(std::cos(theta), std::sin(theta));
  }
  IdxGlobal num_zeros = dimension_size - 2 * committed_size + 1;
  for (IdxGlobal i = 0; i < committed_size; i++) {
    chirp_signal[committed_size + num_zeros + i - 1] = chirp_signal[committed_size - i];
  }
  naive_dft(chirp_signal, chirp_fft, dimension_size);
  queue.copy(reinterpret_cast<T*>(&chirp_fft[0]), ptr, 2 * dimension_size).wait();
}

template <typename T>
void populate_input_and_output_modifiers(T* ptr, IdxGlobal committed_size, IdxGlobal dimension_size,
                                         sycl::queue& queue) {
  using ctype = std::complex<T>;
  ctype* scratch = (ctype*)calloc(dimension_size, sizeof(ctype));
  for (IdxGlobal i = 0; i < committed_size; i++) {
    double theta = -M_PI * static_cast<double>(i * i) / static_cast<double>(committed_size);
    scratch[i] = ctype(std::cos(theta), std::sin(theta));
  }
  queue.copy(reinterpret_cast<T*>(&scratch[0]), ptr, 2 * dimension_size);
}
}  // namespace detail
}  // namespace portfft

#endif