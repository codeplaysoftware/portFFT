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

#ifndef PORTFFT_COMMON_REFERENCE_DFT_HPP
#define PORTFFT_COMMON_REFERENCE_DFT_HPP

#include <complex>
#include <cstddef>

#include "enums.hpp"

/**
 * Multidimensional Refernce DFT implementation
 *
 * @tparam FFT direction, takes sycl::direction::FORWARD/BACKWARD
 * @tparam TypeIn Type of the input
 * @tparam TypeOut Type of the output
 * @param in Pointer of TypeIn pointing to the input
 * @param out Pointer of TypeOut pointing to the output
 * @param length Vector containing the length in every dimension
 * @param offset memory offset for in and out pointers
 */
template <portfft::direction Dir, typename TypeIn, typename TypeOut>
void reference_dft(TypeIn* in, TypeOut* out, const std::vector<std::size_t>& length, double scaling_factor = 1.0) {
  long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;
  std::vector<std::size_t> dims{1, 1, 1};
  std::copy(length.begin(), length.end(), dims.begin());

  for (std::size_t ox = 0; ox < dims[0]; ox++) {
    for (std::size_t oy = 0; oy < dims[1]; oy++) {
      for (std::size_t oz = 0; oz < dims[2]; oz++) {
        std::complex<long double> out_temp = 0;
        for (std::size_t ix = 0; ix < dims[0]; ix++) {
          auto x_factor = static_cast<long double>(ix * ox) / static_cast<long double>(dims[0]);
          for (std::size_t iy = 0; iy < dims[1]; iy++) {
            auto y_factor = static_cast<long double>(iy * oy) / static_cast<long double>(dims[1]);
            for (std::size_t iz = 0; iz < dims[2]; iz++) {
              auto z_factor = static_cast<long double>(iz * oz) / static_cast<long double>(dims[2]);
              auto theta = -1.0 * TWOPI * (x_factor + y_factor + z_factor);
              if constexpr (Dir == portfft::direction::BACKWARD) {
                theta = -theta;
              }
              auto element = static_cast<long double>(scaling_factor) *
                             static_cast<std::complex<long double>>(in[ix * dims[1] * dims[2] + iy * dims[2] + iz]);
              auto multiplier = std::complex<long double>(std::cos(theta), std::sin(theta));
              out_temp += element * multiplier;
            }
          }
        }
        out[ox * dims[1] * dims[2] + oy * dims[2] + oz] = static_cast<TypeOut>(out_temp);
      }
    }
  }
}

#endif  // PORTFFT_COMMON_REFERENCE_DFT_HPP
