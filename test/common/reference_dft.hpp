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

#ifndef SYCL_FFT_COMMON_REFERENCE_DFT_HPP
#define SYCL_FFT_COMMON_REFERENCE_DFT_HPP

#include <cstddef>
#include <complex>
#include "enums.hpp"

/**
 * @brief Multidimensional Refernce DFT implementation
 * 
 * @tparam FFT direction, takes sycl::direction::FORWARD/BACKWARD
 * @tparam TypeIn Type of the input
 * @tparam TypeOut Type of the output
 * @param in Pointer of TypeIn pointing to the input
 * @param out Pointer of TypeOut pointing to the output
 * @param length Vector containing the length in every dimension
 * @param offset memory offset for in and out pointers
 */
template <sycl_fft::direction dir, typename TypeIn, typename TypeOut>
void reference_dft(TypeIn* in, TypeOut* out, std::vector<int> length, size_t offset = 0, double scaling_factor = 1.0) {
  long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;
  std::vector<std::size_t> dims{1, 1, 1};
  std::copy(length.begin(), length.end(), dims.begin());

  for (size_t ox = 0; ox < dims[0]; ox++) {
    for (size_t oy = 0; oy < dims[1]; oy++) {
      for (size_t oz = 0; oz < dims[2]; oz++) {
        std::complex<long double> out_temp = 0;
        for (size_t ix = 0; ix < dims[0]; ix++) {
          for (size_t iy = 0; iy < dims[1]; iy++) {
            for (size_t iz = 0; iz < dims[2]; iz++) {
              double theta = -1 * TWOPI *
                             ((ix * ox / static_cast<double>(dims[0])) + (iy * oy / static_cast<double>(dims[1])) +
                              (iz * oz / static_cast<double>(dims[2])));
              if constexpr(dir == sycl_fft::direction::BACKWARD){
                theta = -theta;
              }
              auto element = static_cast<long double>(scaling_factor) * 
                  static_cast<std::complex<long double>>(in[offset + ix * dims[1] * dims[2] + iy * dims[2] + iz]);
              auto multiplier = std::complex<long double>(std::cos(theta), std::sin(theta));
              out_temp += element * multiplier;
            }
          }
        }
        out[offset + ox * dims[1] * dims[2] + oy * dims[2] + oz] = out_temp;
      }
    }
  }
}

#endif //SYCL_FFT_COMMON_REFERENCE_DFT_HPP
