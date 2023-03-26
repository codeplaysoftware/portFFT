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

#ifndef SYCLFFT_BENCH_UTILS_HPP
#define SYCLFFT_BENCH_UTILS_HPP

#include <iostream>
#include <vector>

template <typename type>
bool compare_arrays(std::vector<type> array1, std::vector<type> array2,
                    double tol) {
  (array1.size(), array2.size());
  bool correct = 1;
  for (size_t i = 0; i < array1.size(); i++) {
    correct = correct & (std::abs(array1[i].real() - array2[i].real()) <= tol);
  }
  return correct;
}

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(std::vector<TypeIn>& in, std::vector<TypeOut>& out,
                           const std::vector<std::size_t> length, size_t offset = 0) {
  long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;
  std::vector<std::size_t> dims{1, 1, 1};
  dims.insert(dims.begin(), length.begin(), length.end());

  for (size_t ox = 0; ox < dims[0]; ox++) {
      for(size_t oy = 0; oy <  dims[1]; oy++){
        for(size_t oz = 0; oz < dims[2]; oz++){
          std::complex<long double> out_temp = 0;
          for(size_t ix = 0; ix < dims[0]; ix++){
            for(size_t iy = 0; iy < dims[1]; iy++){
              for(size_t iz = 0; iz < dims[2]; iz++){
                double theta = -1 * TWOPI * ((ix * ox / dims[0]) + (iy * oy / dims[1]) + (iz * oz / dims[2]));
                auto element = static_cast<std::complex<long double>>(in[offset + ix * dims[1] * dims[2] + iy * dims[2] + iz]);
                auto multiplier = std::complex<long double>(std::cos(theta), std::sin(theta));
                out_temp += element * multiplier;
              }
            }
          }
          out[offset + ox * dims[1] * dims[2] + oy * dims[2] + oz] = static_cast<TypeOut>(out_temp);
        }
      }
  }
}

#endif