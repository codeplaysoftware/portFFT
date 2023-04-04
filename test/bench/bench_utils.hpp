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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

template <typename T>
class memFillKernel;

template <typename T>
void memFill(T* input, sycl::queue& queue, std::size_t num_elements) {
  constexpr int group_dim = 32;
  auto global_range = static_cast<int>(ceil(static_cast<float>(num_elements) / group_dim) * group_dim);
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<memFillKernel<T>>(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(group_dim)),
                                     [=](sycl::nd_item<1> itm) {
                                       auto idx = itm.get_global_id(0);
                                       if (idx < num_elements) {
                                         input[idx] = static_cast<T>(std::complex<float>(idx, num_elements - idx));
                                       }
                                     });
  });
  queue.wait();
}

template <typename type>
bool compare_arrays(type* array1, type* array2, size_t num_elements, double tol) {
  bool correct = true;
  for (size_t i = 0; i < num_elements; i++) {
    correct = correct && (std::abs(array1[i] - array2[i]) <= tol);
  }
  return correct;
}

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(TypeIn* in, TypeOut* out, std::vector<int> length, size_t offset = 0) {
  long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;
  std::vector<std::size_t> dims{1, 1, 1};
  std::copy(length.begin(), length.end(), dims.end());

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
              auto element =
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

#endif