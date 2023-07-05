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

#ifndef SYCL_FFT_COMMON_NUMBER_GENERATORS_HPP
#define SYCL_FFT_COMMON_NUMBER_GENERATORS_HPP

#include <complex>
#include <random>
#include <type_traits>

using engine = std::ranlux48_base;

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>> populate_with_random(std::vector<T>& in,
                                                                      T lowerLimit = T(-1.0),
                                                                      T higherLimit = T(1.0)) {
  engine algo(0);
  std::uniform_real_distribution<T> distribution(lowerLimit, higherLimit);

  for (auto& val : in) {
    val = distribution(algo);
  }
}

template <typename T>
void populate_with_random(std::vector<std::complex<T>>& in, T lowerLimit = T(-1.0),
                          T higherLimit = T(1.0)) {
  engine algo(0);
  std::uniform_real_distribution<T> distribution(lowerLimit, higherLimit);

  for (auto& val : in) {
    val = std::complex<T>(distribution(algo), distribution(algo));
  }
}

#endif  // SYCL_FFT_COMMON_NUMBER_GENERATORS_HPP
