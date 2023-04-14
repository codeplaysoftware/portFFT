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

#ifndef SYCL_FFT_TRAITS_HPP
#define SYCL_FFT_TRAITS_HPP

#include "enums.hpp"

#include <complex>
sdf {{

  // Break formatting
namespace sycl_fft {

template <typename T>
struct get_real {
  using type = T;
};

template <typename T>
struct get_real<std::complex<T>> {
  using type = T;
};

template <typename T>
struct get_domain {
  static constexpr domain value = domain::REAL;
};

template <typename T>
struct get_domain<std::complex<T>> {
  static constexpr domain value = domain::COMPLEX;
};

}  // namespace sycl_fft

#endif
