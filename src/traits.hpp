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

#ifndef PORTFFT_TRAITS_HPP
#define PORTFFT_TRAITS_HPP

#include "enums.hpp"

#include <complex>
#include <sycl/sycl.hpp>

namespace portfft {

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
  // NOLINTNEXTLINE
  static constexpr domain value = domain::REAL;
};

template <typename T>
struct get_domain<std::complex<T>> {
  // NOLINTNEXTLINE
  static constexpr domain value = domain::COMPLEX;
};

namespace detail {
template <typename GroupT>
struct get_level;

template <>
struct get_level<sycl::sub_group> {
  static constexpr level value = level::SUBGROUP;
};

template <int Dims>
struct get_level<sycl::group<Dims>> {
  static constexpr level value = level::WORKGROUP;
};

template <typename GroupT>
constexpr detail::level get_level_v = get_level<GroupT>::value;
}  // namespace detail

}  // namespace portfft

#endif
