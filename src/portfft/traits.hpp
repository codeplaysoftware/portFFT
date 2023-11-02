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

#include <complex>

#include "defines.hpp"
#include "enums.hpp"

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

/** Get the element type of type T
 *  Examples:
 *  * type is T for a pointer T*
 *  * type is T for a padded_view<3, T*>
 *
 *  @tparam T The type to get the element of
 **/
template <typename T>
struct get_element {
  using type = typename T::element_type;
};

/// Specialization of get_elem for pointer
template <typename T>
struct get_element<T*> {
  using type = T;
};

/// get_element::type shortcut
template <typename T>
using get_element_t = typename get_element<T>::type;

/// get_element::type with any topmost const and/or volatile qualifiers removed.
template <typename T>
using get_element_remove_cv_t = std::remove_cv_t<get_element_t<T>>;

/** Test if a view of memory is contiguous
 *  Examples:
 *  * true for a pointer
 *  * false for a padded view where bank_lines_per_pad > 0
 *
 *  @tparam T The type to test for contiguousness
 *  @param view The view type
 **/
template <typename T>
PORTFFT_INLINE constexpr bool is_contiguous_view(T&& view) noexcept {
  return view.is_contiguous();
}

/// Specialization of is_contiguous_view for pointer
template <typename T>
PORTFFT_INLINE constexpr bool is_contiguous_view(T*) noexcept {
  return true;
}

}  // namespace detail

}  // namespace portfft

#endif
