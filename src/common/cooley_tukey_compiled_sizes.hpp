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

#ifndef PORTFFT_COOLEY_TUKEY_COMPILED_SIZES_HPP
#define PORTFFT_COOLEY_TUKEY_COMPILED_SIZES_HPP
#include <defines.hpp>
#include <cstdint>

namespace portfft::detail {

/** A list of supported FFT sizes.
 * @tparam Sizes The supported FFT sizes.
 */
template <IdxGlobal... Sizes>
struct size_list {
  // Specialization only used for zero size Sizes parameter pack.
  static constexpr bool ListEnd = true;
};

template <IdxGlobal ThisSize, IdxGlobal... OtherSizes>
struct size_list<ThisSize, OtherSizes...> {
  using child_t = size_list<OtherSizes...>;
  static constexpr bool ListEnd = false;
  static constexpr IdxGlobal Size = ThisSize;

  /**
   *  Returns true if a size list contains a particular value.
   *  @param x The value to look for
   **/
  static constexpr bool has_size(IdxGlobal x) {
    bool has_size = x == Size;
    if constexpr (!child_t::ListEnd) {
      has_size |= child_t::has_size(x);
    }
    return has_size;
  }
};

using cooley_tukey_size_list_t = size_list<PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES>;

}  // namespace portfft::detail

#endif  // PORTFFT_COOLEY_TUKEY_COMPILED_SIZES_HPP
