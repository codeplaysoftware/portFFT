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
#include <cstdint>

namespace portfft::detail {

/** A list of supported FFT sizes.
 * @tparam Sizes The supported FFT sizes.
 */
template <std::size_t... Sizes>
struct size_list {
  // Specialization only used for zero size Sizes parameter pack.
  static constexpr bool list_end = true;
};

template <std::size_t Size, std::size_t... OtherSizes>
struct size_list<Size, OtherSizes...> {
  using child_t = size_list<OtherSizes...>;
  static constexpr bool list_end = false;
  static constexpr std::size_t size = Size;

  /**
   *  Returns true if a size list contains a particular value.
   *  @param x The value to look for
   **/
  static constexpr bool has_size(std::size_t x) {
    bool hasSize = x == size;
    if constexpr (!child_t::list_end) {
      hasSize |= child_t::has_size(x);
    }
    return hasSize;
  }
};

using cooley_tukey_size_list_t = size_list<PORTFFT_COOLEY_TUKEY_OPTIMIZED_SIZES>;

}  // namespace portfft::detail

#endif  // PORTFFT_COOLEY_TUKEY_COMPILED_SIZES_HPP
