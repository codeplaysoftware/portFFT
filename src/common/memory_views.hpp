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

#ifndef PORTFFT_COMMON_MEMORY_VIEWS_HPP
#define PORTFFT_COMMON_MEMORY_VIEWS_HPP

#include <cstdint>

#include <enums.hpp>

#ifndef PORTFFT_N_LOCAL_BANKS
#define PORTFFT_N_LOCAL_BANKS 32
#endif

namespace portfft::detail {

/**
 * If Pad is true transforms an index into local memory to skip one element for every
 * PORTFFT_N_LOCAL_BANKS elements. Padding in this way avoids bank conflicts when accessing
 * elements with a stride that is multiple of (or has any common divisor greater than 1 with)
 * the number of local banks. Does nothing if Pad is false.
 *
 * Can also be used to transform size of a local allocation to account for padding indices in it this way.
 *
 * @tparam Pad whether to do padding
 * @param local_idx index to transform
 * @param bank_lines_per_pad A padding space will be added after every `bank_lines_per_pad` groups of
 * `PORTFFT_N_LOCAL_BANKS` banks.
 * @return transformed local_idx
 */
template <detail::pad Pad = detail::pad::DO_PAD>
__attribute__((always_inline)) inline std::size_t pad_local(std::size_t local_idx, std::size_t bank_lines_per_pad) {
  if constexpr (Pad == detail::pad::DO_PAD) {
    local_idx += local_idx / (PORTFFT_N_LOCAL_BANKS * bank_lines_per_pad);
  }
  return local_idx;
}

template <pad Pad, std::size_t BankLinesPerPad, typename T>
struct padded_view {
  using element_type = T;
  using reference = T&;
  static constexpr bool is_padded = Pad == pad::DO_PAD;
  T* data;

  // Constructor: Create a view of a pointer.
  constexpr padded_view(T* ptr) noexcept : data(ptr){};

  // Index into the view.
  template <typename IdxT>
  __attribute__((always_inline)) inline constexpr reference operator[](IdxT i) const {
    return data[pad_local<Pad>(i, BankLinesPerPad)];
  }
};

/**
 * Make a padded view from a pointer.
 *
 * @tparam Pad whether to do padding
 * @tparam BankLinesPerPad The padding space to be added after every `bank_lines_per_pad` groups of
 * `PORTFFT_N_LOCAL_BANKS` banks.
 * @tparam T The element type of the view.
 * @param ptr A pointer to the memory to make a view of
 * @return A memory view
 */
template <pad Pad, std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) constexpr padded_view<Pad, BankLinesPerPad, T> make_padded_view(T* ptr) noexcept {
  static_assert(std::is_pointer_v<T*>, "Expected pointer argument.");
  return padded_view<Pad, BankLinesPerPad, T>(ptr);
}

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_MEMORY_VIEWS_HPP
