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

#include <defines.hpp>
#include <enums.hpp>

/*
To describe the frequency of padding spaces in local memory, we have coined the term "bank line" to describe the chunk
of contiguous memory that exactly fits all of the banks in local memory once. e.g. The NVIDIA Ampere architecture has 32
banks in local memory (shared memory in CUDA terms), each 32 bits. In this case we define a "bank line" as 128 8-bit
bytes.
*/

namespace portfft::detail {

/** A view of memory with built-in offset from zero. eg.
 * this[i] is equivalent to parent[i + offset]
 *
 * @tparam ParentT The underlying view or pointer type.
 * @tparam OffsetT Type for the offset. Defaults to Idx type.
 */
template <typename ParentT, typename OffsetT = Idx>
struct offset_view {
  using element_type = get_element_t<ParentT>;
  using reference = element_type&;
  static constexpr bool IsContiguous = IsContiguousViewV<ParentT>;

  ParentT data;
  OffsetT offset;

  /** Constructor.
   * @param parent The parent view
   * @param offset The offset to add to index look-ups.
   */
  constexpr offset_view(ParentT parent, OffsetT offset) noexcept : data(parent), offset(offset){};

  // Index into the view.
  template <typename IndexerT>
  PORTFFT_INLINE inline constexpr reference operator[](IndexerT i) const {
    return data[static_cast<IndexerT>(offset) + i];
  }
};

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
PORTFFT_INLINE Idx pad_local(Idx local_idx, Idx bank_lines_per_pad) {
  if constexpr (Pad == detail::pad::DO_PAD) {
    local_idx += local_idx / (PORTFFT_N_LOCAL_BANKS * bank_lines_per_pad);
  }
  return local_idx;
}

/** A view of memory with built-in index transformation for padding in local memory.
 *
 * @tparam BankLinesPerPad Index padding parameter. 0 indicates no padding.
 * @tparam ParentT The underlying view or pointer type.
 */
template <Idx BankLinesPerPad, typename ParentT>
struct padded_view {
  using element_type = get_element_t<ParentT>;
  using reference = element_type&;
  static constexpr bool IsContiguous = IsContiguousViewV<ParentT> && !(BankLinesPerPad > 0);

  ParentT data;

  // Constructor: Create a view of a pointer.
  constexpr padded_view(ParentT ptr) noexcept : data(ptr){};

  // Index into the view.
  template <typename IdxT>
  PORTFFT_INLINE inline constexpr reference operator[](IdxT i) const {
    if constexpr (BankLinesPerPad == 0) {
      return data[i];
    } else {
      return data[pad_local<pad::DO_PAD>(static_cast<Idx>(i), BankLinesPerPad)];
    }
  }
};

/**
 * Make a padded view from a pointer.
 *
 * @tparam BankLinesPerPad The padding space to be added after every `bank_lines_per_pad` groups of
 * `PORTFFT_N_LOCAL_BANKS` banks. 0 indicates no padding.
 * @tparam T The element type of the view.
 * @param parent A parent view or pointer to the memory to make a view of
 * @return A memory view
 */
template <std::size_t BankLinesPerPad, typename T>
PORTFFT_INLINE constexpr padded_view<BankLinesPerPad, T> make_padded_view(T parent) noexcept {
  return padded_view<BankLinesPerPad, T>(parent);
}

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_MEMORY_VIEWS_HPP
