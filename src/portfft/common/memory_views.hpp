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

#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "portfft/traits.hpp"

/*
To describe the frequency of padding spaces in local memory, we have coined the term "bank line" to describe the chunk
of contiguous memory that exactly fits all of the banks in local memory once. e.g. The NVIDIA Ampere architecture has 32
banks in local memory (shared memory in CUDA terms), each 32 bits. In this case we define a "bank line" as 128 8-bit
bytes.
*/

namespace portfft::detail {

/** A view of memory with built-in offset from zero. eg. this[i] is equivalent to parent[i + offset]
 *
 * @tparam ParentT The underlying view or pointer type.
 * @tparam OffsetT Type for the offset. Defaults to Idx type.
 */
template <typename ParentT, typename OffsetT = Idx>
struct offset_view {
  using element_type = get_element_t<ParentT>;
  using reference = element_type&;

  ParentT parent;
  OffsetT offset;

  /** Constructor.
   * @param parent The parent view
   * @param offset The offset to add to index look-ups.
   */
  constexpr offset_view(ParentT parent, OffsetT offset) noexcept : parent(parent), offset(offset){};

  /// Is this view contiguous?
  PORTFFT_INLINE constexpr bool is_contiguous() const noexcept { return is_contiguous_view(parent); }

  // Index into the view.
  PORTFFT_INLINE constexpr reference operator[](OffsetT i) const { return parent[offset + i]; }
};

/**
 * If Pad is true, transforms an index into local memory to skip one element for every PORTFFT_N_LOCAL_BANKS *
 * bank_lines_per_pad elements. Padding in this way avoids bank conflicts when accessing elements with a stride that is
 * multiple of (or has any common divisor greater than 1 with) the number of local banks. Does nothing if Pad is false.
 *
 * Can also be used to transform size of a local allocation to account for padding indices in it this way.
 *
 * @tparam Pad whether to do padding
 * @tparam T input type to the function
 * @param local_idx index to transform
 * @param bank_lines_per_pad A padding space will be added after every `bank_lines_per_pad` groups of
 * `PORTFFT_N_LOCAL_BANKS` banks.
 * @return transformed local_idx
 */
template <detail::pad Pad = detail::pad::DO_PAD, typename T>
PORTFFT_INLINE T pad_local(T local_idx, T bank_lines_per_pad) {
  if constexpr (Pad == detail::pad::DO_PAD) {
    local_idx += local_idx / (PORTFFT_N_LOCAL_BANKS * bank_lines_per_pad);
  }
  return local_idx;
}

/** A view of memory with built-in index transformation for padding in local memory.
 *
 * @tparam ParentT The underlying view or pointer type.
 */
template <typename ParentT>
struct padded_view {
  using element_type = get_element_t<ParentT>;
  using reference = element_type&;

  ParentT parent;
  Idx bank_lines_per_pad;

  // Constructor: Create a view of a pointer or another view.
  constexpr padded_view(ParentT parent, Idx bank_lines_per_pad) noexcept
      : parent(parent), bank_lines_per_pad(bank_lines_per_pad){};

  /// Is this view contiguous?
  PORTFFT_INLINE constexpr bool is_contiguous() const noexcept {
    return is_contiguous_view(parent) && bank_lines_per_pad == 0;
  }

  // Index into the view.
  PORTFFT_INLINE constexpr reference operator[](Idx i) const {
    if (bank_lines_per_pad == 0) {
      return parent[i];
    }
    return parent[pad_local<pad::DO_PAD>(i, bank_lines_per_pad)];
  }
};

/**
 * Multidimensional view.
 *
 * @tparam NDim number of dimensions
 * @tparam TParent type of the underlying view or pointer
 * @tparam TStrides integral type used for strides
 * @tparam TOffset integral type for offset. Needs to be big enough for the raw index into underlying view.
 */
// NDim is std::size_t to match std::array
template <std::size_t NDim, typename TParent, typename TStrides, typename TOffset = Idx>
struct md_view {
  using element_type = get_element_t<TParent>;

  TParent parent;
  std::array<TStrides, NDim> strides;
  TOffset offset;
  /**
   * Constructor
   *
   * @param parent underlying view or pointer
   * @param strides strides for each of the dimensions
   * @param offset offset
   */
  constexpr md_view(TParent parent, const std::array<TStrides, NDim>& strides, TOffset offset = 0) noexcept
      : parent(parent), strides(strides), offset(offset) {}

  /**
   * Return a view into remaining dimensions after indexing into the first one.
   *
   * @param index index into the first dimension
   * @return view into remaining dimensions
   */
  template <typename T = int, std::enable_if_t<NDim >= 1 && std::is_same_v<T, T>>* = nullptr>
  PORTFFT_INLINE constexpr md_view<NDim - 1, TParent, TStrides, TOffset> inner(TStrides index) noexcept {
    std::array<TStrides, NDim - 1> next_strides;
    PORTFFT_UNROLL
    for (std::size_t j = 0; j < NDim - 1; j++) {
      next_strides[j] = strides[j + 1];
    }
    return {parent, next_strides, offset + static_cast<TOffset>(index) * strides[0]};
  }

  /**
   * Only available on 0-dimensional view. Gets the element the view points to.
   *
   * @return a reference to the element
   */
  template <typename T = int, std::enable_if_t<NDim == 0 && std::is_same_v<T, T>>* = nullptr>
  PORTFFT_INLINE constexpr auto& get() const {
    return parent[offset];
  }
};

/**
 * View with multidimensional strides and offsets
 * @tparam TParent type of the underlying view or pointer
 * @tparam TIdx integral type used strides and offsets
 * @tparam NDim number of dimensions
 *
 */
// NDim is std::size_t to match std::array
template <typename TParent, typename TIdx, std::size_t NDim = 1>
struct strided_view {
  using element_type = get_element_t<TParent>;
  using reference = element_type&;
  TParent parent;
  std::array<TIdx, NDim> sizes;
  std::array<TIdx, NDim> offsets;

  /**
   * Constructor.
   *
   * @param parent underlying view or pointer
   * @param sizes sizes for each of the dimensions
   * @param offsets offsets into each of the dimensions
   */
  constexpr strided_view(TParent parent, const std::array<TIdx, NDim>& sizes,
                         const std::array<TIdx, NDim>& offsets) noexcept
      : parent(parent), sizes(sizes), offsets(offsets) {}

  /**
   * Constructor for 1-dimensional stride and offset.
   *
   * @param parent underlying view or pointer
   * @param sizes size
   * @param offsets offset
   */
  constexpr strided_view(TParent parent, const TIdx size, const TIdx offset = 0) noexcept
      : parent(parent), sizes{size}, offsets{offset} {}

  /**
   * Calculates raw index (index into underlying pointer or view) from an index into this strided view.
   * 
   * @param index 
   * @return PORTFFT_INLINE constexpr 
   */
  PORTFFT_INLINE constexpr TIdx raw_index(Idx index) const {
    TIdx index_calculated = static_cast<TIdx>(index);
    PORTFFT_UNROLL
    for (std::size_t i = 0; i < NDim; i++) {
      index_calculated = index_calculated * sizes[i] + offsets[i];
    }
    return index_calculated;
  }

  /**
   * Index into the view.
   *
   * @param index index
   * @return reference to the indexed element
   */
  PORTFFT_INLINE constexpr reference operator[](Idx index) const {
    return parent[raw_index(index)];
  }
};

/**
 * Get the raw pointer object. No-op for pointers
 *
 * @tparam T type pointed to
 * @param arg pointer or view to get the raw pointer from
 * @return raw pointer
 */
template <typename T>
PORTFFT_INLINE constexpr T* get_raw_pointer(T* arg) {
  return arg;
}

/**
 * Get the raw pointer object from a view.
 *
 * @tparam TView type of the view
 * @param arg pointer or view to get the raw pointer from
 * @return raw pointer
 */
template <typename TView>
PORTFFT_INLINE constexpr get_element_t<TView>* get_raw_pointer(TView arg) {
  return get_raw_pointer(arg.parent);
}

template <typename T, typename TOffset>
PORTFFT_INLINE constexpr T* get_nonstrided_view(T* arg, TOffset offset) {
  return arg + offset;
}

template <typename TParent, typename TIdx, std::size_t NDim, typename TOffset>
PORTFFT_INLINE constexpr auto get_nonstrided_view(strided_view<TParent,TIdx,NDim> arg, TOffset offset) {
  //return offset_view<TParent,TIdx>(arg.parent, arg.raw_index(offset));
  return get_nonstrided_view(arg.parent, arg.raw_index(offset));
}

template <typename TParent, typename TIdx, typename TOffset>
PORTFFT_INLINE constexpr auto get_nonstrided_view(offset_view<TParent,TIdx> arg, TOffset offset) {
  return get_nonstrided_view(arg.parent, offset + arg.offset);
}

template <typename TParent, typename TOffset>
PORTFFT_INLINE constexpr offset_view<padded_view<TParent>,TOffset> get_nonstrided_view(padded_view<TParent> arg, TOffset offset) {
  static_assert(std::is_pointer_v<TParent>, "Getting nonstrided view from a padded_view is only possible if the parent is a raw pointer!");
  return {arg, offset};
}

/**
 * Implementation of `is_view_multidimensional`.
 * 
 * @tparam T type of the view
 */
template<typename T>
struct is_view_multidimensional_impl{
  /**
   * Check if a view is multidimensional.
   * 
   * @return true if the view is multidimensional, false otherwise
   */
  static constexpr bool get(){
    return false;
  }
};
template<std::size_t NDim, typename TParent, typename TStrides, typename TOffset>
struct is_view_multidimensional_impl<md_view<NDim, TParent, TStrides, TOffset>>{
  /**
   * Check if a view is multidimensional.
   * 
   * @return true if the view is multidimensional, false otherwise
   */
  static constexpr bool get(){
    return true;
  }
};

/**
 * Check if a view is multidimensional.
 * 
 * @tparam T type of the view
 * @return true if the view is multidimensional, false otherwise
 */
template<typename T>
constexpr bool is_view_multidimensional(){
  return is_view_multidimensional_impl<T>::get();
}

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_MEMORY_VIEWS_HPP
