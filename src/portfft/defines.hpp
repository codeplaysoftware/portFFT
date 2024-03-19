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

#ifndef PORTFFT_DEFINES_HPP
#define PORTFFT_DEFINES_HPP

#include <array>
#include <cstdint>

#ifdef PORTFFT_KERNEL_LOG
// to avoid extremely long compile times - logging from kernel kills performance anyway
#define PORTFFT_INLINE __attribute__((noinline))
#else
#define PORTFFT_INLINE __attribute__((always_inline))
#endif

#ifndef PORTFFT_N_LOCAL_BANKS
#define PORTFFT_N_LOCAL_BANKS 32
#endif

#ifndef PORTFFT_UNROLL
#define PORTFFT_UNROLL _Pragma("clang loop unroll(full)")
#endif

#define PORTFFT_REQD_SUBGROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]

static_assert((PORTFFT_VEC_LOAD_BYTES & (PORTFFT_VEC_LOAD_BYTES - 1)) == 0,
              "PORTFFT_VEC_LOAD_BYTES should be a power of 2!");

namespace portfft {

using Idx = std::int32_t;
using IdxGlobal = std::int64_t;

/**
 * Struct containing the strides and offsets for a view
 * @tparam Type Type of elements
 * @tparam N Number of elements in each of the two arrays
 */
template <typename T, std::size_t N>
struct stride_offset_struct {
  std::array<T, N> strides;
  std::array<T, N> offsets;
  __attribute__((always_inline)) inline constexpr stride_offset_struct(const std::array<T, N> strides,
                                                                       const std::array<T, N> offsets)
      : strides(strides), offsets(offsets) {}
};

}  // namespace portfft

#endif
