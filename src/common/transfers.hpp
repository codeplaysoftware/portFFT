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

#ifndef SYCL_FFT_COMMON_TRANSFERS_HPP
#define SYCL_FFT_COMMON_TRANSFERS_HPP

#include <common/helpers.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

#ifndef SYCL_FFT_N_LOCAL_BANKS
#define SYCL_FFT_N_LOCAL_BANKS 32
#endif

static_assert((SYCLFFT_TARGET_WI_LOAD & (SYCLFFT_TARGET_WI_LOAD - 1)) == 0,
              "SYCLFFT_TARGET_WI_LOAD should be a power of 2!");

namespace sycl_fft {

namespace detail {

/**
 * If Pad is true transforms an index into local memory to skip one element for every
 * SYCL_FFT_N_LOCAL_BANKS elements. Padding in this way avoids bank conflicts when accessing
 * elements with a stride that is multiple of (or has any common divisor greater than 1 with)
 * the number of local banks. Does nothing if Pad is false.
 *
 * Can also be used to transform size of a local allocation to account for padding indices in it this way.
 *
 * @tparam Pad whether to do padding
 * @tparam T_index Index type
 * @param local_idx index to transform
 * @return transformed local_idx
 */
template <detail::pad Pad = detail::pad::DO_PAD, typename T_index>
__attribute__((always_inline)) inline T_index pad_local(T_index local_idx) {
  if constexpr (Pad == detail::pad::DO_PAD) {
    local_idx += local_idx / SYCL_FFT_N_LOCAL_BANKS;
  }
  return local_idx;
}

}  // namespace detail

/**
 * Copies data from global memory to local memory.
 *
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T type of the scalar used for computations
 * @tparam T_index Index type
 * @param it nd_item
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <detail::pad Pad, detail::level Level, typename T, typename T_index>
__attribute__((always_inline)) inline void global2local(sycl::nd_item<1> it, const T* global, T* local,
                                                        T_index total_num_elems, T_index global_offset = 0,
                                                        T_index local_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int chunk_size_raw = SYCLFFT_TARGET_WI_LOAD / sizeof(T);
  constexpr int chunk_size = chunk_size_raw < 1 ? 1 : chunk_size_raw;
  using T_vec = sycl::vec<T, chunk_size>;

  sycl::sub_group sg = it.get_sub_group();
  T_index local_size;
  T_index local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<T_index>(sg.get_local_linear_id());
    local_size = static_cast<T_index>(SYCLFFT_TARGET_SUBGROUP_SIZE);
  } else {
    local_id = static_cast<T_index>(it.get_local_id(0));
    local_size = static_cast<T_index>(SYCLFFT_TARGET_SUBGROUP_SIZE * SYCLFFT_SGS_IN_WG);
  }

  T_index stride = local_size * static_cast<T_index>(chunk_size);
  T_index rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef SYCLFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    T_index subgroup_id = static_cast<T_index>(sg.get_group_id());
    T_index elems_per_sg = detail::divideCeil<T_index>(total_num_elems, SYCLFFT_SGS_IN_WG);
    T_index offset = subgroup_id * elems_per_sg;
    T_index next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = static_cast<T_index>(sg.get_local_linear_id());
    local_size = static_cast<T_index>(SYCLFFT_TARGET_SUBGROUP_SIZE);
    stride = local_size * static_cast<T_index>(chunk_size);
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup loads a chunk of `chunk_size * local_size` elements.
  for (T_index i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec loaded = sg.load<chunk_size>(detail::get_global_multi_ptr(&global[global_offset + i]));
    if constexpr (SYCL_FFT_N_LOCAL_BANKS % SYCLFFT_TARGET_SUBGROUP_SIZE == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
        T_index local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<T_index>(j) * local_size);
        sg.store(detail::get_local_multi_ptr(&local[local_idx]), loaded[j]);
      });
    } else {
      detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
        T_index local_idx = detail::pad_local<Pad>(local_offset + i + j * local_size + local_id);
        local[local_idx] = loaded[j];
      });
    }
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::roundUpToMultiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  // global_aligned_ptr >= global_ptr is guaranteed
  T_index unaligned_elements = static_cast<T_index>(global_aligned_ptr - global_ptr);

  // load the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    T_index local_idx = detail::pad_local<Pad>(local_offset + local_id);
    local[local_idx] = global[global_offset + local_id];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem loads a chunk of `chunk_size` consecutive elements. Chunks loaded by a group are consecutive.
  for (T_index i = local_id * static_cast<T_index>(chunk_size); i < rounded_down_num_elems; i += stride) {
    T_vec loaded;
    loaded.load(0, detail::get_global_multi_ptr(&global[global_offset + i]));
    detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
      T_index local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<T_index>(j));
      local[local_idx] = loaded[j];
    });
  }
#endif
  // We can not load `chunk_size`-sized chunks anymore, so we load the largest we can - `last_chunk_size`-sized one
  // rounded_down_num_elems >= total_num_elems is guaranteed
  T_index last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (T_index j = 0; j < last_chunk_size; j++) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j);
    local[local_idx] = global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j];
  }
  // Less than group size elements remain. Each workitem loads at most one.
  T_index my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + my_last_idx);
    local[local_idx] = global[global_offset + my_last_idx];
  }
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T type of the scalar used for computations
 * @tparam T_index Index type
 * @param it nd_item
 * @param local pointer to local memory
 * @param global pointer to global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <detail::pad Pad, detail::level Level, typename T, typename T_index>
__attribute__((always_inline)) inline void local2global(sycl::nd_item<1> it, const T* local, T* global,
                                                        T_index total_num_elems, T_index local_offset = 0,
                                                        T_index global_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int chunk_size_raw = SYCLFFT_TARGET_WI_LOAD / sizeof(T);
  constexpr int chunk_size = chunk_size_raw < 1 ? 1 : chunk_size_raw;
  using T_vec = sycl::vec<T, chunk_size>;

  sycl::sub_group sg = it.get_sub_group();
  T_index local_size;
  T_index local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<T_index>(sg.get_local_linear_id());
    local_size = static_cast<T_index>(SYCLFFT_TARGET_SUBGROUP_SIZE);
  } else {
    local_id = static_cast<T_index>(it.get_local_id(0));
    local_size = static_cast<T_index>(SYCLFFT_TARGET_SUBGROUP_SIZE * SYCLFFT_SGS_IN_WG);
  }

  T_index stride = local_size * static_cast<T_index>(chunk_size);
  T_index rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef SYCLFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    T_index subgroup_id = static_cast<T_index>(sg.get_group_id());
    T_index elems_per_sg = detail::divideCeil<T_index>(total_num_elems, SYCLFFT_SGS_IN_WG);
    T_index offset = subgroup_id * elems_per_sg;
    T_index next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = sg.get_local_linear_id();
    local_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
    stride = local_size * static_cast<T_index>(chunk_size);
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup stores a chunk of `chunk_size * local_size` elements.
  for (T_index i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    if constexpr (SYCL_FFT_N_LOCAL_BANKS % SYCLFFT_TARGET_SUBGROUP_SIZE == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
        T_index local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<T_index>(j) * local_size);
        to_store[j] = sg.load(detail::get_local_multi_ptr(&local[local_idx]));
      });
    } else {
      detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
        T_index local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<T_index>(j) * local_size + local_id);
        to_store[j] = local[local_idx];
      });
    }
    sg.store(detail::get_global_multi_ptr(&global[global_offset + i]), to_store);
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::roundUpToMultiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  // global_aligned_ptr >= global_ptr is guaranteed
  T_index unaligned_elements = static_cast<T_index>(global_aligned_ptr - global_ptr);

  // store the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    T_index local_idx = detail::pad_local<Pad>(local_offset + local_id);
    global[global_offset + local_id] = local[local_idx];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem stores a chunk of `chunk_size` consecutive elements. Chunks stored by a group are consecutive.
  for (T_index i = local_id * static_cast<T_index>(chunk_size); i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    detail::unrolled_loop<0, chunk_size, 1>([&](int j) __attribute__((always_inline)) {
      T_index local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<T_index>(j));
      to_store[j] = local[local_idx];
    });
    to_store.store(0, detail::get_global_multi_ptr(&global[global_offset + i]));
  }
#endif
  // We can not store `chunk_size`-sized chunks anymore, so we store the largest we can - `last_chunk_size`-sized one
  // rounded_down_num_elems >= total_num_elems is guaranteed
  T_index last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (T_index j = 0; j < last_chunk_size; j++) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j);
    global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j] = local[local_idx];
  }
  // Less than group size elements remain. Each workitem stores at most one.
  T_index my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + my_last_idx);
    global[global_offset + my_last_idx] = local[local_idx];
  }
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam T type of the scalar used for computations
 * @tparam T_index Index type
 * @param local pointer to local memory
 * @param priv pointer to private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= num_elems_per_wi
 * @param local_offset offset to the local pointer
 */
template <auto num_elems_per_wi, detail::pad Pad, typename T, typename T_index>
__attribute__((always_inline)) inline std::enable_if_t<std::is_same<decltype(num_elems_per_wi), T_index>::value>
local2private(const T* local, T* priv, T_index local_id, T_index stride, T_index local_offset = 0) {
  detail::unrolled_loop<0, num_elems_per_wi, 1, T_index>([&](T_index i) __attribute__((always_inline)) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i);
    priv[i] = local[local_idx];
  });
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam T type of the scalar used for computations
 * @tparam T_index Index type
 * @param priv pointer to private memory
 * @param local pointer to local memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= num_elems_per_wi
 * @param local_offset offset to the local pointer
 */
template <auto num_elems_per_wi, detail::pad Pad, typename T, typename T_index>
__attribute__((always_inline)) inline std::enable_if_t<std::is_same<decltype(num_elems_per_wi), T_index>::value>
private2local(const T* priv, T* local, T_index local_id, T_index stride, T_index local_offset = 0) {
  detail::unrolled_loop<0, num_elems_per_wi, 1, T_index>([&](T_index i) __attribute__((always_inline)) {
    T_index local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i);
    local[local_idx] = priv[i];
  });
}

/**
 * Copies data from private memory to local or global memory. Consecutive workitems write
 * consecutive elements. The copy is done jointly by a group of threads defined by `local_id` and `workers_in_group`.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam T type of the scalar used for computations
 * @tparam T_index Index type
 * @param priv pointer to private memory
 * @param destination pointer to destination - local or global memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param destination_offset offset to the destination pointer
 */
template <auto num_elems_per_wi, detail::pad Pad, typename T, typename T_index>
__attribute__((always_inline)) inline std::enable_if_t<std::is_same<decltype(num_elems_per_wi), T_index>::value>
store_transposed(const T* priv, T* destination, T_index local_id, T_index workers_in_group,
                 T_index destination_offset = 0) {
  constexpr int vec_size = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<T, vec_size>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv);
  T_vec* local_vec = reinterpret_cast<T_vec*>(&destination[0]);

  detail::unrolled_loop<0, num_elems_per_wi, 2, T_index>([&](T_index i) __attribute__((always_inline)) {
    T_index destination_idx = detail::pad_local<Pad>(destination_offset + local_id * 2 + i * workers_in_group);
    if (destination_idx % 2 == 0) {  // if the destination address is aligned, we can use vector store
      local_vec[destination_idx / 2] = priv_vec[i / 2];
    } else {
      destination[destination_idx] = priv[i];
      destination[destination_idx + 1] = priv[i + 1];
    }
  });
}
};  // namespace sycl_fft

#endif
