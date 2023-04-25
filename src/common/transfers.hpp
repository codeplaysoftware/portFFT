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
#include <sycl/sycl.hpp>

#ifndef SYCL_FFT_N_LOCAL_BANKS
#define SYCL_FFT_N_LOCAL_BANKS 32
#endif

namespace sycl_fft {

namespace detail {

/**
 * If Pad is true ransforms an index into local memory to skip one element for every
 * SYCL_FFT_N_LOCAL_BANKS elements. Padding in this way avoids bank conflicts when accessing
 * elements with a stride that is multiple of (or has any common divisor greater than 1 with)
 * the number of local banks. Does nothing if Pad is false.
 *
 * Can also be used to transform size of a local allocation to account for padding indices in it this way.
 *
 * @tparam Pad whether to do padding
 * @param local_idx index to transform
 * @return transformed local_idx
 */
template <bool Pad = true>
inline std::size_t pad_local(std::size_t local_idx) {
  if constexpr (Pad) {
    local_idx += local_idx / SYCL_FFT_N_LOCAL_BANKS;
  }
  return local_idx;
}

}  // namespace detail

/**
 * Copies data from global memory to local memory. Depending on how parameters
 * are set, this can work on work group or subgroup level.
 *
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam T_glob_ptr type of pointer to global memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param local_size local size of group that is doing the copying
 * @param local_id local id of work item withing the group that is doing the
 * copying
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <bool Pad, typename T_glob_ptr, typename T_loc_ptr>
inline void global2local(T_glob_ptr global, T_loc_ptr local, std::size_t total_num_elems, std::size_t local_size,
                         std::size_t local_id, std::size_t global_offset = 0, std::size_t local_offset = 0) {
  using T = detail::remove_ptr<T_loc_ptr>;
  constexpr int chunk_size_raw = SYCLFFT_TARGET_WG_LOAD / sizeof(T);
  constexpr int chunk_size = chunk_size_raw < 1 ? 1 : chunk_size_raw;
  using T_vec = sycl::vec<T, chunk_size>;
  int stride = local_size * chunk_size;
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

<<<<<<< Updated upstream
  void* global_void = static_cast<void*>(const_cast<T*>(&global[global_offset]));
  std::size_t space = total_num_elems * sizeof(T);  // on input total space, on output aligned space
  std::align(alignof(T_vec), 1, global_void, space);

  // load the first few unaligned elements
  std::size_t unaligned_elements = (total_num_elems * sizeof(T) - space) / sizeof(T);
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
=======
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(detail::roundUpToMultiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  std::size_t unaligned_elements = global_aligned_ptr - global_ptr;

  // load the first few unaligned elements
  if (local_id < unaligned_elements) { // assuming unaligned_elements <= local_size
>>>>>>> Stashed changes
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id);
    local[local_idx] = global[global_offset + local_id];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem loads a chunk of `chunk_size` consecutive elements. Chunks loaded by a group are consecutive.
  for (std::size_t i = local_id * chunk_size; i < rounded_down_num_elems; i += stride) {
    T_vec loaded;
    loaded.load(0, sycl::make_ptr<const T, sycl::access::address_space::global_space>(&global[global_offset + i]));
    for (int j = 0; j < chunk_size; j++) {
      std::size_t local_idx = detail::pad_local<Pad>(local_offset + i + j);
      local[local_idx] = loaded[j];
    }
  }
  // We can not load `chunk_size`-sized chunks anymore, so we load the largest we can - `last_chunk_size`-sized one
  int last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (int j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j);
    local[local_idx] = global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j];
  }
  // Less than group size elements remain. Each workitem loads at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + my_last_idx);
    local[local_idx] = global[global_offset + my_last_idx];
  }
}

/**
 * Copies data from local memory to global memory. Depending of how parameters
 * are set, this can work on work group or subgroup level.
 *
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_glob_ptr type of pointer to global memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param local pointer to local memory
 * @param global pointer to global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_size local size of group that is doing the copying
 * @param local_id local id of work item withing the group that is doing the
 * copying
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <bool Pad, typename T_loc_ptr, typename T_glob_ptr>
inline void local2global(T_loc_ptr local, T_glob_ptr global, std::size_t total_num_elems, std::size_t local_size,
                         std::size_t local_id, std::size_t local_offset = 0, std::size_t global_offset = 0) {
  using T = detail::remove_ptr<T_loc_ptr>;
  constexpr int chunk_size_raw = SYCLFFT_TARGET_WG_LOAD / sizeof(T);
  constexpr int chunk_size = chunk_size_raw < 1 ? 1 : chunk_size_raw;
  using T_vec = sycl::vec<T, chunk_size>;
  int stride = local_size * chunk_size;
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

<<<<<<< Updated upstream
  void* global_void = static_cast<void*>(&global[global_offset]);
  std::size_t space = total_num_elems * sizeof(T);  // on input total space, on output aligned space
  std::align(alignof(T_vec), 1, global_void, space);

  // store the first few unaligned elements
  std::size_t unaligned_elements = (total_num_elems * sizeof(T) - space) / sizeof(T);
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
=======
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(detail::roundUpToMultiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  std::size_t unaligned_elements = global_aligned_ptr - global_ptr;

  // store the first few unaligned elements
  if (local_id < unaligned_elements) { // assuming unaligned_elements <= local_size
>>>>>>> Stashed changes
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id);
    global[global_offset + local_id] = local[local_idx];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem stores a chunk of `chunk_size` consecutive elements. Chunks stored by a group are consecutive.
  for (std::size_t i = local_id * chunk_size; i < rounded_down_num_elems; i += stride) {
    T_vec* global_vec = reinterpret_cast<T_vec*>(&global[global_offset + i]);
    T_vec to_store;
    for (int j = 0; j < chunk_size; j++) {
      std::size_t local_idx = detail::pad_local<Pad>(local_offset + i + j);
      to_store[j] = local[local_idx];
    }
    to_store.store(0, sycl::make_ptr<T, sycl::access::address_space::global_space>(&global[global_offset + i]));
    //*global_vec = to_store;
  }
  // We can not store `chunk_size`-sized chunks anymore, so we store the largest we can - `last_chunk_size`-sized one
  int last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (int j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j);
    global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j] = local[local_idx];
  }
  // Less than group size elements remain. Each workitem stores at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + my_last_idx);
    global[global_offset + my_last_idx] = local[local_idx];
  }
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_priv_ptr type of pointer to private memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param local pointer to local memory
 * @param priv pointer to private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= num_elems_per_wi
 * @param local_offset offset to the local pointer
 */
template <std::size_t num_elems_per_wi, bool Pad, typename T_loc_ptr, typename T_priv_ptr>
inline void local2private(T_loc_ptr local, T_priv_ptr priv, std::size_t local_id, std::size_t stride,
                          std::size_t local_offset = 0) {
  for (std::size_t i = 0; i < num_elems_per_wi; i++) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i);
    priv[i] = local[local_idx];
  }
}

/**
 * Copies data from local memory to private memory. Consecutive workitems get
 * consecutive elements.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_priv_ptr type of pointer to private memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param local pointer to local memory
 * @param priv pointer to private memory
 * @param local_id local id of work item
 * @param workers_in_sg how many workitems are working in each subgroup (can be
 * less than subgroup size)
 * @param local_offset offset to the local pointer
 */
template <std::size_t num_elems_per_wi, bool Pad, typename T_loc_ptr, typename T_priv_ptr>
inline void local2private_transposed(T_loc_ptr local, T_priv_ptr priv, std::size_t local_id, std::size_t workers_in_sg,
                                     std::size_t local_offset = 0) {
  for (std::size_t i = 0; i < num_elems_per_wi; i++) {
    priv[i] = local[local_offset + local_id + i * workers_in_sg];
  }
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam Pad whether to skip each SYCL_FFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam T_priv_ptr type of pointer to private memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param priv pointer to private memory
 * @param local pointer to local memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= num_elems_per_wi
 * @param local_offset offset to the local pointer
 */
template <std::size_t num_elems_per_wi, bool Pad, typename T_priv_ptr, typename T_loc_ptr>
inline void private2local(T_priv_ptr priv, T_loc_ptr local, std::size_t local_id, std::size_t stride,
                          std::size_t local_offset = 0) {
  for (std::size_t i = 0; i < num_elems_per_wi; i++) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i);
    local[local_idx] = priv[i];
  }
}

/**
 * Copies data from private memory to local memory. Consecutive workitems write
 * consecutive elements.
 *
 * @tparam num_elems_per_wi Number of elements to copy by each work item
 * @tparam T_priv_ptr type of pointer to private memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @tparam T_loc_ptr type of pointer to local memory. Can be raw pointer or
 * sycl::multi_ptr.
 * @param priv pointer to private memory
 * @param local pointer to local memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param local_offset offset to the local pointer
 */
template <std::size_t num_elems_per_wi, typename T_priv_ptr, typename T_loc_ptr>
inline void private2local_transposed(T_priv_ptr priv, T_loc_ptr local, std::size_t local_id,
                                     std::size_t workers_in_group, std::size_t local_offset = 0) {
  for (std::size_t i = 0; i < num_elems_per_wi; i += 2) {
    local[local_offset + local_id * 2 + i * workers_in_group] = priv[i];
    local[local_offset + local_id * 2 + i * workers_in_group + 1] = priv[i + 1];
  }
}
};  // namespace sycl_fft

#endif
