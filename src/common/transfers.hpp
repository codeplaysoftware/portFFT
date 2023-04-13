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
  for (std::size_t i = local_id; i < total_num_elems; i += local_size) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + i);
    local[local_idx] = global[global_offset + i];
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
  for (std::size_t i = local_id; i < total_num_elems; i += local_size) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + i);
    global[global_offset + i] = local[local_idx];
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
