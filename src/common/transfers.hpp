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

#ifndef PORTFFT_COMMON_TRANSFERS_HPP
#define PORTFFT_COMMON_TRANSFERS_HPP

#include <common/helpers.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

#ifndef PORTFFT_N_LOCAL_BANKS
#define PORTFFT_N_LOCAL_BANKS 32
#endif

static_assert((PORTFFT_VEC_LOAD_BYTES & (PORTFFT_VEC_LOAD_BYTES - 1)) == 0,
              "PORTFFT_VEC_LOAD_BYTES should be a power of 2!");

namespace portfft {

namespace detail {

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
 * @param LinesPerPad A padding space will be added after `LinesPerPad` groups of `PORTFFT_N_LOCAL_BANKS` banks.
 * @return transformed local_idx
 */
template <detail::pad Pad = detail::pad::DO_PAD>
__attribute__((always_inline)) inline std::size_t pad_local(std::size_t local_idx, std::size_t LinesPerPad = 1) {
  if constexpr (Pad == detail::pad::DO_PAD) {
    local_idx += local_idx / (PORTFFT_N_LOCAL_BANKS * LinesPerPad);
  }
  return local_idx;
}

}  // namespace detail

/**
 * Copies data from global memory to local memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam Pad whether to skip each PORTFFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param it nd_item
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <detail::level Level, int SubgroupSize, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void global2local(sycl::nd_item<1> it, const T* global, T* local,
                                                        std::size_t total_num_elems, std::size_t global_offset = 0,
                                                        std::size_t local_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;

  sycl::sub_group sg = it.get_sub_group();
  std::size_t local_id;
  std::size_t local_size;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = sg.get_local_linear_id();
    local_size = SubgroupSize;
  } else {
    local_id = it.get_local_id(0);
    local_size = it.get_local_range(0);
  }

  std::size_t stride = local_size * static_cast<std::size_t>(ChunkSize);
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    std::size_t subgroup_id = sg.get_group_id();
    std::size_t elems_per_sg = detail::divide_ceil<std::size_t>(total_num_elems, local_size / SubgroupSize);
    std::size_t offset = subgroup_id * elems_per_sg;
    std::size_t next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = sg.get_local_linear_id();
    local_size = SubgroupSize;
    stride = local_size * ChunkSize;
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup loads a chunk of `ChunkSize * local_size` elements.
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec loaded = sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset + i]));
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        std::size_t local_idx =
            detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j) * local_size, LinesPerPad);
        sg.store(detail::get_local_multi_ptr(&local[local_idx]), loaded[j]);
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        std::size_t local_idx =
            detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j) * local_size + local_id, LinesPerPad);
        local[local_idx] = loaded[j];
      });
    }
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  std::size_t unaligned_elements = static_cast<std::size_t>(global_aligned_ptr - global_ptr);

  // load the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id, LinesPerPad);
    local[local_idx] = global[global_offset + local_id];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem loads a chunk of `ChunkSize` consecutive elements. Chunks loaded by a group are consecutive.
  for (std::size_t i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec loaded;
    loaded.load(0, detail::get_global_multi_ptr(&global[global_offset + i]));
    detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
      std::size_t local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j), LinesPerPad);
      local[local_idx] = loaded[j];
    });
  }
#endif
  // We can not load `ChunkSize`-sized chunks anymore, so we load the largest we can - `last_chunk_size`-sized one
  std::size_t last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (std::size_t j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j, LinesPerPad);
    local[local_idx] = global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j];
  }
  // Less than group size elements remain. Each workitem loads at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + my_last_idx, LinesPerPad);
    local[local_idx] = global[global_offset + my_last_idx];
  }
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam Pad whether to skip each LinesPerPad*PORTFFT_N_LOCAL_BANKS element in local to allow
 * strided reads without bank conflicts
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param it nd_item
 * @param local pointer to local memory
 * @param global pointer to global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <detail::level Level, int SubgroupSize, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void local2global(sycl::nd_item<1> it, const T* local, T* global,
                                                        std::size_t total_num_elems, std::size_t local_offset = 0,
                                                        std::size_t global_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;

  sycl::sub_group sg = it.get_sub_group();
  std::size_t local_size;
  std::size_t local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = sg.get_local_linear_id();
    local_size = SubgroupSize;
  } else {
    local_id = it.get_local_id(0);
    local_size = it.get_local_range(0);
  }

  std::size_t stride = local_size * static_cast<std::size_t>(ChunkSize);
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    std::size_t subgroup_id = sg.get_group_id();
    std::size_t elems_per_sg = detail::divide_ceil<std::size_t>(total_num_elems, local_size / SubgroupSize);
    std::size_t offset = subgroup_id * elems_per_sg;
    std::size_t next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = sg.get_local_linear_id();
    local_size = SubgroupSize;
    stride = local_size * static_cast<std::size_t>(ChunkSize);
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup stores a chunk of `ChunkSize * local_size` elements.
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        std::size_t local_idx =
            detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j) * local_size, LinesPerPad);
        to_store[j] = sg.load(detail::get_local_multi_ptr(&local[local_idx]));
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        std::size_t local_idx =
            detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j) * local_size + local_id, LinesPerPad);
        to_store[j] = local[local_idx];
      });
    }
    sg.store(detail::get_global_multi_ptr(&global[global_offset + i]), to_store);
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  std::size_t unaligned_elements = static_cast<std::size_t>(global_aligned_ptr - global_ptr);

  // store the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id, LinesPerPad);
    global[global_offset + local_id] = local[local_idx];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem stores a chunk of `ChunkSize` consecutive elements. Chunks stored by a group are consecutive.
  for (std::size_t i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
      std::size_t local_idx = detail::pad_local<Pad>(local_offset + i + static_cast<std::size_t>(j), LinesPerPad);
      to_store[j] = local[local_idx];
    });
    to_store.store(0, detail::get_global_multi_ptr(&global[global_offset + i]));
  }
#endif
  // We can not store `ChunkSize`-sized chunks anymore, so we store the largest we can - `last_chunk_size`-sized one
  std::size_t last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (std::size_t j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j, LinesPerPad);
    global[global_offset + rounded_down_num_elems + local_id * last_chunk_size + j] = local[local_idx];
  }
  // Less than group size elements remain. Each workitem stores at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + my_last_idx, LinesPerPad);
    global[global_offset + my_last_idx] = local[local_idx];
  }
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad whether to skip each PORTFFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param local pointer to local memory
 * @param priv pointer to private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <std::size_t NumElemsPerWI, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void local2private(const T* local, T* priv, std::size_t local_id,
                                                         std::size_t stride, std::size_t local_offset = 0) {
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](std::size_t i) __attribute__((always_inline)) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i, LinesPerPad);
    priv[i] = local[local_idx];
  });
}

/**
 * Views the data in the local memory as an NxM matrix, and loads a column into the private memory
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam Pad Whether data in the local memory is padded or not
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param local Pointer to local memory
 * @param priv Pointer to private memory
 * @param thread_id ID of the working thread in FFT
 * @param col_num Column number which is to be loaded
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <int NumElementsPerWI, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void local2private_transposed(const T* local, T* priv, int thread_id, int col_num,
                                                                    int stride) {
  detail::unrolled_loop<0, NumElementsPerWI, 1>([&](const int i) __attribute__((always_inline)) {
    std::size_t local_idx = detail::pad_local<Pad>(
        static_cast<std::size_t>(2 * stride * (thread_id * NumElementsPerWI + i) + 2 * col_num), LinesPerPad);
    priv[2 * i] = local[local_idx];
    priv[2 * i + 1] = local[local_idx + 1];
  });
}

/**
 * Stores data from the local memory to the global memory, in a transposed manner.
 * @tparam Pad Whether or not to consider local memory as padded
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param it Associated nd_item
 * @param N Number of rows
 * @param M Number of Cols
 * @param stride Stride between two contiguous elements in global memory in local memory.
 * @param local pointer to the local memory
 * @param global pointer to the global memory
 * @param offset offset to the global memory pointer
 */
template <detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void local2global_transposed(sycl::nd_item<1> it, std::size_t N, std::size_t M,
                                                                   std::size_t stride, T* local, T* global,
                                                                   std::size_t offset) {
  std::size_t num_threads = it.get_local_range(0);
  for (std::size_t i = it.get_local_linear_id(); i < N * M; i += num_threads) {
    std::size_t source_row = i / N;
    std::size_t source_col = i % N;
    std::size_t source_index = detail::pad_local<Pad>(2 * (stride * source_col + source_row), LinesPerPad);
    sycl::vec<T, 2> v{local[source_index], local[source_index + 1]};
    *reinterpret_cast<sycl::vec<T, 2>*>(&global[offset + 2 * i]) = v;
  }
}

/**
 * Loads data from global memory where consecutive elements of a problem are separated by stride.
 * Loads half of workgroup size equivalent number of consecutive batches from global memory.
 *
 * @tparam pad Whether or not to consider padding in local memory
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T Scalar Type
 *
 * @param it Associated nd_item
 * @param global_base_ptr Global Pointer
 * @param local_ptr Local Pointer
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::level Level, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void global2local_transposed(sycl::nd_item<1> it, const T* global_base_ptr,
                                                                   T* local_ptr, std::size_t offset,
                                                                   std::size_t num_complex, std::size_t stride_global,
                                                                   std::size_t stride_local) {
  sycl::sub_group sg = it.get_sub_group();
  std::size_t local_id;

  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = sg.get_local_linear_id();
  } else {
    local_id = it.get_local_id(0);
  }
  for (std::size_t i = 0; i < num_complex; i++) {
    std::size_t local_index = detail::pad_local<Pad>(2 * i * stride_local + local_id, LinesPerPad);
    std::size_t global_index = offset + local_id + 2 * i * stride_global;
    local_ptr[local_index] = global_base_ptr[global_index];
  }
}

/**
 * Views the data in the local memory as an NxM matrix, and stores data from the private memory along the column
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam Pad Whether data in the local memory is padded or not
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param priv Pointer to private memory
 * @param local Pointer to local memory
 * @param thread_id Id of the working thread for the FFT
 * @param num_workers Number of threads working for that FFt
 * @param col_num Column number in which the data will be stored
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <int NumElementsPerWI, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void private2local_transposed(const T* priv, T* local, int thread_id,
                                                                    int num_workers, int col_num, int stride) {
  detail::unrolled_loop<0, NumElementsPerWI, 1>([&](const int i) __attribute__((always_inline)) {
    std::size_t loc_base_offset = detail::pad_local<Pad>(
        static_cast<std::size_t>(2L * stride * (i * num_workers + thread_id) + 2L * col_num), LinesPerPad);
    local[loc_base_offset] = priv[2 * i];
    local[loc_base_offset + 1] = priv[2 * i + 1];
  });
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad whether to skip each PORTFFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param priv pointer to private memory
 * @param local pointer to local memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <std::size_t NumElemsPerWI, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void private2local(const T* priv, T* local, std::size_t local_id,
                                                         std::size_t stride, std::size_t local_offset = 0) {
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](std::size_t i) __attribute__((always_inline)) {
    std::size_t local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i, LinesPerPad);
    local[local_idx] = priv[i];
  });
}

/**
 * Copies data from private memory to local or global memory. Consecutive workitems write
 * consecutive elements. The copy is done jointly by a group of threads defined by `local_id` and `workers_in_group`.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad whether to skip each PORTFFT_N_LOCAL_BANKS element in local avoiding bank conflicts
 * @tparam LinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param priv pointer to private memory
 * @param destination pointer to destination - local or global memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param destination_offset offset to the destination pointer
 */
template <int NumElemsPerWI, detail::pad Pad, std::size_t LinesPerPad = 1, typename T>
__attribute__((always_inline)) inline void store_transposed(const T* priv, T* destination, std::size_t local_id,
                                                            std::size_t workers_in_group,
                                                            std::size_t destination_offset = 0) {
  constexpr int VecSize = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<T, VecSize>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv);
  T_vec* destination_vec = reinterpret_cast<T_vec*>(&destination[0]);

  detail::unrolled_loop<0, NumElemsPerWI, 2>([&](int i) __attribute__((always_inline)) {
    std::size_t destination_idx = detail::pad_local<Pad>(
        destination_offset + local_id * 2 + static_cast<std::size_t>(i) * workers_in_group, LinesPerPad);
    if (destination_idx % 2 == 0) {  // if the destination address is aligned, we can use vector store
      destination_vec[destination_idx / 2] = priv_vec[i / 2];
    } else {
      destination[destination_idx] = priv[i];
      destination[destination_idx + 1] = priv[i + 1];
    }
  });
}
};  // namespace portfft

#endif
