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
#include <common/logging.hpp>
#include <common/memory_views.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

static_assert((PORTFFT_VEC_LOAD_BYTES & (PORTFFT_VEC_LOAD_BYTES - 1)) == 0,
              "PORTFFT_VEC_LOAD_BYTES should be a power of 2!");

/*
To describe the frequency of padding spaces in local memory, we have coined the term "bank line" to describe the chunk
of contiguous memory that exactly fits all of the banks in local memory once. e.g. The NVIDIA Ampere architecture has 32
banks in local memory (shared memory in CUDA terms), each 32 bits. In this case we define a "bank line" as 128 8-bit
bytes.
*/

namespace portfft {

/**
 * Copies data from global memory to local memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @tparam LocalViewT The type of the local memory view
 * @param global_data global data for the kernel
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <detail::level Level, int SubgroupSize, typename T, typename LocalViewT>
PORTFFT_INLINE void global2local(detail::global_data_struct global_data, const T* global, LocalViewT local,
                                 std::size_t total_num_elems, std::size_t global_offset = 0,
                                 std::size_t local_offset = 0) {
  static_assert(std::is_same_v<T, typename LocalViewT::element_type>, "Different source / destination types.");
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;
  const char* func_name = __func__;

  global_data.log_message_local(func_name, "total_num_elems", total_num_elems, "global_offset", global_offset,
                                "local_offset", local_offset);

  std::size_t local_id;
  std::size_t local_size;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = global_data.sg.get_local_linear_id();
    local_size = SubgroupSize;
  } else {
    local_id = global_data.it.get_local_id(0);
    local_size = global_data.it.get_local_range(0);
  }

  std::size_t stride = local_size * static_cast<std::size_t>(ChunkSize);
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    std::size_t subgroup_id = global_data.sg.get_group_id();
    std::size_t elems_per_sg = detail::divide_ceil<std::size_t>(total_num_elems, local_size / SubgroupSize);
    std::size_t offset = subgroup_id * elems_per_sg;
    std::size_t next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = global_data.sg.get_local_linear_id();
    local_size = SubgroupSize;
    stride = local_size * ChunkSize;
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup loads a chunk of `ChunkSize * local_size` elements.
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec loaded = global_data.sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset + i]));
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || !LocalViewT::is_padded) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        global_data.sg.store(
            detail::get_local_multi_ptr(&local[local_offset + i + static_cast<std::size_t>(j) * local_size]),
            loaded[j]);
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
        std::size_t local_idx = local_offset + i + static_cast<std::size_t>(j) * local_size + local_id;
        local[local_idx] = loaded[j];
        global_data.log_message(func_name, "from", global_offset + i + static_cast<std::size_t>(j), "to", local_idx,
                                "value", loaded[j]);
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
    std::size_t local_idx = local_offset + local_id;
    global_data.log_message(func_name, "first unaligned from", global_offset + local_id, "to", local_idx, "value",
                            global[global_offset + local_id]);
    local[local_idx] = global[global_offset + local_id];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem loads a chunk of `ChunkSize` consecutive elements. Chunks loaded by a group are consecutive.
  for (std::size_t i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec loaded;
    loaded = *reinterpret_cast<const T_vec*>(&global[global_offset + i]);
    detail::unrolled_loop<0, ChunkSize, 1>([&](int j) __attribute__((always_inline)) {
      std::size_t local_idx = local_offset + i + static_cast<std::size_t>(j);
      global_data.log_message(func_name, "aligned chunk from", global_offset + i, "to", local_idx, "value", loaded[j]);
      local[local_idx] = loaded[j];
    });
  }
#endif
  // We can not load `ChunkSize`-sized chunks anymore, so we load the largest we can - `last_chunk_size`-sized one
  std::size_t last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (std::size_t j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx = local_offset + rounded_down_num_elems + local_id * last_chunk_size + j;
    std::size_t global_idx = global_offset + rounded_down_num_elems + local_id * last_chunk_size + j;
    global_data.log_message(func_name, "last chunk from", global_idx, "to", local_idx, "value", global[global_idx]);
    local[local_idx] = global[global_idx];
  }
  // Less than group size elements remain. Each workitem loads at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    std::size_t local_idx = local_offset + my_last_idx;
    global_data.log_message(func_name, "last element from", global_offset + my_last_idx, "to", local_idx, "value",
                            global[global_offset + my_last_idx]);
    local[local_idx] = global[global_offset + my_last_idx];
  }
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam LocalViewT The type of the local memory view
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param local pointer to local memory
 * @param global pointer to global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <detail::level Level, int SubgroupSize, typename LocalViewT, typename T>
PORTFFT_INLINE void local2global(detail::global_data_struct global_data, const LocalViewT local, T* global,
                                 std::size_t total_num_elems, std::size_t local_offset = 0,
                                 std::size_t global_offset = 0) {
  static_assert(std::is_same_v<typename LocalViewT::element_type, T>, "Mismatching source / destination element type.");
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr int ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;
  const char* func_name = __func__;

  global_data.log_message_local(func_name, "total_num_elems", total_num_elems, "local_offset", local_offset,
                                "global_offset", global_offset);

  std::size_t local_size;
  std::size_t local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = global_data.sg.get_local_linear_id();
    local_size = SubgroupSize;
  } else {
    local_id = global_data.it.get_local_id(0);
    local_size = global_data.it.get_local_range(0);
  }

  std::size_t stride = local_size * static_cast<std::size_t>(ChunkSize);
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    std::size_t subgroup_id = global_data.sg.get_group_id();
    std::size_t elems_per_sg = detail::divide_ceil<std::size_t>(total_num_elems, local_size / SubgroupSize);
    std::size_t offset = subgroup_id * elems_per_sg;
    std::size_t next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += offset;
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = global_data.sg.get_local_linear_id();
    local_size = SubgroupSize;
    stride = local_size * static_cast<std::size_t>(ChunkSize);
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup stores a chunk of `ChunkSize * local_size` elements.
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || !LocalViewT::is_padded) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) PORTFFT_INLINE {
        std::size_t local_idx = local_offset + i + static_cast<std::size_t>(j) * local_size;
        to_store[j] = global_data.sg.load(detail::get_local_multi_ptr(&local[local_idx]));
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](int j) PORTFFT_INLINE {
        std::size_t local_idx = local_offset + i + static_cast<std::size_t>(j) * local_size + local_id;
        global_data.log_message(func_name, "from", local_idx, "to", global_offset + i + static_cast<std::size_t>(j),
                                "value", to_store[j]);
        to_store[j] = local[local_idx];
      });
    }
    global_data.sg.store(detail::get_global_multi_ptr(&global[global_offset + i]), to_store);
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  std::size_t unaligned_elements = static_cast<std::size_t>(global_aligned_ptr - global_ptr);

  // store the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    std::size_t local_idx = local_offset + local_id;
    global_data.log_message(func_name, "first unaligned from", local_idx, "to", global_offset + local_id, "value",
                            local[local_idx]);
    global[global_offset + local_id] = local[local_idx];
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;

  // Each workitem stores a chunk of `ChunkSize` consecutive elements. Chunks stored by a group are consecutive.
  for (std::size_t i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    detail::unrolled_loop<0, ChunkSize, 1>([&](int j) PORTFFT_INLINE((always_inline)) {
      std::size_t local_idx = local_offset + i + static_cast<std::size_t>(j);
      global_data.log_message(func_name, "aligned chunk from", local_idx, "to",
                              global_offset + i + static_cast<std::size_t>(j), "value", to_store[j]);
      to_store[j] = local[local_idx];
    });
    *reinterpret_cast<T_vec*>(&global[global_offset + i]) = to_store;
  }
#endif
  // We can not store `ChunkSize`-sized chunks anymore, so we store the largest we can - `last_chunk_size`-sized one
  std::size_t last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (std::size_t j = 0; j < last_chunk_size; j++) {
    std::size_t local_idx = local_offset + rounded_down_num_elems + local_id * last_chunk_size + j;
    std::size_t global_idx = global_offset + rounded_down_num_elems + local_id * last_chunk_size + j;
    global_data.log_message(func_name, "last chunk from", local_idx, "to", global_idx, "value", local[local_idx]);
    global[global_idx] = local[local_idx];
  }
  // Less than group size elements remain. Each workitem stores at most one.
  std::size_t my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    global_data.log_message(func_name, "last element from", local_offset + my_last_idx, "to",
                            global_offset + my_last_idx, "value", local[local_offset + my_last_idx]);
    global[global_offset + my_last_idx] = local[local_offset + my_last_idx];
  }
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam PrivViewT The type of the private memory view
 * @tparam LocalViewT A view of local memory. Type must match T
 * @param global_data global data for the kernel
 * @param local A local memory view
 * @param priv View of private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <std::size_t NumElemsPerWI, typename LocalViewT, typename PrivViewT>
PORTFFT_INLINE inline void local2private(detail::global_data_struct global_data, const LocalViewT local, PrivViewT priv,
                                         std::size_t local_id, std::size_t stride, std::size_t local_offset = 0) {
  static_assert(std::is_same_v<typename PrivViewT::element_type, typename LocalViewT::element_type>,
                "Different source / destination element types.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "NumElemsPerWI", NumElemsPerWI, "local_id", local_id, "stride", stride,
                                "local_offset", local_offset);
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](std::size_t i) __attribute__((always_inline)) {
    std::size_t local_idx = local_offset + local_id * stride + i;
    global_data.log_message(func_name, "from", local_idx, "to", i, "value", local[local_idx]);
    priv[i] = local[local_offset + local_id * stride + i];
  });
}

/**
 * Stores data from the local memory to the global memory, in a transposed manner.
 * @tparam LocalViewT The type of the local memory view
 * @tparam T type of the scalar used for computations
 *
 * @param global_data global data for the kernel
 * @param N Number of rows
 * @param M Number of Cols
 * @param stride Stride between two contiguous elements in global memory in local memory.
 * @param local view of local memory
 * @param global pointer to the global memory
 * @param offset offset to the global memory pointer
 */
template <typename LocalViewT, typename T>
PORTFFT_INLINE void local2global_transposed(detail::global_data_struct global_data, std::size_t N, std::size_t M,
                                            std::size_t stride, const LocalViewT local, T* global, std::size_t offset) {
  static_assert(std::is_same_v<T, typename LocalViewT::element_type>, "Different source / destination element types.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "N", N, "M", M, "stride", stride, "offset", offset);
  std::size_t num_threads = global_data.it.get_local_range(0);
  for (std::size_t i = global_data.it.get_local_linear_id(); i < N * M; i += num_threads) {
    std::size_t source_row = i / N;
    std::size_t source_col = i % N;
    std::size_t source_index = 2 * (stride * source_col + source_row);
    sycl::vec<T, 2> v{local[source_index], local[source_index + 1]};
    global_data.log_message(func_name, "from", source_index, "to", offset + 2 * i, "value", v);
    *reinterpret_cast<sycl::vec<T, 2>*>(&global[offset + 2 * i]) = v;
  }
}

/**
 * Loads data from global memory where consecutive elements of a problem are separated by stride.
 * Loads half of workgroup size equivalent number of consecutive batches from global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T Scalar Type
 * @tparam LocalViewT The type of the local memory view
 *
 * @param global_data global data for the kernel
 * @param global_base_ptr Global Pointer
 * @param local Local memory view
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::level Level, typename LocalViewT, typename T>
PORTFFT_INLINE inline void global2local_transposed(detail::global_data_struct global_data, const T* global_base_ptr,
                                                   LocalViewT local, std::size_t offset, std::size_t num_complex,
                                                   std::size_t stride_global, std::size_t stride_local) {
  static_assert(std::is_same_v<T, typename LocalViewT::element_type>, "Different source / destination element types.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "offset", offset, "num_complex", num_complex, "stride_global", stride_global,
                                "stride_local", stride_local);
  std::size_t local_id;

  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = global_data.sg.get_local_linear_id();
  } else {
    local_id = global_data.it.get_local_id(0);
  }
  for (std::size_t i = 0; i < num_complex; i++) {
    std::size_t global_index = offset + local_id + 2 * i * stride_global;
    std::size_t local_index = 2 * i * stride_local + local_id;
    global_data.log_message(func_name, "from", global_index, "to", local_index, "value", global_base_ptr[global_index]);
    local[local_index] = global_base_ptr[global_index];
  }
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam PrivViewT The type of the private memory view
 * @tparam LocalViewT The view type of local memory
 * @param global_data global data for the kernel
 * @param priv A private memory view
 * @param local A local memory view
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <std::size_t NumElemsPerWI, typename PrivateViewT, typename LocalViewT>
PORTFFT_INLINE void private2local(detail::global_data_struct global_data, const PrivateViewT priv, LocalViewT local,
                                  std::size_t local_id, std::size_t stride, std::size_t local_offset = 0) {
  static_assert(std::is_same_v<typename PrivateViewT::element_type, typename LocalViewT::element_type>,
                "Source / destination element type mismatch.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "stride", stride, "local_offset", local_offset);
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](std::size_t i) __attribute__((always_inline)) {
    std::size_t local_idx = local_offset + local_id * stride + i;
    global_data.log_message(func_name, "from", i, "to", local_idx, "value", priv[i]);
    local[local_idx] = priv[i];
  });
}

/**
 * Copies data from private memory to local or global memory. Consecutive workitems write
 * consecutive elements. The copy is done jointly by a group of threads defined by `local_id` and `workers_in_group`.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam T type of the scalar used for computations
 * @tparam DestViewT The view type of destination memory
 * @param global_data global data for the kernel
 * @param priv pointer to private memory
 * @param destination pointer to destination - local or global memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param destination_offset offset to the destination pointer
 */
template <int NumElemsPerWI, typename T, typename DestViewT>
PORTFFT_INLINE void store_transposed(detail::global_data_struct global_data, const T* priv, DestViewT destination,
                                     std::size_t local_id, std::size_t workers_in_group,
                                     std::size_t destination_offset = 0) {
  static_assert(std::is_same_v<T, typename DestViewT::element_type>, "Source / destination element type mismatch.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "workers_in_group", workers_in_group,
                                "destination_offset", destination_offset);
  constexpr int VecSize = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<T, VecSize>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv);
  T_vec* destination_vec = reinterpret_cast<T_vec*>(&destination[0]);

  detail::unrolled_loop<0, NumElemsPerWI, 2>([&](int i) PORTFFT_INLINE {
    std::size_t destination_idx = destination_offset + local_id * 2 + static_cast<std::size_t>(i) * workers_in_group;
    global_data.log_message(func_name, "from", i, "to", destination_idx, "value", priv[i]);
    global_data.log_message(func_name, "from", i + 1, "to", destination_idx + 1, "value", priv[i + 1]);
    if (!DestViewT::is_padded &&
        destination_idx % 2 == 0) {  // if the destination address is aligned, we can use vector store
      destination_vec[destination_idx / 2] = priv_vec[i / 2];
    } else {
      destination[destination_idx] = priv[i];
      destination[destination_idx + 1] = priv[i + 1];
    }
  });
}

/**
 * Transfer data between local and private memory, with 3 levels of transpositions / strides.
 * priv[i] <-> loc[s1 (s2 (s3 * i + o3) + o2) + o1]
 *
 * @tparam TransferDirection Direction of Transfer
 * @tparam NumComplexElements Number of complex elements to transfer between the two.
 * @tparam PrivViewT The view type of the private memory
 * @tparam LocalViewT The view type of local memory
 *
 * @param global_data global data for the kernel
 * @param priv A view of private memory
 * @param loc A view of local memory
 * @param stride_1 Innermost stride
 * @param offset_1 Innermost offset
 * @param stride_2 2nd level of stride
 * @param offset_2 2nd level of offset
 * @param stride_3 Outermost stride
 * @param offset_3 Outermost offset
 */
template <detail::transfer_direction TransferDirection, int NumComplexElements, typename PrivViewT, typename LocalViewT>
PORTFFT_INLINE void transfer_strided(detail::global_data_struct global_data, PrivViewT priv, LocalViewT loc,
                                     std::size_t stride_1, std::size_t offset_1, std::size_t stride_2,
                                     std::size_t offset_2, std::size_t stride_3, std::size_t offset_3) {
  static_assert(std::is_same_v<typename PrivViewT::element_type, typename LocalViewT::element_type>,
                "Source / destination element type mismatch.");
  const char* func_name = __func__;
  global_data.log_message_local(__func__, "stride_1", stride_1, "offset_1", offset_1, "stride_2", stride_2, "offset_2",
                                offset_2, "stride_3", stride_3, "offset_3", offset_3);
  detail::unrolled_loop<0, NumComplexElements, 1>([&](const int j) PORTFFT_INLINE {
    std::size_t j_size_t = static_cast<std::size_t>(j);
    std::size_t base_offset = stride_1 * (stride_2 * (j_size_t * stride_3 + offset_3) + offset_2) + offset_1;
    if constexpr (TransferDirection == detail::transfer_direction::LOCAL_TO_PRIVATE) {
      global_data.log_message(func_name, "from", base_offset, "to", j, "value", loc[base_offset]);
      priv[j] = loc[base_offset];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_LOCAL) {
      global_data.log_message(func_name, "from", j, "to", base_offset, "value", priv[j]);
      loc[base_offset] = priv[j];
    }
  });
}

/**
 * Views the data in the local memory as an NxM matrix, and stores data from the private memory along the column
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam LocalViewT The type of the local memory view
 * @tparam PrivViewT The type of the private memory view
 *
 * @param priv Pointer to private memory
 * @param local Pointer to local memory
 * @param thread_id Id of the working thread for the FFT
 * @param num_workers Number of threads working for that FFt
 * @param col_num Column number in which the data will be stored
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <int NumElementsPerWI, typename PrivateViewT, typename LocalViewT>
__attribute__((always_inline)) inline void private2local_transposed(detail::global_data_struct global_data,
                                                                    const PrivateViewT priv, LocalViewT local,
                                                                    std::size_t thread_id, std::size_t num_workers,
                                                                    std::size_t col_num, std::size_t stride) {
  transfer_strided<detail::transfer_direction::PRIVATE_TO_LOCAL, NumElementsPerWI>(
      global_data, priv, local, 1, 0, stride, col_num, num_workers, thread_id);
}

/**
 * Views the data in the local memory as an NxM matrix, and loads a column into the private memory
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam LocalViewT The type of the local memory view
 * @tparam PrivViewT The type of the private memory view
 *
 * @param local View of local memory
 * @param priv View of private memory
 * @param thread_id ID of the working thread in FFT
 * @param col_num Column number which is to be loaded
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <int NumElementsPerWI, typename LocalViewT, typename PrivViewT>
__attribute__((always_inline)) inline void local2private_transposed(detail::global_data_struct global_data,
                                                                    const LocalViewT local, PrivViewT priv,
                                                                    std::size_t thread_id, std::size_t col_num,
                                                                    std::size_t stride) {
  transfer_strided<detail::transfer_direction::LOCAL_TO_PRIVATE, NumElementsPerWI>(
      global_data, priv, local, 1, 0, stride, col_num, 1, thread_id * NumElementsPerWI);
}

/**
 * Transfers data from local memory which is strided to global memory, which too is strided in a transposed fashion
 *
 * @tparam LocalViewT The view type of local memory
 * @tparam T Scalar type
 *
 * @param loc Pointer to local memory
 * @param global Pointer to global memory
 * @param global_offset Offset to global memory
 * @param local_stride stride value in local memory
 * @param N Number of rows
 * @param M Number of Columns
 * @param fft_size Size of the problem
 * @param global_data global data for the kernel
 */
template <typename LocalViewT, typename T>
PORTFFT_INLINE void local_strided_2_global_strided_transposed(LocalViewT loc, T* global, std::size_t global_offset,
                                                              std::size_t local_stride, std::size_t N, std::size_t M,
                                                              std::size_t fft_size,
                                                              detail::global_data_struct global_data) {
  static_assert(std::is_same_v<T, typename LocalViewT::element_type>, "Source / destination element type mismatch.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "global_offset", global_offset, "local_stride", local_stride, "N", N, "M", M,
                                "fft_size", fft_size);
  std::size_t batch_num = global_data.it.get_local_linear_id() / 2;
  for (std::size_t i = 0; i < fft_size; i++) {
    std::size_t source_row = i / N;
    std::size_t source_col = i % N;
    std::size_t local_idx = local_stride * (source_col * M + source_row) + global_data.it.get_local_id(0);
    std::size_t global_idx =
        global_offset + 2 * batch_num * fft_size + 2 * i + global_data.it.get_local_linear_id() % 2;
    global_data.log_message(func_name, "from", local_idx, "to", global_idx, "value", loc[local_idx]);
    global[global_idx] = loc[local_idx];
  }
}

};  // namespace portfft

#endif
