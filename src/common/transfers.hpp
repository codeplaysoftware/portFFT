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
#include <enums.hpp>
#include <sycl/sycl.hpp>

#ifndef PORTFFT_N_LOCAL_BANKS
#define PORTFFT_N_LOCAL_BANKS 32
#endif

static_assert((PORTFFT_VEC_LOAD_BYTES & (PORTFFT_VEC_LOAD_BYTES - 1)) == 0,
              "PORTFFT_VEC_LOAD_BYTES should be a power of 2!");

/*
To describe the frequency of padding spaces in local memory, we have coined the term "bank line" to describe the chunk
of contiguous memory that exactly fits all of the banks in local memory once. e.g. The NVIDIA Ampere architecture has 32
banks in local memory (shared memory in CUDA terms), each 32 bits. In this case we define a "bank line" as 128 8-bit
bytes.
*/

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

}  // namespace detail

/**
 * Copies data from global memory to local memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <detail::level Level, Idx SubgroupSize, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void global2local(detail::global_data_struct global_data, const T* global, T* local,
                                 Idx total_num_elems, IdxGlobal global_offset = 0,
                                 Idx local_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr Idx ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr Idx ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;
  const char* func_name = __func__;

  global_data.log_message_local(func_name, "total_num_elems", total_num_elems, "global_offset", global_offset,
                                "local_offset", local_offset);

  Idx local_id;
  Idx local_size;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
    local_size = SubgroupSize;
  } else {
    local_id = static_cast<Idx>(global_data.it.get_local_id(0));
    local_size = static_cast<Idx>(global_data.it.get_local_range(0));
  }

  Idx stride = local_size * ChunkSize;
  Idx rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    Idx subgroup_id = static_cast<Idx>(global_data.sg.get_group_id());
    Idx elems_per_sg = detail::divide_ceil(total_num_elems, local_size / SubgroupSize);
    Idx offset = subgroup_id * elems_per_sg;
    Idx next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += static_cast<IdxGlobal>(offset);
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
    local_size = SubgroupSize;
    stride = local_size * ChunkSize;
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup loads a chunk of `ChunkSize * local_size` elements.
  for (Idx i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec loaded = global_data.sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset + static_cast<IdxGlobal>(i)]));
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
        Idx local_idx =
            detail::pad_local<Pad>(local_offset + i + j * local_size, BankLinesPerPad);
        global_data.sg.store(detail::get_local_multi_ptr(&local[local_idx]), loaded[static_cast<int>(j)]);
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
        Idx local_idx = detail::pad_local<Pad>(
            local_offset + i + j * local_size + local_id, BankLinesPerPad);
        local[local_idx] = loaded[static_cast<int>(j)];
        global_data.log_message(func_name, "from", global_offset + static_cast<IdxGlobal>(i + j), "to", local_idx,
                                "value", loaded[static_cast<int>(j)]);
      });
    }
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  Idx unaligned_elements = static_cast<Idx>(global_aligned_ptr - global_ptr);

  // load the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    Idx local_idx = detail::pad_local<Pad>(local_offset + local_id, BankLinesPerPad);
    global_data.log_message(func_name, "first unaligned from", global_offset + static_cast<IdxGlobal>(local_id), "to", local_idx, "value",
                            global[global_offset + static_cast<IdxGlobal>(local_id)]);
    local[local_idx] = global[global_offset + static_cast<IdxGlobal>(local_id)];
  }
  local_offset += unaligned_elements;
  global_offset += static_cast<IdxGlobal>(unaligned_elements);

  // Each workitem loads a chunk of `ChunkSize` consecutive elements. Chunks loaded by a group are consecutive.
  for (Idx i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec loaded;
    loaded = *reinterpret_cast<const T_vec*>(&global[global_offset + static_cast<IdxGlobal>(i)]);
    detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
      Idx local_idx = detail::pad_local<Pad>(local_offset + i + j, BankLinesPerPad);
      global_data.log_message(func_name, "aligned chunk from", global_offset + static_cast<IdxGlobal>(i), "to", local_idx, "value", loaded[static_cast<int>(j)]);
      local[local_idx] = loaded[static_cast<int>(j)];
    });
  }
#endif
  // We can not load `ChunkSize`-sized chunks anymore, so we load the largest we can - `last_chunk_size`-sized one
  Idx last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (Idx j = 0; j < last_chunk_size; j++) {
    Idx local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j, BankLinesPerPad);
    IdxGlobal global_idx = global_offset + static_cast<IdxGlobal>(rounded_down_num_elems + local_id * last_chunk_size + j);
    global_data.log_message(func_name, "last chunk from", global_idx, "to", local_idx, "value", global[global_idx]);
    local[local_idx] = global[global_idx];
  }
  // Less than group size elements remain. Each workitem loads at most one.
  Idx my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    Idx local_idx = detail::pad_local<Pad>(local_offset + my_last_idx, BankLinesPerPad);
    global_data.log_message(func_name, "last element from", global_offset + static_cast<IdxGlobal>(my_last_idx), "to", local_idx, "value",
                            global[global_offset + static_cast<IdxGlobal>(my_last_idx)]);
    local[local_idx] = global[global_offset + static_cast<IdxGlobal>(my_last_idx)];
  }
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param local pointer to local memory
 * @param global pointer to global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <detail::level Level, Idx SubgroupSize, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local2global(detail::global_data_struct global_data, const T* local, T* global,
                                 Idx total_num_elems, Idx local_offset = 0,
                                 IdxGlobal global_offset = 0) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  constexpr Idx ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(T);
  constexpr Idx ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  using T_vec = sycl::vec<T, ChunkSize>;
  const char* func_name = __func__;

  global_data.log_message_local(func_name, "total_num_elems", total_num_elems, "local_offset", local_offset,
                                "global_offset", global_offset);

  Idx local_size;
  Idx local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
    local_size = SubgroupSize;
  } else {
    local_id = static_cast<Idx>(global_data.it.get_local_id(0));
    local_size = static_cast<Idx>(global_data.it.get_local_range(0));
  }

  Idx stride = local_size * ChunkSize;
  Idx rounded_down_num_elems = (total_num_elems / stride) * stride;

#ifdef PORTFFT_USE_SG_TRANSFERS
  if constexpr (Level == detail::level::WORKGROUP) {  // recalculate parameters for subgroup transfer
    Idx subgroup_id = static_cast<Idx>(global_data.sg.get_group_id());
    Idx elems_per_sg = detail::divide_ceil(total_num_elems, local_size / SubgroupSize);
    Idx offset = subgroup_id * elems_per_sg;
    Idx next_offset = (subgroup_id + 1) * elems_per_sg;
    local_offset += offset;
    global_offset += static_cast<IdxGlobal>(offset);
    total_num_elems = sycl::min(total_num_elems, next_offset) - sycl::min(total_num_elems, offset);
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
    local_size = SubgroupSize;
    stride = local_size * ChunkSize;
    rounded_down_num_elems = (total_num_elems / stride) * stride;
  }
  // Each subgroup stores a chunk of `ChunkSize * local_size` elements.
  for (Idx i = 0; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    if constexpr (PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || Pad == detail::pad::DONT_PAD) {
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
        Idx local_idx =
            detail::pad_local<Pad>(local_offset + i + j * local_size, BankLinesPerPad);
        global_data.log_message(func_name, "from", local_idx, "to", global_offset + static_cast<IdxGlobal>(i + j),
                                "value", to_store[static_cast<int>(j)]);
        to_store[static_cast<int>(j)] = global_data.sg.load(detail::get_local_multi_ptr(&local[local_idx]));
      });
    } else {
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
        Idx local_idx = detail::pad_local<Pad>(
            local_offset + i + j * local_size + local_id, BankLinesPerPad);
        global_data.log_message(func_name, "from", local_idx, "to", global_offset + static_cast<IdxGlobal>(i + j),
                                "value", to_store[static_cast<int>(j)]);
        to_store[static_cast<int>(j)] = local[local_idx];
      });
    }
    global_data.sg.store(detail::get_global_multi_ptr(&global[global_offset + static_cast<IdxGlobal>(i)]), to_store);
  }
#else
  const T* global_ptr = &global[global_offset];
  const T* global_aligned_ptr = reinterpret_cast<const T*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(T_vec)));
  Idx unaligned_elements = static_cast<Idx>(global_aligned_ptr - global_ptr);

  // store the first few unaligned elements
  if (local_id < unaligned_elements) {  // assuming unaligned_elements <= local_size
    Idx local_idx = detail::pad_local<Pad>(local_offset + local_id, BankLinesPerPad);
    global_data.log_message(func_name, "first unaligned from", local_idx, "to", global_offset + local_id, "value",
                            local[local_idx]);
    global[global_offset + static_cast<IdxGlobal>(local_id)] = local[local_idx];
  }
  local_offset += unaligned_elements;
  global_offset += static_cast<IdxGlobal>(unaligned_elements);

  // Each workitem stores a chunk of `ChunkSize` consecutive elements. Chunks stored by a group are consecutive.
  for (Idx i = local_id * ChunkSize; i < rounded_down_num_elems; i += stride) {
    T_vec to_store;
    detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) PORTFFT_INLINE {
      Idx local_idx = detail::pad_local<Pad>(local_offset + i + j, BankLinesPerPad);
      global_data.log_message(func_name, "aligned chunk from", local_idx, "to",
                              global_offset + static_cast<IdxGlobal>(i + j), "value", to_store[j]);
      to_store[static_cast<int>(j)] = local[local_idx];
    });
    *reinterpret_cast<T_vec*>(&global[global_offset + static_cast<IdxGlobal>(i)]) = to_store;
  }
#endif
  // We can not store `ChunkSize`-sized chunks anymore, so we store the largest we can - `last_chunk_size`-sized one
  Idx last_chunk_size = (total_num_elems - rounded_down_num_elems) / local_size;
  for (Idx j = 0; j < last_chunk_size; j++) {
    Idx local_idx =
        detail::pad_local<Pad>(local_offset + rounded_down_num_elems + local_id * last_chunk_size + j, BankLinesPerPad);
    IdxGlobal global_idx = global_offset + static_cast<IdxGlobal>(rounded_down_num_elems + local_id * last_chunk_size + j);
    global_data.log_message(func_name, "last chunk from", local_idx, "to", global_idx, "value", local[local_idx]);
    global[global_idx] = local[local_idx];
  }
  // Less than group size elements remain. Each workitem stores at most one.
  Idx my_last_idx = rounded_down_num_elems + last_chunk_size * local_size + local_id;
  if (my_last_idx < total_num_elems) {
    Idx local_idx = detail::pad_local<Pad>(local_offset + my_last_idx, BankLinesPerPad);
    global_data.log_message(func_name, "last element from", local_idx, "to", global_offset + static_cast<IdxGlobal>(my_last_idx), "value",
                            local[local_idx]);
    global[global_offset + static_cast<IdxGlobal>(my_last_idx)] = local[local_idx];
  }
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param local pointer to local memory
 * @param priv pointer to private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <Idx NumElemsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local2private(detail::global_data_struct global_data, const T* local, T* priv, Idx local_id,
                                  Idx stride, Idx local_offset = 0) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "NumElemsPerWI", NumElemsPerWI, "local_id", local_id, "stride", stride,
                                "local_offset", local_offset);
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](Idx i) PORTFFT_INLINE {
    Idx local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i, BankLinesPerPad);
    global_data.log_message(func_name, "from", local_idx, "to", i, "value", local[local_idx]);
    priv[i] = local[local_idx];
  });
}

/**
 * Views the data in the local memory as an NxM matrix, and loads a column into the private memory
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param global_data global data for the kernel
 * @param local Pointer to local memory
 * @param priv Pointer to private memory
 * @param thread_id ID of the working thread in FFT
 * @param col_num Column number which is to be loaded
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <Idx NumElementsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local2private_transposed(detail::global_data_struct global_data, const T* local, T* priv,
                                             Idx thread_id, Idx col_num, Idx stride) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "NumElementsPerWI", NumElementsPerWI, "thread_id", thread_id, "col_num",
                                col_num, "stride", stride);
  detail::unrolled_loop<0, NumElementsPerWI, 1>([&](const Idx i) PORTFFT_INLINE {
    Idx local_idx = detail::pad_local<Pad>(
        2 * stride * (thread_id * NumElementsPerWI + i) + 2 * col_num, BankLinesPerPad);
    global_data.log_message(func_name, "from", local_idx, "to", 2 * i, "value", local[local_idx]);
    global_data.log_message(func_name, "from", local_idx + 1, "to", 2 * i + 1, "value", local[local_idx + 1]);
    priv[2 * i] = local[local_idx];
    priv[2 * i + 1] = local[local_idx + 1];
  });
}

/**
 * Stores data from the local memory to the global memory, in a transposed manner.
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param global_data global data for the kernel
 * @param N Number of rows
 * @param M Number of Cols
 * @param stride Stride between two contiguous elements in global memory in local memory.
 * @param local pointer to the local memory
 * @param global pointer to the global memory
 * @param offset offset to the global memory pointer
 */
template <detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local2global_transposed(detail::global_data_struct global_data, Idx N, Idx M,
                                            Idx stride, T* local, T* global, IdxGlobal offset) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "N", N, "M", M, "stride", stride, "offset", offset);
  Idx num_threads = static_cast<Idx>(global_data.it.get_local_range(0));
  for (Idx i = static_cast<Idx>(global_data.it.get_local_linear_id()); i < N * M; i += num_threads) {
    Idx source_row = i / N;
    Idx source_col = i % N;
    Idx source_index = detail::pad_local<Pad>(2 * (stride * source_col + source_row), BankLinesPerPad);
    sycl::vec<T, 2> v{local[source_index], local[source_index + 1]};
    IdxGlobal global_idx = offset + static_cast<IdxGlobal>(2 * i);
    global_data.log_message(func_name, "from", source_index, "to", global_idx, "value", v);
    *reinterpret_cast<sycl::vec<T, 2>*>(&global[global_idx]) = v;
  }
}

/**
 * Loads data from global memory where consecutive elements of a problem are separated by stride.
 * Loads half of workgroup size equivalent number of consecutive batches from global memory.
 *
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T Scalar Type
 *
 * @param global_data global data for the kernel
 * @param global_base_ptr Global Pointer
 * @param local_ptr Local Pointer
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::level Level, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void global2local_transposed(detail::global_data_struct global_data, const T* global_base_ptr,
                                            T* local_ptr, IdxGlobal offset, Idx num_complex,
                                            IdxGlobal stride_global, Idx stride_local) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "offset", offset, "num_complex", num_complex, "stride_global", stride_global,
                                "stride_local", stride_local);
  Idx local_id;

  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  } else {
    local_id = static_cast<Idx>(global_data.it.get_local_id(0));
  }
  for (Idx i = 0; i < num_complex; i++) {
    Idx local_index = detail::pad_local<Pad>(2 * i * stride_local + local_id, BankLinesPerPad);
    IdxGlobal global_index = offset + static_cast<IdxGlobal>(local_id) + 2 * static_cast<IdxGlobal>(i) * stride_global;
    global_data.log_message(func_name, "from", global_index, "to", local_index, "value", global_base_ptr[global_index]);
    local_ptr[local_index] = global_base_ptr[global_index];
  }
}

/**
 * Views the data in the local memory as an NxM matrix, and stores data from the private memory along the column
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param global_data global data for the kernel
 * @param priv Pointer to private memory
 * @param local Pointer to local memory
 * @param thread_id Id of the working thread for the FFT
 * @param num_workers Number of threads working for that FFt
 * @param col_num Column number in which the data will be stored
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <Idx NumElementsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void private2local_transposed(detail::global_data_struct global_data, const T* priv, T* local,
                                             Idx thread_id, Idx num_workers, Idx col_num, Idx stride) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "thread_id", thread_id, "num_workers", num_workers, "col_num", col_num,
                                "stride", stride);
  detail::unrolled_loop<0, NumElementsPerWI, 1>([&](const Idx i) PORTFFT_INLINE {
    Idx loc_base_offset = detail::pad_local<Pad>(
        2 * stride * (i * num_workers + thread_id) + 2 * col_num, BankLinesPerPad);
    global_data.log_message(func_name, "from", 2 * i, "to", loc_base_offset, "value", priv[2 * i]);
    global_data.log_message(func_name, "from", 2 * i + 1, "to", loc_base_offset + 1, "value", priv[2 * i + 1]);
    local[loc_base_offset] = priv[2 * i];
    local[loc_base_offset + 1] = priv[2 * i + 1];
  });
}

/**
 * Copies data from private to local memory, allowing separate strides for thread id and id of element in workitem,
 * allowing for up to two transposes of the data.
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 *
 * @param global_data global data for the kernel
 * @param priv Pointer to private memory
 * @param local Pointer to local memory
 * @param thread_id Id of the working thread for the FFT
 * @param stride_num_workers stride in local memory between consecutive elements in a workitem
 * @param destination_offset Offset to local memory destination
 * @param stride Stride in local memory between consecutive workitems
 */
template <Idx NumElementsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void private2local_2strides(detail::global_data_struct global_data, const T* priv, T* local,
                                           Idx thread_id, Idx stride_num_workers, Idx destination_offset, Idx stride) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "thread_id", thread_id, "stride_num_workers", stride_num_workers,
                                "destination_offset", destination_offset, "stride", stride);
  detail::unrolled_loop<0, NumElementsPerWI, 1>([&](const Idx i) PORTFFT_INLINE {
    Idx loc_base_offset = detail::pad_local<Pad>(
        2 * (stride_num_workers * i + stride * thread_id + destination_offset),
        BankLinesPerPad);
    global_data.log_message(func_name, "from", 2 * i, "to", loc_base_offset, "value", priv[2 * i]);
    global_data.log_message(func_name, "from", 2 * i + 1, "to", loc_base_offset + 1, "value", priv[2 * i + 1]);
    local[loc_base_offset] = priv[2 * i];
    local[loc_base_offset + 1] = priv[2 * i + 1];
  });
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param priv pointer to private memory
 * @param local pointer to local memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <Idx NumElemsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void private2local(detail::global_data_struct global_data, const T* priv, T* local, Idx local_id,
                                  Idx stride, Idx local_offset = 0) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "stride", stride, "local_offset", local_offset);
  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](Idx i) PORTFFT_INLINE {
    Idx local_idx = detail::pad_local<Pad>(local_offset + local_id * stride + i, BankLinesPerPad);
    global_data.log_message(func_name, "from", i, "to", local_idx, "value", priv[i]);
    local[local_idx] = priv[i];
  });
}

/**
 * Copies data from private memory to local or global memory. Consecutive workitems write
 * consecutive elements. The copy is done jointly by a group of threads defined by `local_id` and `workers_in_group`.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T type of the scalar used for computations
 * @tparam TDstIdx type of destination index
 * @param global_data global data for the kernel
 * @param priv pointer to private memory
 * @param destination pointer to destination - local or global memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param destination_offset offset to the destination pointer
 */
template <Idx NumElemsPerWI, detail::pad Pad, Idx BankLinesPerPad, typename T, typename TDstIdx>
PORTFFT_INLINE void store_transposed(detail::global_data_struct global_data, const T* priv, T* destination,
                                     Idx local_id, Idx workers_in_group,
                                     TDstIdx destination_offset = 0) {
  static_assert(((Pad == detail::pad::DO_PAD) && std::is_same_v<TDstIdx, Idx>) || 
                ((Pad == detail::pad::DONT_PAD) && std::is_same_v<TDstIdx, IdxGlobal>));
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "workers_in_group", workers_in_group,
                                "destination_offset", destination_offset);
  constexpr Idx VecSize = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<T, VecSize>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv);
  T_vec* destination_vec = reinterpret_cast<T_vec*>(&destination[0]);

  detail::unrolled_loop<0, NumElemsPerWI, 2>([&](Idx i) PORTFFT_INLINE {
    TDstIdx destination_idx_unpadded = destination_offset + static_cast<TDstIdx>(local_id * 2 + i * workers_in_group);
    TDstIdx destination_idx;
    if constexpr(Pad == detail::pad::DO_PAD){
      destination_idx = detail::pad_local<Pad>(destination_idx_unpadded, BankLinesPerPad);
    } else{
      destination_idx = destination_idx_unpadded;
    }
    global_data.log_message(func_name, "from", i, "to", destination_idx, "value", priv[i]);
    global_data.log_message(func_name, "from", i + 1, "to", destination_idx + 1, "value", priv[i + 1]);
    if (destination_idx % 2 == 0) {  // if the destination address is aligned, we can use vector store
      destination_vec[destination_idx / 2] = priv_vec[i / 2];
    } else {
      destination[destination_idx] = priv[i];
      destination[destination_idx + 1] = priv[i + 1];
    }
  });
}

/**
 * Function meant to transfer data between local and private memory, and is able to handle 3 levels
 * of transpositions / strides and combine them into a single load / store.
 *
 * @tparam T Scalar Type
 * @tparam Pad Whether or not to pad
 * @tparam NumComplexElements Number of complex elements to transfer between the two.
 * @tparam TransferDirection Direction of Transfer
 *
 * @param global_data global data for the kernel
 * @param priv Pointer to private memory
 * @param loc Pointer to local memory
 * @param stride_1 Innermost stride
 * @param offset_1 Innermost offset
 * @param stride_2 2nd level of stride
 * @param offset_2 2nd level of offset
 * @param stride_3 Outermost stride
 * @param offset_3 Outermost offset
 * @param bank_lines_per_pad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 */
template <detail::transfer_direction TransferDirection, detail::pad Pad, Idx NumComplexElements, typename T>
PORTFFT_INLINE void transfer_strided(detail::global_data_struct global_data, T* priv, T* loc, Idx stride_1,
                                     Idx offset_1, Idx stride_2, Idx offset_2,
                                     Idx stride_3, Idx offset_3, Idx bank_lines_per_pad) {
  const char* func_name = __func__;
  global_data.log_message_local(__func__, "stride_1", stride_1, "offset_1", offset_1, "stride_2", stride_2, "offset_2",
                                offset_2, "stride_3", stride_3, "offset_3", offset_3);
  detail::unrolled_loop<0, NumComplexElements, 1>([&](const Idx j) PORTFFT_INLINE {
    Idx base_offset = stride_1 * (stride_2 * (j * stride_3 + offset_3) + offset_2) + offset_1;
    if constexpr (TransferDirection == detail::transfer_direction::LOCAL_TO_PRIVATE) {
      global_data.log_message(func_name, "from", detail::pad_local<Pad>(base_offset, bank_lines_per_pad), "to", 2 * j,
                              "value", loc[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)]);
      global_data.log_message(func_name, "from", detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad), "to",
                              2 * j + 1, "value", loc[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)]);
      priv[2 * j] = loc[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)];
      priv[2 * j + 1] = loc[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_LOCAL) {
      global_data.log_message(func_name, "from", 2 * j, "to", detail::pad_local<Pad>(base_offset, bank_lines_per_pad),
                              "value", priv[2 * j]);
      global_data.log_message(func_name, "from", 2 * j + 1, "to",
                              detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad), "value", priv[2 * j + 1]);
      loc[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)] = priv[2 * j];
      loc[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)] = priv[2 * j + 1];
    }
  });
}

/**
 * Transfers data from local memory which is strided to global memory, which too is strided in a transposed fashion
 *
 * @tparam Pad Whether or not to pad local memory
 * @tparam T Scalar type
 *
 * @param loc Pointer to local memory
 * @param global Pointer to global memory
 * @param global_offset Offset to global memory
 * @param local_stride stride value in local memory
 * @param N Number of rows
 * @param M Number of Columns
 * @param fft_size Size of the problem
 * @param bank_lines_per_pad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 * @param global_data global data for the kernel
 */
template <detail::pad Pad, typename T>
PORTFFT_INLINE void local_strided_2_global_strided_transposed(T* loc, T* global, IdxGlobal global_offset,
                                                              Idx local_stride, Idx N, Idx M,
                                                              Idx fft_size, Idx bank_lines_per_pad,
                                                              detail::global_data_struct global_data) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "global_offset", global_offset, "local_stride", local_stride, "N", N, "M", M,
                                "fft_size", fft_size);
  Idx batch_num = static_cast<Idx>(global_data.it.get_local_linear_id()) / 2;
  for (Idx i = 0; i < fft_size; i++) {
    Idx source_row = i / N;
    Idx source_col = i % N;
    Idx local_idx = detail::pad_local<Pad>(
        local_stride * (source_col * M + source_row) + static_cast<Idx>(global_data.it.get_local_id(0)), bank_lines_per_pad);
    IdxGlobal global_idx =
        global_offset + static_cast<IdxGlobal>(2 * batch_num * fft_size + 2 * i + static_cast<Idx>(global_data.it.get_local_linear_id()) % 2);
    global_data.log_message(func_name, "from", local_idx, "to", global_idx, "value", loc[local_idx]);
    global[global_idx] = loc[local_idx];
  }
}

};  // namespace portfft

#endif
