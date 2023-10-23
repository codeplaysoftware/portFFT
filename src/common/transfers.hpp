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
#include <defines.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>
#include <traits.hpp>

namespace portfft {

namespace detail {

namespace impl {
/** Copy between index-contiguous global and local memory using sub-group loads/stores for real data.
 *  Works on a fixed size block. Arguments are expected to be the same for all values in the sub-group.
 *  This function is expected to be called from `subgroup_block_copy`.
 *  global[global_offset + i] <-> local[local_offset + i] for i in [0, SgBlockCopyBlockSize)
 *
 *  @tparam TransferDirection Direction of memory transfer
 *  @tparam SubgroupSize The subgroup size
 *  @tparam ChunkSize The size of vector for the sub-group to use subgroup loads with.
 *  @tparam GlobalViewT The type of the global memory view
 *  @tparam LocalViewT The type of the local memory view
 *  @param global_data global data for the kernel
 *  @param global The global memory view to copy to/from. Expects to be real element type
 *  @param global_offset The offset into global memory to start copying at
 *  @param local The local memory view. Expects to be real element type
 *  @param local_offset The offset into local memory to start copying at
 *  @returns The number of reals copied
 */
template <transfer_direction TransferDirection, Idx SubgroupSize, Idx ChunkSize, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE Idx subgroup_single_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                              IdxGlobal global_offset, LocalViewT local, Idx local_offset) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  constexpr Idx SgBlockCopyBlockSize = ChunkSize * SubgroupSize;
  using vec_t = sycl::vec<real_t, ChunkSize>;

  // Is the local memory suitable for using Intel's subgroup copy extensions with?
  // NB: This assumes any offset aligns with padding in a padded view
  constexpr bool IsSgContiguous = PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || IsContiguousViewV<LocalViewT>;
  const char* func_name = __func__;
  global_data.log_message_subgroup(func_name, "SgBlockCopyBlockSize", SgBlockCopyBlockSize, "global_offset",
                                   global_offset, "local_offset", local_offset, "IsSgContiguous", IsSgContiguous);
  Idx local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  // A helper function to generate indexes in local memory.
  auto index_transform = [=](Idx i) PORTFFT_INLINE { return local_offset + i * SubgroupSize + local_id; };
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    vec_t vec = global_data.sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset]));
    PORTFFT_UNROLL
    for (Idx j = 0; j < ChunkSize; j++) {
      if constexpr (IsSgContiguous) {
        global_data.sg.store(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]),
                             vec[static_cast<int>(j)]);
      } else {
        local[index_transform(j)] = vec[static_cast<int>(j)];
      }
    };
  } else {
    vec_t vec;
    PORTFFT_UNROLL
    for (Idx j = 0; j < ChunkSize; j++) {
      if constexpr (IsSgContiguous) {
        vec[static_cast<int>(j)] =
            global_data.sg.load(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]));
      } else {
        vec[static_cast<int>(j)] = local[index_transform(j)];
      }
    };
    global_data.sg.store(detail::get_global_multi_ptr(&global[global_offset]), vec);
  }
  return SgBlockCopyBlockSize;
}

/** Copy between index-contiguous global and local memory using sub-groups loads/stores for real data.
 *  Data does not need to be aligned to alignof(vec_t). Arguments are expected to be the same for all values in the
 *  sub-group. Copies in multiples of SgBlockCopyBlockSize, and may copy less than the given item count n.
 *  global[global_offset + i] <-> local[local_offset + i] for i in [0, m) where m <= n
 *
 *  @tparam TransferDirection Direction of memory transfer
 *  @tparam Level Is this being called at subgroup or work-group level?
 *  @tparam SubgroupSize The subgroup size
 *  @tparam ChunkSize The size of vector for the sub-group to use subgroup loads with.
 *  @tparam Pad Do or don't pad local memory
 *  @tparam BankLinesPerPad Paramater for local memory padding
 *  @tparam GlobalViewT The view of local memory
 *  @tparam LocalViewT The type of the local memory view
 *  @param global_data global data for the kernel
 *  @param global The global memory view to copy to/from. Expects to be real element type
 *  @param global_offset The offset into global memory to start copying at
 *  @param local The local memory view. Expects to be real element type
 *  @param local_offset The offset into local memory to start copying at
 *  @param n The count of reals to copy
 *  @returns The number of reals copied. May be less than n.
 */
template <transfer_direction TransferDirection, level Level, Idx ChunkSize, Idx SubgroupSize, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE Idx subgroup_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                       IdxGlobal global_offset, LocalViewT local, Idx local_offset, Idx n) {
  static constexpr Idx BlockSize = ChunkSize * SubgroupSize;
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  static_assert(std::is_same_v<real_t, get_element_remove_cv_t<LocalViewT>>, "Mismatch between global and local types");
  static_assert(Level == level::SUBGROUP || Level == level::WORKGROUP, "Only subgroup and workgroup level supported");

  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset, "n",
                                        n);

  Idx block_count = n / BlockSize;
  if constexpr (Level == level::SUBGROUP) {
    for (Idx block_idx{0}; block_idx < block_count; ++block_idx) {
      Idx offset = block_idx * BlockSize;
      subgroup_single_block_copy<TransferDirection, SubgroupSize, ChunkSize>(
          global_data, global, global_offset + offset, local, local_offset + offset);
    }
  } else {  // Level == level::WORKGROUP
    auto sg = global_data.sg;
    Idx subgroup_id = static_cast<Idx>(sg.get_group_id());
    Idx subgroup_count = static_cast<Idx>(sg.get_group_linear_range());
    // NB: For work-groups this may lead to divergence between sub-groups on the final loop iteration.
    for (Idx block_idx{subgroup_id}; block_idx < block_count; block_idx += subgroup_count) {
      Idx offset = block_idx * BlockSize;
      subgroup_single_block_copy<TransferDirection, SubgroupSize, ChunkSize>(
          global_data, global, global_offset + offset, local, local_offset + offset);
    }
  }
  global_data.log_message_scoped<Level>(func_name, "copied_value_count", block_count * BlockSize);
  return block_count * BlockSize;
}

/** Copy between index-contiguous global and local memory using work-group or sub-group. Global memory argument must
 * be aligned to alignof(sycl::vec<real_t, ChunkSize). Copies in block of Chunksize * group.get_local_range(0) elements,
 * and may copy fewer elements than requested.
 * global[global_offset + i] <-> local[local_offset + i] for i in [0, m) where m <= n.
 *
 *  @tparam TransferDirection Direction of memory transfer
 *  @tparam Level Group type to use for copies.
 *  @tparam ChunkSize The element count for sycl::vec<RealT, ChunkSize>
 *  @tparam GlobalViewT The view of local memory
 *  @tparam LocalViewT The type of the local memory view
 *  @param global_data global data for the kernel
 *  @param global The global memory view to copy to/from. Expects to be real element type
 *  @param global_offset The offset into global memory to start copying at
 *  @param local The local memory view. Expects to be real element type
 *  @param local_offset The offset into local memory to start copying at
 *  @param n The desired number of reals to copy
 *  @returns The number of reals copied - may be less than n
 */
template <transfer_direction TransferDirection, level Level, Idx ChunkSize, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE Idx vec_aligned_group_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                                IdxGlobal global_offset, LocalViewT local, Idx local_offset, Idx n) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  using vec_t = sycl::vec<real_t, ChunkSize>;
  auto group = global_data.get_group<Level>();

  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "ChunkSize", ChunkSize, "global_offset", global_offset,
                                        "local_offset", local_offset, "copy_block_size",
                                        ChunkSize * group.get_local_range()[0], "n", n);

  Idx block_size = ChunkSize * static_cast<Idx>(group.get_local_range()[0]);
  Idx block_count = n / block_size;
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  Idx wi_offset = local_id * ChunkSize;
  auto index_transform = [=](Idx inner, Idx outer)
                             PORTFFT_INLINE { return local_offset + wi_offset + inner + outer * block_size; };
  for (Idx loop_idx{0}; loop_idx < block_count; ++loop_idx) {
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      vec_t loaded;
      loaded = *reinterpret_cast<const vec_t*>(&global[global_offset + wi_offset + block_size * loop_idx]);
      PORTFFT_UNROLL
      for (Idx j = 0; j < ChunkSize; j++) {
        local[index_transform(j, loop_idx)] = loaded[static_cast<int>(j)];
      };
    } else {  // LOCAL_TO_GLOBAL
      vec_t to_store;
      PORTFFT_UNROLL
      for (Idx j = 0; j < ChunkSize; j++) {
        to_store[static_cast<int>(j)] = local[index_transform(j, loop_idx)];
      };
      *reinterpret_cast<vec_t*>(&global[global_offset + wi_offset + block_size * loop_idx]) = to_store;
    }
  }
  global_data.log_message_scoped<Level>(func_name, "copied_value_count", block_count * block_size);
  return block_count * block_size;
}

/** Copy between index-contiguous global and local memory for n values, where n < group.get_local_range()[0].
 *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
 *
 *  @tparam TransferDirection Direction of memory transfer
 *  @tparam Level Group type to use for copies.
 *  @tparam GlobalViewT The view of local memory
 *  @tparam LocalViewT The type of the local memory view
 *  @param global_data global data for the kernel
 *  @param global The global memory view to copy to/from. Expects to be real element type
 *  @param global_offset The offset into global memory to start copying at
 *  @param local The local memory view. Expects to be real element type
 *  @param local_offset The offset into local memory to start copying at
 *  @param n The number of reals to copy. Must be less than group.get_local_range()[0]
 *  @returns The number of reals copied
 */
template <transfer_direction TransferDirection, level Level, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE Idx subrange_copy(detail::global_data_struct global_data, GlobalViewT global, IdxGlobal global_offset,
                                 LocalViewT local, Idx local_offset, Idx n) {
  auto group = global_data.get_group<Level>();
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset,
                                        "group_size", group.get_local_range()[0], "n", n);
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  if (local_id < n) {
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      local[local_offset + local_id] = global[global_offset + static_cast<IdxGlobal>(local_id)];
    } else {  // LOCAL_TO_GLOBAL
      global[global_offset + static_cast<IdxGlobal>(local_id)] = local[local_offset + local_id];
    }
  }
  return n;
}

/** Copy between index-contiguous global and local memory for n values. No particular requirements for alignment.
 *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
 *
 *  @tparam TransferDirection Direction of memory transfer
 *  @tparam Level Group type to use for copies.
 *  @tparam GlobalViewT The view of local memory
 *  @tparam LocalViewT The type of the local memory view
 *  @param global_data global data for the kernel
 *  @param group The sub-group or work-group
 *  @param global The global memory view to copy to/from. Expects to be real element type
 *  @param global_offset The offset into global memory to start copying at
 *  @param local The local memory view. Expects to be real element type
 *  @param local_offset The offset into local memory to start copying at
 *  @param n The number of reals to copy
 *  @returns The number of reals copied
 */
template <transfer_direction TransferDirection, level Level, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE Idx naive_copy(detail::global_data_struct global_data, GlobalViewT global, IdxGlobal global_offset,
                              LocalViewT local, Idx local_offset, Idx n) {
  auto group = global_data.get_group<Level>();
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  Idx local_size = static_cast<Idx>(group.get_local_range()[0]);
  Idx loop_iters = n / local_size;
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset, "n",
                                        n);
  for (Idx j = 0; j < loop_iters; j++) {
    Idx local_idx = local_offset + local_id + j * local_size;
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      local[local_idx] = global[global_offset + local_id + j * local_size];
    } else {  // LOCAL_TO_GLOBAL
      global[global_offset + local_id + j * local_size] = local[local_idx];
    }
  }
  Idx loop_copies = loop_iters * local_size;
  subrange_copy<TransferDirection, Level>(global_data, global, global_offset + loop_copies, local,
                                          local_offset + loop_copies, n - loop_copies);
  return n;
}

}  // namespace impl

/**
 * Copies data from global memory to local memory. Expects the value of most input arguments to be the
 * same for work-items in the group described by template parameter "Level".
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam LocalViewT The type of the local memory view
 * @tparam GlobalViewT The type of the global memory view
 * @param global_data global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <transfer_direction TransferDirection, level Level, Idx SubgroupSize, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE void global_local_contiguous_copy(detail::global_data_struct global_data, GlobalViewT global,
                                                 LocalViewT local, Idx total_num_elems, IdxGlobal global_offset = 0,
                                                 Idx local_offset = 0) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  static_assert(std::is_floating_point_v<real_t>, "Expecting floating-point data type");
  static_assert(std::is_same_v<real_t, get_element_remove_cv_t<LocalViewT>>, "Type mismatch between global and local");
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset);
  static constexpr Idx ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(real_t);
  static constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;

#ifdef PORTFFT_USE_SG_TRANSFERS
  Idx copied_by_sg = impl::subgroup_block_copy<TransferDirection, Level, ChunkSize, SubgroupSize>(
      global_data, global, global_offset, local, local_offset, total_num_elems);
  local_offset += copied_by_sg;
  global_offset += copied_by_sg;
  total_num_elems -= copied_by_sg;
#else
  using vec_t = sycl::vec<real_t, ChunkSize>;
  const real_t* global_ptr = &global[global_offset];
  const real_t* global_aligned_ptr = reinterpret_cast<const real_t*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(vec_t)));
  Idx unaligned_elements = static_cast<Idx>(global_aligned_ptr - global_ptr);

  // Load the first few unaligned elements. Assumes group size > alignof(vec_t) / sizeof(vec_t).
  impl::subrange_copy<TransferDirection, Level>(global_data, global, global_offset, local, local_offset,
                                                unaligned_elements);
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;
  total_num_elems -= unaligned_elements;

  // Each workitem loads a chunk of consecutive elements. Chunks loaded by a group are consecutive.
  Idx block_copied_elements = impl::vec_aligned_group_block_copy<TransferDirection, Level, ChunkSize>(
      global_data, global, global_offset, local, local_offset, total_num_elems);
  local_offset += block_copied_elements;
  global_offset += block_copied_elements;
  total_num_elems -= block_copied_elements;
#endif
  // We cannot load fixed-size blocks of data anymore, so we use naive copies.
  impl::naive_copy<TransferDirection, Level>(global_data, global, global_offset, local, local_offset, total_num_elems);
}

}  // namespace detail

/**
 * Copies data from global memory to local memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam LocalViewT The type of the local memory view
 * @tparam GlobalViewT The type of the global memory view
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <detail::level Level, Idx SubgroupSize, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE void global2local(detail::global_data_struct global_data, GlobalViewT global, LocalViewT local,
                                 Idx total_num_elems, IdxGlobal global_offset = 0, Idx local_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::GLOBAL_TO_LOCAL, Level, SubgroupSize>(
      global_data, global, local, total_num_elems, global_offset, local_offset);
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam LocalViewT The type of the local memory view
 * @tparam GlobalViewT The type of the global memory view
 * @tparam T type of the scalar used for computations
 * @param global_data global data for the kernel
 * @param local View of local memory
 * @param global View of global memory
 * @param total_num_elems total number of values to copy per group
 * @param local_offset offset to the local pointer
 * @param global_offset offset to the global pointer
 */
template <detail::level Level, Idx SubgroupSize, typename LocalViewT, typename GlobalViewT>
PORTFFT_INLINE void local2global(detail::global_data_struct global_data, LocalViewT local, GlobalViewT global,
                                 Idx total_num_elems, Idx local_offset = 0, IdxGlobal global_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::LOCAL_TO_GLOBAL, Level, SubgroupSize>(
      global_data, global, local, total_num_elems, global_offset, local_offset);
}

/**
 * Copies data from local memory to private memory. Each work item gets a chunk
 * of consecutive values from local memory.
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam PrivT The type of view of private memory
 * @tparam LocalT The type of view of local memory
 * @param global_data global data for the kernel
 * @param local View of local memory
 * @param priv View of private memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local pointer
 */
template <typename LocalT, typename PrivT>
PORTFFT_INLINE void local2private(Idx num_elements_per_wi, detail::global_data_struct global_data, LocalT local,
                                  PrivT priv, Idx local_id, Idx stride, Idx local_offset = 0) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "NumElemsPerWI", num_elements_per_wi, "local_id", local_id, "stride", stride,
                                "local_offset", local_offset);
  PORTFFT_UNROLL
  for (Idx i = 0; i < num_elements_per_wi; i++) {
    Idx local_idx = local_offset + local_id * stride + i;
    global_data.log_message(func_name, "from", local_idx, "to", i, "value", local[local_idx]);
    priv[i] = local[local_idx];
  };
}

/**
 * Stores data from the local memory to the global memory, in a transposed manner.
 * global[offset + 2 * i] = local[2 * (stride * (i % N) + (i / N))]
 * global[offset + 2 * i + 1] = local[2 * (stride * (i % N) + (i / N)) + 1]
 * for i in [0, N * M)
 *
 * @tparam LocalT The view type of local memory
 * @tparam GlobalT The view type of global memory
 *
 * @param global_data global data for the kernel
 * @param N Number of rows
 * @param M Number of Cols
 * @param stride Stride between two contiguous elements in global memory in local memory.
 * @param local View of local memory
 * @param global View of global memory
 * @param offset offset to the global memory pointer
 */
template <typename LocalT, typename GlobalT>
PORTFFT_INLINE void local2global_transposed(detail::global_data_struct global_data, Idx N, Idx M, Idx stride,
                                            LocalT local, GlobalT global, IdxGlobal offset) {
  using real_t = detail::get_element_remove_cv_t<LocalT>;
  static_assert(std::is_same_v<real_t, detail::get_element_t<GlobalT>>, "Type mismatch between local and global views");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "N", N, "M", M, "stride", stride, "offset", offset);
  Idx num_threads = static_cast<Idx>(global_data.it.get_local_range(0));
  for (Idx i = static_cast<Idx>(global_data.it.get_local_linear_id()); i < N * M; i += num_threads) {
    Idx source_row = i / N;
    Idx source_col = i % N;
    Idx source_index = 2 * (stride * source_col + source_row);
    sycl::vec<real_t, 2> v{local[source_index], local[source_index + 1]};
    IdxGlobal global_idx = offset + static_cast<IdxGlobal>(2 * i);
    global_data.log_message(func_name, "from", source_index, "to", global_idx, "value", v);
    *reinterpret_cast<sycl::vec<real_t, 2>*>(&global[global_idx]) = v;
  }
}

/**
 * Loads data from global memory where consecutive elements of a problem are separated by stride.
 * Loads half of workgroup size equivalent number of consecutive batches from global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam GlobalT The global memory view type
 * @tparam LocalT The local memory view type
 *
 * @param global_data global data for the kernel
 * @param global A view of global memory
 * @param local A view of local
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::level Level, typename GlobalT, typename LocalT>
PORTFFT_INLINE void global2local_transposed(detail::global_data_struct global_data, GlobalT global, LocalT local,
                                            IdxGlobal offset, Idx num_complex, IdxGlobal stride_global,
                                            Idx stride_local) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<GlobalT>, detail::get_element_t<LocalT>>,
                "Type mismatch between global and local views");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "offset", offset, "num_complex", num_complex, "stride_global", stride_global,
                                "stride_local", stride_local);
  Idx local_id = static_cast<Idx>(global_data.get_group<Level>().get_local_id()[0]);
  for (Idx i = 0; i < num_complex; i++) {
    Idx local_index = 2 * i * stride_local + local_id;
    IdxGlobal global_index = offset + static_cast<IdxGlobal>(local_id) + 2 * static_cast<IdxGlobal>(i) * stride_global;
    global_data.log_message(func_name, "from", global_index, "to", local_index, "value", global[global_index]);
    local[local_index] = global[global_index];
  }
}

/**
 * Copies data from private memory to local memory. Each work item writes a
 * chunk of consecutive values to local memory.
 * local[local_offset + local_id * stride + i] = priv[i] for i in [0, NumElemsPerWI)
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam PrivT The type of view of private memory
 * @tparam LocalT The type of view of local memorys
 * @param global_data global data for the kernel
 * @param priv A view of private memory
 * @param local A view of local memory
 * @param local_id local id of work item
 * @param stride stride between two chunks assigned to consecutive work items.
 * Should be >= NumElemsPerWI
 * @param local_offset offset to the local base index
 */
template <typename PrivT, typename LocalT>
PORTFFT_INLINE void private2local(Idx num_elemets_per_wi, detail::global_data_struct global_data, PrivT priv,
                                  LocalT local, Idx local_id, Idx stride, Idx local_offset = 0) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "stride", stride, "local_offset", local_offset);
  PORTFFT_UNROLL
  for (Idx i = 0; i < num_elemets_per_wi; i++) {
    global_data.log_message(func_name, "from", i, "to", local_offset + local_id * stride + i, "value", priv[i]);
    local[local_offset + local_id * stride + i] = priv[i];
  }
}

/**
 * Copies data from private memory to local or global memory. Consecutive workitems write
 * consecutive elements. The copy is done jointly by a group of threads defined by `local_id` and `workers_in_group`.
 * destination[destination_offset + local_id * 2 + i * workers_in_group    ] := priv[i]
 * destination[destination_offset + local_id * 2 + i * workers_in_group + 1] := priv[i + 1]
 * for i in [0, NumElemsPerWI)
 *
 * @tparam NumElemsPerWI Number of elements to copy by each work item
 * @tparam PrivT The type of the private memory view
 * @tparam DestT The type of the destination memory view
 * @tparam TDstIdx type of destination index
 * @param global_data global data for the kernel
 * @param priv View of private memory
 * @param destination View of destination - local or global memory
 * @param local_id local id of work item
 * @param workers_in_group how many workitems are working in each group (can be
 * less than the group size)
 * @param destination_offset offset to the destination pointer
 */
template <typename PrivT, typename DestT, typename TDstIdx>
PORTFFT_INLINE void store_transposed(Idx num_elements_per_wi, detail::global_data_struct global_data, PrivT priv,
                                     DestT destination, Idx local_id, Idx workers_in_group,
                                     TDstIdx destination_offset = 0) {
  using real_t = detail::get_element_remove_cv_t<PrivT>;
  static_assert(std::is_same_v<real_t, detail::get_element_t<DestT>>,
                "Type mismatch between private and destination views");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "workers_in_group", workers_in_group,
                                "destination_offset", destination_offset);
  constexpr Idx VecSize = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<real_t, VecSize>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv);
  T_vec* destination_vec = reinterpret_cast<T_vec*>(&destination[0]);
  PORTFFT_UNROLL
  for (Idx i = 0; i < num_elements_per_wi; i += 2) {
    TDstIdx destination_idx = destination_offset + static_cast<TDstIdx>(local_id * 2 + i * workers_in_group);
    global_data.log_message(func_name, "from", i, "to", destination_idx, "value", priv[i]);
    global_data.log_message(func_name, "from", i + 1, "to", destination_idx + 1, "value", priv[i + 1]);
    auto dest_ptr_real = reinterpret_cast<std::uintptr_t>(&destination[destination_idx]);
    auto dest_ptr_imag = reinterpret_cast<std::uintptr_t>(&destination[destination_idx + 1]);
    if (dest_ptr_real % alignof(T_vec) == 0 && (dest_ptr_imag + sizeof(real_t) == dest_ptr_real)) {
      // If the destination address is aligned and contiguous, we can use vector store
      destination_vec[destination_idx / 2] = priv_vec[i / 2];
    } else {
      destination[destination_idx] = priv[i];
      destination[destination_idx + 1] = priv[i + 1];
    }
  }
}

/**
 * Transfer data between local and private memory, with 3 levels of transpositions / strides.
 * local_or_global_idx = s1 * (s2 * (s3 * i + o3) + o2) + o1
 * private_idx = 2 * i
 * output[out_idx] = input[in_idx]
 * output[out_idx + 1] = input[in_idx + 1]
 * where out_idx and in_idx are chosen according to TransferDirection
 * for i in [0, NumComplexElements) where loc is indexed repecting padding.
 *
 * @tparam TransferDirection Direction of Transfer
 * @tparam NumComplexElements Number of complex elements to transfer between the two.
 * @tparam TDstIdx type of destination index
 * @tparam InputT The type of the input memory view
 * @tparam DestT The type of the dest memory view
 *
 * @param global_data global data for the kernel
 * @param input Input view
 * @param output Output view
 * @param stride_1 Innermost stride
 * @param offset_1 Innermost offset
 * @param stride_2 2nd level of stride
 * @param offset_2 2nd level of offset
 * @param stride_3 Outermost stride
 * @param offset_3 Outermost offset
 */
template <detail::transfer_direction TransferDirection, typename TDstIdx, typename InputT, typename DestT>
PORTFFT_INLINE void transfer_strided(Idx num_complex_elements, detail::global_data_struct global_data, InputT input,
                                     DestT output, TDstIdx stride_1, TDstIdx offset_1, TDstIdx stride_2,
                                     TDstIdx offset_2, TDstIdx stride_3, TDstIdx offset_3) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<InputT>, detail::get_element_t<DestT>>,
                "Type mismatch between local and private views");
  const char* func_name = __func__;
  global_data.log_message_local(__func__, "stride_1", stride_1, "offset_1", offset_1, "stride_2", stride_2, "offset_2",
                                offset_2, "stride_3", stride_3, "offset_3", offset_3);
  PORTFFT_UNROLL
  for (Idx j = 0; j < num_complex_elements; j++) {
    TDstIdx base_offset = stride_1 * (stride_2 * static_cast<TDstIdx>((j * stride_3 + offset_3)) + offset_2) + offset_1;
    if constexpr (TransferDirection == detail::transfer_direction::LOCAL_TO_PRIVATE) {
      global_data.log_message(func_name, "from", base_offset, "to", 2 * j, "value", input[base_offset]);
      global_data.log_message(func_name, "from", base_offset + 1, "to", 2 * j + 1, "value", input[base_offset + 1]);
      output[2 * j] = input[base_offset];
      output[2 * j + 1] = input[base_offset + 1];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_LOCAL) {
      global_data.log_message(func_name, "from", 2 * j, "to", base_offset, "value", input[2 * j]);
      global_data.log_message(func_name, "from", 2 * j + 1, "to", base_offset + 1, "value", input[2 * j + 1]);
      output[base_offset] = input[2 * j];
      output[base_offset + 1] = input[2 * j + 1];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
      output[base_offset] = input[2 * j];
      output[base_offset + 1] = input[2 * j + 1];
    }
  }
}

/**
 * Views the data in the local memory as an NxM matrix, and stores data from the private memory along the column:
 * loc[2 * stride * (num_workers * i + thread_id) + 2 * col_num] := priv[i]
 * loc[2 * stride * (num_workers * i + thread_id) + 2 * col_num + 1] := priv[i + 1]
 * for i in [0, NumElementsPerWI).
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam PrivT The type of view of private memory
 * @tparam LocalT The type of view of local memory
 *
 * @param global_data global data for the kernel
 * @param priv View of private memory
 * @param local View of local memory
 * @param thread_id Id of the working thread for the FFT
 * @param num_workers Number of threads working for that FFt
 * @param col_num Column number in which the data will be stored
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <typename PrivT, typename LocalT>
PORTFFT_INLINE void private2local_transposed(Idx num_elements_per_wi, detail::global_data_struct global_data,
                                             PrivT priv, LocalT local, Idx thread_id, Idx num_workers, Idx col_num,
                                             Idx stride) {
  transfer_strided<detail::transfer_direction::PRIVATE_TO_LOCAL>(num_elements_per_wi, global_data, priv, local, 1, 0,
                                                                 2 * stride, 2 * col_num, num_workers, thread_id);
}

/**
 * Views the data in the local memory as an NxM matrix, and loads a column into the private memory
 * priv[2 * i] := loc[2 * stride * (i + thread_id * NumElementsPerWI) + 2 * col_num]
 * priv[2 * i + 1] := loc[2 * stride * (i + thread_id * NumElementsPerWI) + 2 * col_num + 1]
 * for i in [0, NumElementsPerWI)
 *
 * @tparam NumElementsPerWI Elements per workitem
 * @tparam PrivT The type of view of private memory
 * @tparam LocalT The type of view of local memory
 *
 * @param global_data global data for the kernel
 * @param local View of local memory
 * @param priv View of private memory
 * @param thread_id ID of the working thread in FFT
 * @param col_num Column number which is to be loaded
 * @param stride Inner most dimension of the reinterpreted matrix
 */
template <typename LocalT, typename PrivT>
PORTFFT_INLINE void local2private_transposed(Idx num_elements_per_wi, detail::global_data_struct global_data,
                                             LocalT local, PrivT priv, Idx thread_id, Idx col_num, Idx stride) {
  transfer_strided<detail::transfer_direction::LOCAL_TO_PRIVATE>(
      num_elements_per_wi, global_data, local, priv, 1, 0, 2 * stride, 2 * col_num, 1, thread_id * num_elements_per_wi);
}

/**
 * Transfers data from local memory which is strided to global memory, which too is strided in a transposed fashion
 * global[global_offset + 2 * batch_num * fft_size + 2 * i + local_id % 2] =
 *        loc[local_stride * ((i % N) * M + i / N) + local_id] for i in [0, fft_size)
 *
 * @tparam LocalT The type of view of local memory
 * @tparam GlobalT The type of view of global memory
 *
 * @param global_data global data for the kernel
 * @param loc View of local memory
 * @param global View of global memory
 * @param global_offset Offset to global memory
 * @param local_stride stride value in local memory
 * @param N Number of rows
 * @param M Number of Columns
 * @param fft_size Size of the problem
 */
template <typename LocalT, typename GlobalT>
PORTFFT_INLINE void local_strided_2_global_strided_transposed(detail::global_data_struct global_data, LocalT loc,
                                                              GlobalT global, IdxGlobal global_offset, Idx local_stride,
                                                              Idx N, Idx M, Idx fft_size) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<LocalT>, detail::get_element_remove_cv_t<GlobalT>>,
                "Type mismatch between local and global views");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "global_offset", global_offset, "local_stride", local_stride, "N", N, "M", M,
                                "fft_size", fft_size);
  Idx batch_num = static_cast<Idx>(global_data.it.get_local_linear_id()) / 2;
  for (Idx i = 0; i < fft_size; i++) {
    Idx source_row = i / N;
    Idx source_col = i % N;
    Idx local_idx = local_stride * (source_col * M + source_row) + static_cast<Idx>(global_data.it.get_local_id(0));
    IdxGlobal global_idx =
        global_offset + static_cast<IdxGlobal>(2 * batch_num * fft_size + 2 * i +
                                               static_cast<Idx>(global_data.it.get_local_linear_id()) % 2);
    global_data.log_message(func_name, "from", local_idx, "to", global_idx, "value", loc[local_idx]);
    global[global_idx] = loc[local_idx];
  }
}

/**
 * Stores data to global memory where consecutive elements of a problem are separated by stride.
 * Data layout in local memory is same as global memory. Each workitem is responsible for
 * transferring all of either real or imaginary components of the computed FFT of a batch
 * Stores half of workgroup size equivalent number of consecutive batches to global memory.
 * Call site is resposible for managing OOB accesses.
 *  assumption: `nd_item.get_local_linear_id() / 2 < number_of_batches_in_local_mem
 *
 * @tparam GlobalT The type of view of global memory
 * @tparam LocalT The type of view of local memory
 *
 * @param global_data  global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::level Level, typename GlobalT, typename LocalT>
PORTFFT_INLINE void local_transposed2_global_transposed(detail::global_data_struct global_data, GlobalT global,
                                                        LocalT local, IdxGlobal offset, Idx num_complex,
                                                        IdxGlobal stride_global, Idx stride_local) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<LocalT>, detail::get_element_remove_cv_t<GlobalT>>,
                "Type mismatch between local and global views");
  global_data.log_message_local(__func__,
                                "Tranferring data from local to global memory with stride_global:", stride_global,
                                " global offset = ", offset, "number of elements per workitem = ", num_complex,
                                " and local stride:", stride_local);
  Idx local_id = static_cast<Idx>(global_data.get_group<Level>().get_local_id()[0]);
  for (Idx i = 0; i < num_complex; i++) {
    Idx local_index = 2 * i * stride_local + local_id;
    IdxGlobal global_index = offset + static_cast<IdxGlobal>(local_id) + static_cast<IdxGlobal>(2 * i) * stride_global;
    global_data.log_message(__func__, "from", local_index, "to", global_index, "value", local[local_index]);
    global[global_index] = local[local_index];
  }
}

/**
 * Transfers data from local memory (which is in contiguous layout) to global memory (which is in strided layout),
 * by interpreting the local memory as if strided. To be used specifically for workgroup FFTs, where input in PACKED but
 * output is BATCHED_INTERLEAVED
 *
 * @tparam GlobalT The type of view of global memory
 * @tparam LocalT The type of view of local memory
 *
 * @param global_data global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param global_stride Stride applicable to global memory
 * @param global_offset  Offset applicable to global memory
 * @param num_elements Total number of elements to be transferred
 * @param N Viewing num_elements as product of two factors, N being the first factor
 * @param M Viewing num_elements as product of two factors, M being the second factor
 */
template <typename GlobalT, typename LocalT>
PORTFFT_INLINE void localstrided_2global_strided(detail::global_data_struct global_data, GlobalT global, LocalT local,
                                                 IdxGlobal global_stride, IdxGlobal global_offset, Idx num_elements,
                                                 Idx N, Idx M) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<LocalT>, detail::get_element_remove_cv_t<GlobalT>>,
                "Type mismatch between local and global views");
  global_data.log_message_global(__func__, "transferring data with global_stride = ", global_stride,
                                 " global offset = ", global_offset);
  Idx start_index = static_cast<Idx>(global_data.it.get_local_linear_id());
  Idx index_stride = static_cast<Idx>(global_data.it.get_local_range(0));
  for (Idx idx = start_index; idx < num_elements; idx += index_stride) {
    Idx source_row = idx / N;
    Idx source_col = idx % N;
    Idx base_offset = 2 * source_col * M + 2 * source_row;
    IdxGlobal base_global_idx = static_cast<IdxGlobal>(idx) * global_stride + global_offset;
    global_data.log_message(__func__, "from (", base_offset, ",", base_offset, ") ", "to (", base_global_idx,
                            base_global_idx + 1, "values = (", local[base_offset], ",", local[base_offset + 1], ")");
    global[base_global_idx] = local[base_offset];
    global[base_global_idx + 1] = local[base_offset + 1];
  }
}

/**
 * Transfers data from local memory (which is in strided layout) to global memory (which is in strided layout),
 * by adding another stride to local memory. To be used specifically for workgroup FFTs, where input is
 * BATCHED_INTERLEAVED and output is BATCHED_INTERLEAVED as well.
 * Call site is resposible for managing OOB accesses
 *
 * @tparam GlobalT The type of view of global memory
 * @tparam LocalT The type of view of local memory
 *
 * @param global_data global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param global_stride Stride applicable to global memory
 * @param global_offset Offset applicable to global memory
 * @param local_stride Stride applicable to local memory
 * @param num_elements Total number of elements to be transferred per workitem
 * @param N Viewing num_elements as product of two factors, N being the first factor
 * @param M Viewing num_elements as product of two factors, M being the second factor
 */
template <typename GlobalT, typename LocalT>
PORTFFT_INLINE void local2strides_2global_strided(detail::global_data_struct global_data, GlobalT global, LocalT local,
                                                  IdxGlobal global_stride, IdxGlobal global_offset, Idx local_stride,
                                                  Idx num_elements, Idx N, Idx M) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<LocalT>, detail::get_element_remove_cv_t<GlobalT>>,
                "Type mismatch between local and global views");
  global_data.log_message_global(__func__, "transferring data with global_stride = ", global_stride,
                                 " global offset = ", global_offset, " local stride = ", local_stride);
  for (Idx idx = 0; idx < num_elements; idx++) {
    Idx local_stride_2 = (idx % N) * M + (idx / N);
    Idx base_offset = local_stride_2 * local_stride + static_cast<Idx>(global_data.it.get_local_id(0));
    IdxGlobal global_idx = static_cast<IdxGlobal>(idx) * global_stride + global_offset +
                           static_cast<IdxGlobal>(global_data.it.get_local_id(0));
    global_data.log_message(__func__, "from", base_offset, "to", global_idx, "value", local[base_offset]);
    global[global_idx] = local[base_offset];
  }
}

};  // namespace portfft

#endif
