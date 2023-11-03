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

#include <sycl/sycl.hpp>

#include "helpers.hpp"
#include "logging.hpp"
#include "memory_views.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "portfft/traits.hpp"

namespace portfft {

/**
 * Copy data. Each workitem does the copy independently.
 *
 * @tparam VectorSize Size of the vector to copy - number of consecutive elements. Warning: even if VectorSize > 1 is
 * used elements and any indexing in the views is done in scalars, not vectors!
 * @tparam View1 type of the source pointer or view
 * @tparam View2 type of the destination pointer or view
 * @param global_data global_data
 * @param src source pointer or view
 * @param dst destination pointer or view
 * @param size number of consecutive elements to copy (use views for strides)
 */
template <int VectorSize = 1, typename View1, typename View2>
PORTFFT_INLINE void copy_wi(detail::global_data_struct global_data, View1 src, View2 dst, Idx size) {
  using Scalar = detail::get_element_t<View2>;
#pragma clang loop unroll(full)
  for (Idx i = 0; i < size; i++) {
    const Scalar* src_start = &src[i];
    Scalar* dst_start = &dst[i];
#pragma clang loop unroll(full)
    for (Idx j = 0; j < VectorSize; j++) {
      global_data.log_message(__func__, "from", &src_start[j] - detail::get_raw_pointer(src), "to",
                              &dst_start[j] - detail::get_raw_pointer(dst), "value", src_start[j]);
      dst_start[j] = src_start[j];
    }
  }
}

/**
 * Copy data jointly by workitems in a group.
 *
 * @tparam View1 type of the source pointer or view
 * @tparam View2 type of the destination pointer or view
 * @param global_data global_data
 * @param group_size size of the group
 * @param local_id id of workitem in the group
 * @param src source pointer or view
 * @param dst destination pointer or view
 * @param size number of consecutive elements to copy (use views for strides)
 */
template <typename View1, typename View2>
PORTFFT_INLINE void copy_group(detail::global_data_struct global_data, Idx group_size, Idx local_id, View1 src,
                               View2 dst, Idx size) {
#pragma clang loop unroll(full)
  for (Idx i = local_id; i < size; i += group_size) {
    dst[i] = src[i];
    global_data.log_message(__func__, "from", &src[i] - detail::get_raw_pointer(src), "to",
                            &dst[i] - detail::get_raw_pointer(dst), "value", src[i]);
  }
}

/**
 * Copy multidimensional data.
 *
 * @tparam TParent1 type of the underlying pointer or view for source multidimensional view
 * @tparam TParent2 type of the underlying pointer or view for destination multidimensional view
 * @tparam NDim number of dimensions
 * @param global_data global_data
 * @param src source multidimensional view
 * @param dst destination multidimensional view
 * @param sizes sizes (for each dimension) of the data to copy
 */
template <typename TParent1, typename TParent2, std::size_t NDim>
PORTFFT_INLINE void copy_wi(detail::global_data_struct global_data, detail::md_view<NDim, TParent1, Idx, Idx> src,
                            detail::md_view<NDim, TParent2, Idx, Idx> dst, std::array<Idx, NDim> sizes) {
  if constexpr (NDim == 0) {
    global_data.log_message(__func__, "from", &src.get() - detail::get_raw_pointer(src), "to",
                            &dst.get() - detail::get_raw_pointer(dst), "value", src.get());
    dst.get() = src.get();
  } else {
    std::array<Idx, NDim - 1> next_sizes;
#pragma clang loop unroll(full)
    for (std::size_t j = 0; j < NDim - 1; j++) {
      next_sizes[j] = sizes[j + 1];
    }
#pragma clang loop unroll(full)
    for (Idx i = 0; i < sizes[0]; i++) {
      copy_wi<TParent1, TParent2, NDim - 1>(src.inner(i), dst.inner(i), next_sizes);
    }
  }
}

/**
 * Copy multidimensional data jointly by a group. Work is distributed across workitems along the last two dimensions.
 *
 * @tparam TParent1 type of the underlying pointer or view for source view
 * @tparam TStrides1 integral type used for strides in the source view
 * @tparam TOffset1 integral type for offset in the source view
 * @tparam TParent2 type of the underlying pointer or view for destination view
 * @tparam TStrides2 integral type used for strides in the destination view
 * @tparam TOffset2 integral type for offset in the destination view
 * @tparam NDim number of dimensions
 * @param global_data global_data
 * @param group_size size of the group
 * @param local_id id of workitem in the group
 * @param src source multidimensional view
 * @param dst destination multidimensional view
 * @param sizes sizes (for each dimension) of the data to copy
 */
template <typename TParent1, typename TStrides1, typename TOffset1, typename TParent2, typename TStrides2,
          typename TOffset2, std::size_t NDim>
PORTFFT_INLINE void copy_group(detail::global_data_struct global_data, Idx group_size, Idx local_id,
                               detail::md_view<NDim, TParent1, TStrides1, TOffset1> src,
                               detail::md_view<NDim, TParent2, TStrides2, TOffset2> dst, std::array<Idx, NDim> sizes) {
  if constexpr (NDim == 2) {
#pragma clang loop unroll(full)
    for (Idx ij = local_id; ij < sizes[0] * sizes[1]; ij += group_size) {
      Idx i = ij / sizes[1];
      Idx j = ij % sizes[1];
      const auto& src_ref = src.inner(i).inner(j).get();
      auto& dst_ref = dst.inner(i).inner(j).get();
      global_data.log_message(__func__, "from", &src_ref - detail::get_raw_pointer(src), "to",
                              &dst_ref - detail::get_raw_pointer(dst), "value", src_ref);
      dst_ref = src_ref;
    }
  } else if constexpr (NDim == 1) {
#pragma clang loop unroll(full)
    for (Idx i = local_id; i < sizes[0]; i += group_size) {
      const auto& src_ref = src.inner(i).get();
      auto& dst_ref = dst.inner(i).get();
      global_data.log_message(__func__, "from", &src_ref - detail::get_raw_pointer(src), "to",
                              &dst_ref - detail::get_raw_pointer(dst), "value", src_ref);
      dst_ref = src_ref;
    }
  } else {
    std::array<Idx, NDim - 1> next_sizes;
#pragma clang loop unroll(full)
    for (std::size_t j = 0; j < NDim - 1; j++) {
      next_sizes[j] = sizes[j + 1];
    }
#pragma clang loop unroll(full)
    for (Idx i = 0; i < sizes[0]; i++) {
      copy_group<TParent1, TStrides1, TOffset1, TParent2, TStrides2, TOffset2, NDim - 1>(
          global_data, group_size, local_id, src.inner(i), dst.inner(i), next_sizes);
    }
  }
}

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
  const bool is_sg_contiguous = PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || is_contiguous_view(local);
  const char* func_name = __func__;
  global_data.log_message_subgroup(func_name, "SgBlockCopyBlockSize", SgBlockCopyBlockSize, "global_offset",
                                   global_offset, "local_offset", local_offset, "is_sg_contiguous", is_sg_contiguous);
  Idx local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  // A helper function to generate indexes in local memory.
  auto index_transform = [=](Idx i) PORTFFT_INLINE { return local_offset + i * SubgroupSize + local_id; };
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    vec_t vec = global_data.sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset]));
    PORTFFT_UNROLL
    for (Idx j = 0; j < ChunkSize; j++) {
      if (is_sg_contiguous) {
        global_data.sg.store(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]),
                             vec[static_cast<int>(j)]);
      } else {
        local[index_transform(j)] = vec[static_cast<int>(j)];
      }
    }
  } else {
    vec_t vec;
    PORTFFT_UNROLL
    for (Idx j = 0; j < ChunkSize; j++) {
      if (is_sg_contiguous) {
        vec[static_cast<int>(j)] =
            global_data.sg.load(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]));
      } else {
        vec[static_cast<int>(j)] = local[index_transform(j)];
      }
    }
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
      }
    } else {  // LOCAL_TO_GLOBAL
      vec_t to_store;
      PORTFFT_UNROLL
      for (Idx j = 0; j < ChunkSize; j++) {
        to_store[static_cast<int>(j)] = local[index_transform(j, loop_idx)];
      }
      *reinterpret_cast<vec_t*>(&global[global_offset + wi_offset + block_size * loop_idx]) = to_store;
    }
  }
  global_data.log_message_scoped<Level>(func_name, "copied_value_count", block_count * block_size);
  return block_count * block_size;
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

  auto group = global_data.get_group<Level>();
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  Idx local_size = static_cast<Idx>(group.get_local_range()[0]);

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
  // Load the first few unaligned elements.
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group(global_data, local_size, local_id, offset_view(global, global_offset), offset_view(local, local_offset),
               unaligned_elements);
  } else {  // LOCAL_TO_GLOBAL
    copy_group(global_data, local_size, local_id, offset_view(local, local_offset), offset_view(global, global_offset),
               unaligned_elements);
  }
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
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group(global_data, local_size, local_id, offset_view(global, global_offset), offset_view(local, local_offset),
               total_num_elems);
  } else {  // LOCAL_TO_GLOBAL
    copy_group(global_data, local_size, local_id, offset_view(local, local_offset), offset_view(global, global_offset),
               total_num_elems);
  }
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

}  // namespace portfft

#endif
