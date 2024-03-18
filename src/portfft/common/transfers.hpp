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
 * There is no requirement that any of the arguments are the same between workitems in a workgroup/subgroup.
 *
 * @tparam VectorSize Size of the vector to copy - number of consecutive elements. Warning: if VectorSize > 1 is
 * used, elements and any strides in the views is done in vectors. Offsets in views are still in scalars. Also, if this
 * copy operates on padded local memory as either source or destination, it is assumed that padding never falls between
 * elements of a copied vector.
 * @tparam ViewSrc type of the source pointer or view
 * @tparam ViewDst type of the destination pointer or view
 * @param global_data global_data
 * @param src source pointer or view
 * @param dst destination pointer or view
 * @param size number of consecutive elements to copy (use views for strides). If VectorSize > 1, those elements are
 * vectors of that size.
 */
template <Idx VectorSize = 1, typename ViewSrc, typename ViewDst>
PORTFFT_INLINE void copy_wi(detail::global_data_struct<1> global_data, ViewSrc src, ViewDst dst, Idx size) {
  static_assert(!detail::is_view_multidimensional<ViewSrc>() && !detail::is_view_multidimensional<ViewDst>(),
                "This overload of copy_wi expects one-dimensional view arguments!");
  PORTFFT_UNROLL
  for (Idx i = 0; i < size; i++) {
    auto src_start = &src[i * VectorSize];
    auto dst_start = &dst[i * VectorSize];
    PORTFFT_UNROLL
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
 * Work is distributed between workitems in the group, so all workitems in the group must call the function and for each
 * call `src`, `dst` and `size` must have the same value for all workitems in the group.
 *
 * @tparam Level Which group is to jointly execute the copy; subgroup or workgroup
 * @tparam ViewSrc type of the source pointer or view
 * @tparam ViewDst type of the destination pointer or view
 * @param global_data global_data
 * @param src source pointer or view
 * @param dst destination pointer or view
 * @param size number of consecutive elements to copy (use views for strides)
 */
template <detail::level Level, typename ViewSrc, typename ViewDst>
PORTFFT_INLINE void copy_group(detail::global_data_struct<1> global_data, ViewSrc src, ViewDst dst, Idx size) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only subgroup and workgroup level supported");
  static_assert(!detail::is_view_multidimensional<ViewSrc>() && !detail::is_view_multidimensional<ViewDst>(),
                "This overload of copy_wi expects one-dimensional view arguments!");
  auto group = global_data.get_group<Level>();
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  Idx local_size = static_cast<Idx>(group.get_local_range()[0]);
  PORTFFT_UNROLL
  for (Idx i = local_id; i < size; i += local_size) {
    dst[i] = src[i];
    global_data.log_message(__func__, "from", &src[i] - detail::get_raw_pointer(src), "to",
                            &dst[i] - detail::get_raw_pointer(dst), "value", src[i]);
  }
}

/**
 * Copy multidimensional data.
 *
 * There is no requirement that any of the arguments are the same between workitems in a workgroup/subgroup.
 *
 * @tparam SrcParent type of the underlying pointer or view for source multidimensional view
 * @tparam DstParent type of the underlying pointer or view for destination multidimensional view
 * @tparam NDim number of dimensions
 * @param global_data global_data
 * @param src source multidimensional view
 * @param dst destination multidimensional view
 * @param sizes sizes (for each dimension) of the data to copy
 */
template <typename SrcParent, typename DstParent, std::size_t NDim>
PORTFFT_INLINE void copy_wi(detail::global_data_struct<1> global_data, detail::md_view<NDim, SrcParent, Idx, Idx> src,
                            detail::md_view<NDim, DstParent, Idx, Idx> dst, std::array<Idx, NDim> sizes) {
  if constexpr (NDim == 0) {
    global_data.log_message(__func__, "from", &src.get() - detail::get_raw_pointer(src), "to",
                            &dst.get() - detail::get_raw_pointer(dst), "value", src.get());
    dst.get() = src.get();
  } else {
    std::array<Idx, NDim - 1> next_sizes;
    PORTFFT_UNROLL
    for (std::size_t j = 0; j < NDim - 1; j++) {
      next_sizes[j] = sizes[j + 1];
    }
    PORTFFT_UNROLL
    for (Idx i = 0; i < sizes[0]; i++) {
      copy_wi<SrcParent, DstParent, NDim - 1>(src.inner(i), dst.inner(i), next_sizes);
    }
  }
}

/**
 * Copy multidimensional data jointly by a group. Work is distributed across workitems along the last two dimensions.
 *
 * Work is distributed between workitems in the group, so all workitems in the group must call the function and for each
 * call `src`, `dst` and `sizes` must have the same value for all workitems in the group.
 *
 * @tparam Level Which group is to jointly execute the copy; subgroup or workgroup
 * @tparam SrcParent type of the underlying pointer or view for source view
 * @tparam SrcStrides integral type used for strides in the source view
 * @tparam SrcOffset integral type for offset in the source view
 * @tparam DstParent type of the underlying pointer or view for destination view
 * @tparam DstStrides integral type used for strides in the destination view
 * @tparam DstOffset integral type for offset in the destination view
 * @tparam NDim number of dimensions
 * @param global_data global_data
 * @param src source multidimensional view
 * @param dst destination multidimensional view
 * @param sizes sizes (for each dimension) of the data to copy
 */
template <detail::level Level, typename SrcParent, typename SrcStrides, typename SrcOffset, typename DstParent,
          typename DstStrides, typename DstOffset, std::size_t NDim>
PORTFFT_INLINE void copy_group(detail::global_data_struct<1> global_data,
                               detail::md_view<NDim, SrcParent, SrcStrides, SrcOffset> src,
                               detail::md_view<NDim, DstParent, DstStrides, DstOffset> dst,
                               std::array<Idx, NDim> sizes) {
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only subgroup and workgroup level supported");
  auto group = global_data.get_group<Level>();
  Idx local_id = static_cast<Idx>(group.get_local_id()[0]);
  Idx local_size = static_cast<Idx>(group.get_local_range()[0]);
  if constexpr (NDim == 2) {
    PORTFFT_UNROLL
    for (Idx ij = local_id; ij < sizes[0] * sizes[1]; ij += local_size) {
      Idx i = ij / sizes[1];
      Idx j = ij % sizes[1];
      const auto& src_ref = src.inner(i).inner(j).get();
      auto& dst_ref = dst.inner(i).inner(j).get();
      global_data.log_message(__func__, "from", &src_ref - detail::get_raw_pointer(src), "to",
                              &dst_ref - detail::get_raw_pointer(dst), "value", src_ref);
      dst_ref = src_ref;
    }
  } else if constexpr (NDim == 1) {
    PORTFFT_UNROLL
    for (Idx i = local_id; i < sizes[0]; i += local_size) {
      const auto& src_ref = src.inner(i).get();
      auto& dst_ref = dst.inner(i).get();
      global_data.log_message(__func__, "from", &src_ref - detail::get_raw_pointer(src), "to",
                              &dst_ref - detail::get_raw_pointer(dst), "value", src_ref);
      dst_ref = src_ref;
    }
  } else {
    std::array<Idx, NDim - 1> next_sizes;
    PORTFFT_UNROLL
    for (std::size_t j = 0; j < NDim - 1; j++) {
      next_sizes[j] = sizes[j + 1];
    }
    PORTFFT_UNROLL
    for (Idx i = 0; i < sizes[0]; i++) {
      copy_group<Level, SrcParent, SrcStrides, SrcOffset, DstParent, DstStrides, DstOffset, NDim - 1>(
          global_data, src.inner(i), dst.inner(i), next_sizes);
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
PORTFFT_INLINE Idx subgroup_single_block_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
                                              IdxGlobal global_offset, LocalViewT local, Idx local_offset) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  constexpr Idx SgBlockCopyBlockSize = ChunkSize * SubgroupSize;
  using vec_t = sycl::vec<real_t, ChunkSize>;

  // Is the local memory suitable for using Intel's subgroup copy extensions with?
  const bool is_sg_contiguous = is_contiguous_view(local);
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
PORTFFT_INLINE Idx subgroup_block_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
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
PORTFFT_INLINE Idx vec_aligned_group_block_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
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
PORTFFT_INLINE void global_local_contiguous_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
                                                 LocalViewT local, Idx total_num_elems, IdxGlobal global_offset = 0,
                                                 Idx local_offset = 0) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  static_assert(std::is_floating_point_v<real_t>, "Expecting floating-point data type");
  static_assert(std::is_same_v<real_t, get_element_remove_cv_t<LocalViewT>>, "Type mismatch between global and local");
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset);
  static constexpr Idx ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(real_t);
  static constexpr int ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;

  using vec_t = sycl::vec<real_t, ChunkSize>;
  const real_t* global_ptr = &global[global_offset];
  const real_t* global_aligned_ptr = reinterpret_cast<const real_t*>(
      detail::round_up_to_multiple(reinterpret_cast<std::uintptr_t>(global_ptr), alignof(vec_t)));
  Idx unaligned_elements = static_cast<Idx>(global_aligned_ptr - global_ptr);
  // Load the first few unaligned elements.
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group<Level>(global_data, offset_view(global, global_offset), offset_view(local, local_offset),
                      unaligned_elements);
  } else {  // LOCAL_TO_GLOBAL
    copy_group<Level>(global_data, offset_view(local, local_offset), offset_view(global, global_offset),
                      unaligned_elements);
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;
  total_num_elems -= unaligned_elements;

#ifdef PORTFFT_USE_SG_TRANSFERS
  // Unaligned subgroup copies cause issues when writing to buffers in some circumstances for unknown reasons.
  Idx copied_by_sg = impl::subgroup_block_copy<TransferDirection, Level, ChunkSize, SubgroupSize>(
      global_data, global, global_offset, local, local_offset, total_num_elems);
  local_offset += copied_by_sg;
  global_offset += copied_by_sg;
  total_num_elems -= copied_by_sg;
#else
  // Each workitem loads a chunk of consecutive elements. Chunks loaded by a group are consecutive.
  Idx block_copied_elements = impl::vec_aligned_group_block_copy<TransferDirection, Level, ChunkSize>(
      global_data, global, global_offset, local, local_offset, total_num_elems);
  local_offset += block_copied_elements;
  global_offset += block_copied_elements;
  total_num_elems -= block_copied_elements;
#endif
  // We cannot load fixed-size blocks of data anymore, so we use naive copies.
  if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
    copy_group<Level>(global_data, offset_view(global, global_offset), offset_view(local, local_offset),
                      total_num_elems);
  } else {  // LOCAL_TO_GLOBAL
    copy_group<Level>(global_data, offset_view(local, local_offset), offset_view(global, global_offset),
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
PORTFFT_INLINE void global2local(detail::global_data_struct<1> global_data, GlobalViewT global, LocalViewT local,
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
PORTFFT_INLINE void local2global(detail::global_data_struct<1> global_data, LocalViewT local, GlobalViewT global,
                                 Idx total_num_elems, Idx local_offset = 0, IdxGlobal global_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::LOCAL_TO_GLOBAL, Level, SubgroupSize>(
      global_data, global, local, total_num_elems, global_offset, local_offset);
}

/**
 * Driver function to copy data between local and global memory when data is in PACKED format in both, local
 * as well as global memory, when the storage scheme is INTERLEAVED_COMPLEX
 *
 * @tparam Group Group level taking part in the copy, should be one of level::SUBGROUP or level::WORKGROUP
 * @tparam Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam SubgroupSize Subgroup Size
 * @tparam LocView Type of view of the local memory
 * @tparam T Scalar Type
 * @param global_ptr Pointer to the input / output global memory
 * @param loc_view Local memory view containing the input
 * @param global_offset Offset to be applied to the input / output pointer
 * @param local_offset Offset to be applied to local memory view
 * @param n_elements_to_copy Number of scalar elements to copy
 * @param global_data global_data_struct associated with the kernel launch
 */
template <detail::level Group, detail::transfer_direction Direction, Idx SubgroupSize, typename LocView, typename T>
PORTFFT_INLINE void local_global_packed_copy(T* global_ptr, LocView& loc_view, IdxGlobal global_offset,
                                             Idx local_offset, Idx n_elements_to_copy,
                                             detail::global_data_struct<1>& global_data) {
  global_data.log_message(__func__, "storage scheme: INTERLEAVED_COMPLEX");
  if constexpr (Direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    global_data.log_message(__func__,
                            "Transferring from global to local memory, number of elements: ", n_elements_to_copy,
                            " global offset: ", global_offset, " local_offset: ", local_offset);
    global2local<Group, SubgroupSize>(global_data, global_ptr, loc_view, n_elements_to_copy, global_offset,
                                      local_offset);
  } else {
    global_data.log_message(__func__,
                            "Transferring from global to local memory, number of elements: ", n_elements_to_copy,
                            " global offset: ", global_offset, " local_offset: ", local_offset);
    local2global<Group, SubgroupSize>(global_data, loc_view, global_ptr, n_elements_to_copy, local_offset,
                                      global_offset);
  }
}

/**
 * Driver function to copy data between local and global memory when data is in PACKED format in both, local
 * as well as global memory, when the storage scheme is SPLIT_COMPLEX
 *
 * @tparam Group Group level taking part in the copy, should be one of level::SUBGROUP or level::WORKGROUP
 * @tparam Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam SubgroupSize Subgroup Size
 * @tparam LocView Type of view of the local memory
 * @tparam T Scalar Type
 * @param global_ptr Pointer to the input / output global memory containing the real part of the data
 * @param global_imag_ptr ointer to the input / output global memory containing the imaginary part of the data
 * @param loc_view Local memory view containing the input
 * @param global_offset Offset to be applied to the input / output pointer
 * @param local_offset Offset to be applied to local memory view
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored.
 * @param n_elements_to_copy Number of scalar elements to copy
 * @param global_data global_data_struct associated with the kernel launch
 */
template <detail::level Group, detail::transfer_direction Direction, Idx SubgroupSize, typename LocView, typename T>
PORTFFT_INLINE void local_global_packed_copy(T* global_ptr, T* global_imag_ptr, LocView& loc_view,
                                             IdxGlobal global_offset, Idx local_offset, Idx local_imag_offset,
                                             Idx n_elements_to_copy, detail::global_data_struct<1>& global_data) {
  global_data.log_message(__func__, "storage scheme: SPLIT_COMPLEX");
  if constexpr (Direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    global_data.log_message(__func__,
                            "Transferring from global to local memory, number of elements: ", n_elements_to_copy,
                            " global offset: ", global_offset, " local_offset: ", local_offset);
    global2local<Group, SubgroupSize>(global_data, global_ptr, loc_view, n_elements_to_copy, global_offset,
                                      local_offset);
    global2local<Group, SubgroupSize>(global_data, global_imag_ptr, loc_view, n_elements_to_copy, global_offset,
                                      local_offset + local_imag_offset);
  } else {
    global_data.log_message(__func__,
                            "Transferring from global to local memory, number of elements: ", n_elements_to_copy,
                            " global offset: ", global_offset, " local_offset: ", local_offset);
    local2global<Group, SubgroupSize>(global_data, loc_view, global_ptr, n_elements_to_copy, local_offset,
                                      global_offset);
    local2global<Group, SubgroupSize>(global_data, loc_view, global_imag_ptr, n_elements_to_copy,
                                      local_offset + local_imag_offset, global_offset);
  }
}

/**
 * Driver function for copying data between local and global memory when the data layout is arbitrarily
 * strided in either or both, local and global memory, when the storage scheme is INTERLEAVED_COMPLEX
 *
 * @tparam Group Group level taking part in the copy, should be one of level::SUBGROUP or level::WORKGROUP
 * @tparam Direction Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam GlobalDim Number of dimension of the md_view to be created for the global memory
 * @tparam LocalDim Number of dimension of the md_view to be created for the global memory
 * @tparam CopyDims Number of dimensions over which the data will be copied
 * @tparam T Scalar Type
 * @tparam LocView Type of view created for the local memory
 * @param global_ptr Pointer to the input / output global memory
 * @param loc_view Local memory view containing the input
 * @param strides_global An array specifying the strides for the global memory
 * @param strides_local An array specifying the strides for the local memory
 * @param offset_global Offset value to be applied to the global memory
 * @param offset_local Offset value to be applied to the local memory
 * @param copy_lengths number of scalars (for each dimension) of the data to copy
 * @param global_data global_data_struct associated with the kernel launch
 */
template <detail::level Group, detail::transfer_direction Direction, std::size_t GlobalDim, std::size_t LocalDim,
          std::size_t CopyDims, typename T, typename LocView>
PORTFFT_INLINE void local_global_strided_copy(T* global_ptr, LocView& loc_view,
                                              std::array<IdxGlobal, GlobalDim> strides_global,
                                              std::array<Idx, LocalDim> strides_local, IdxGlobal offset_global,
                                              Idx offset_local, std::array<Idx, CopyDims> copy_lengths,
                                              detail::global_data_struct<1> global_data) {
  global_data.log_message(__func__, "storage scheme: INTERLEAVED_COMPLEX");
  detail::md_view global_md_view{global_ptr, strides_global, offset_global};
  detail::md_view local_md_view{loc_view, strides_local, offset_local};
  if constexpr (Direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    global_data.log_message(__func__, "transferring strided data from global to local memory");
    copy_group<Group>(global_data, global_md_view, local_md_view, copy_lengths);
  } else {
    global_data.log_message(__func__, "transferring strided data from local to global memory");
    copy_group<Group>(global_data, local_md_view, global_md_view, copy_lengths);
  }
}

/**
 * Driver function for copying data between local and global memory when the data layout is arbitrarily
 * strided in either or both, local and global memory, when the storage scheme is SPLIT_COMPLEX
 *
 * @tparam Group Group level taking part in the copy, should be one of level::SUBGROUP or level::WORKGROUP
 * @tparam Direction Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam GlobalDim Number of dimension of the md_view to be created for the global memory
 * @tparam LocalDim Number of dimension of the md_view to be created for the global memory
 * @tparam CopyDims Number of dimensions over which the data will be copied
 * @tparam T Scalar Type
 * @tparam LocView Type of view created for the local memory
 * @param global_ptr Pointer to the input / output global memory containing the real part of the data
 * @param global_imag_ptr ointer to the input / output global memory containing the imaginary part of the data
 * @param loc_view View of the local memory
 * @param strides_global An array specifying the strides for the global memory
 * @param strides_local An array specifying the strides for the local memory
 * @param offset_global  Offset value to be applied to the global memory
 * @param local_offset Offset value to be applied to the local memory
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param copy_lengths number of scalars (for each dimension) of the data to copy
 * @param global_data global_data_struct associated with the kernel launch
 */
template <detail::level Group, detail::transfer_direction Direction, std::size_t GlobalDim, std::size_t LocalDim,
          std::size_t CopyDims, typename T, typename LocView>
PORTFFT_INLINE void local_global_strided_copy(T* global_ptr, T* global_imag_ptr, LocView& loc_view,
                                              std::array<IdxGlobal, GlobalDim> strides_global,
                                              std::array<Idx, LocalDim> strides_local, IdxGlobal offset_global,
                                              Idx local_offset, Idx local_imag_offset,
                                              std::array<Idx, CopyDims> copy_lengths,
                                              detail::global_data_struct<1> global_data) {
  global_data.log_message(__func__, "storage scheme: SPLIT_COMPLEX");
  detail::md_view global_md_real_view{global_ptr, strides_global, offset_global};
  detail::md_view global_md_imag_view{global_imag_ptr, strides_global, offset_global};
  detail::md_view local_md_real_view{loc_view, strides_local, local_offset};
  detail::md_view local_md_imag_view{loc_view, strides_local, local_offset + local_imag_offset};
  if constexpr (Direction == detail::transfer_direction::GLOBAL_TO_LOCAL) {
    global_data.log_message(__func__, "transferring strided data from global to local memory");
    copy_group<Group>(global_data, global_md_real_view, local_md_real_view, copy_lengths);
    copy_group<Group>(global_data, global_md_imag_view, local_md_imag_view, copy_lengths);
  } else {
    global_data.log_message(__func__, "transferring strided data from local to global memory");
    copy_group<Group>(global_data, local_md_real_view, global_md_real_view, copy_lengths);
    copy_group<Group>(global_data, local_md_imag_view, global_md_imag_view, copy_lengths);
  }
}

/**
 * Driver function for copying data between local and private memory when the storage scheme is INTERLEAVED_COMPLEX
 * This can also be used when directly copying data from private memory to global memory
 *
 * @tparam PtrViewNDim Number of Dimension of the local / global memory view
 * @tparam IdxType Integer type of the strides and offset of the local / global memory
 * @tparam PtrView View type of the local / global memory
 * @tparam T Scalar Type
 * @param ptr_view View of the local / global memory taking part in the copy
 * @param priv Pointer to the private memory array
 * @param ptr_view_strides_offsets An array of 2 arrays containing PtrViewNDim elements of IdxType, containing strides
 * and offsets for the strided view to be constructed for the local / global memory
 * @param num_elements_to_copy Number of scalar elements to copy
 * @param direction direction of copy, should be one of LOCAL_TO_PRIVATE, PRIVATE_TO_LOCAL or PRIVATE_TO_GLOBAL
 * @param global_data global data struct associated with the kernel launch
 */
template <Idx PtrViewNDim, typename IdxType, typename PtrView, typename T>
PORTFFT_INLINE void local_private_strided_copy(PtrView& ptr_view, T* priv,
                                               stride_offset_struct<IdxType, PtrViewNDim> ptr_view_strides_offsets,
                                               Idx num_elements_to_copy, detail::transfer_direction direction,
                                               detail::global_data_struct<1> global_data) {
  global_data.log_message(__func__, "storage scheme: INTERLEAVED_COMPLEX");
  detail::strided_view ptr_strided_view{ptr_view, ptr_view_strides_offsets.strides, ptr_view_strides_offsets.offsets};
  if (direction == detail::transfer_direction::LOCAL_TO_PRIVATE) {
    copy_wi<2>(global_data, ptr_strided_view, priv, num_elements_to_copy);
  } else if (direction == detail::transfer_direction::PRIVATE_TO_LOCAL ||
             direction == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
    copy_wi<2>(global_data, priv, ptr_strided_view, num_elements_to_copy);
  }
}

/**
 * Driver function for copying data between local and private memory when the storage scheme is SPLIT_COMPLEX
 * This can also be used when directly copying data from private memory to global memory
 *
 * @tparam PtrViewNDim Number of Dimension of the local / global memory view
 * @tparam IdxType Integer type of the strides and offset of the local / global memory
 * @tparam PtrView View type of the local / global memory
 * @tparam T Scalar Type
 * @param ptr_view View of the local / global memory containing the real component of the data
 * @param ptr_imag_view View of the local / global memory containing the imaginary component of the data
 * @param priv Pointer to the private memory array
 * @param ptr_view_strides_offsets Struct containing strides
 * and offsets for the strided view to be constructed for the local / global memory containing the real part of the data
 * @param ptr_imag_view_strides_offsets Struct containing
 * strides and offsets for the strided view to be constructed for the local / global memory containing the imaginary
 * part of the data
 * @param num_elements_to_copy Number of elements to copy
 * @param direction direction of copy, should be one of LOCAL_TO_PRIVATE, PRIVATE_TO_LOCAL or PRIVATE_TO_GLOBAL
 * @param global_data global data struct associated with the kernel launch
 */
template <Idx PtrViewNDim, typename IdxType, typename PtrView, typename T>
PORTFFT_INLINE void local_private_strided_copy(PtrView& ptr_view, PtrView& ptr_imag_view, T* priv,
                                               stride_offset_struct<IdxType, PtrViewNDim> ptr_view_strides_offsets,
                                               stride_offset_struct<IdxType, PtrViewNDim> ptr_imag_view_strides_offsets,
                                               Idx num_elements_to_copy, detail::transfer_direction direction,
                                               detail::global_data_struct<1> global_data) {
  global_data.log_message(__func__, "storage scheme: INTERLEAVED_COMPLEX");
  detail::strided_view ptr_strided_real_view{ptr_view, ptr_view_strides_offsets.strides,
                                             ptr_view_strides_offsets.offsets};
  detail::strided_view ptr_strided_imag_view{ptr_imag_view, ptr_imag_view_strides_offsets.strides,
                                             ptr_imag_view_strides_offsets.offsets};
  detail::strided_view priv_strided_real_view{priv, 2};
  detail::strided_view priv_strided_imag_view{priv, 2, 1};
  if (direction == detail::transfer_direction::LOCAL_TO_PRIVATE) {
    copy_wi(global_data, ptr_strided_real_view, priv_strided_real_view, num_elements_to_copy);
    copy_wi(global_data, ptr_strided_imag_view, priv_strided_imag_view, num_elements_to_copy);
  } else if (direction == detail::transfer_direction::PRIVATE_TO_LOCAL ||
             direction == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
    copy_wi(global_data, priv_strided_real_view, ptr_strided_real_view, num_elements_to_copy);
    copy_wi(global_data, priv_strided_imag_view, ptr_strided_imag_view, num_elements_to_copy);
  }
}

}  // namespace portfft

#endif
