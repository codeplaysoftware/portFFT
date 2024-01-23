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
#include "portfft/common/compiletime_tuning_profile.hpp"
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
template <transfer_direction TransferDirection, level Level, ct_profile Config, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE Idx subgroup_block_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
                                       IdxGlobal global_offset, LocalViewT local, Idx local_offset, Idx n) {
  using profile_t = kernel_spec<Config>;
  static_assert(profile_t::UseSgTransfers);
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  static constexpr Idx SubgroupSize = profile_t::SgSize;
  static constexpr Idx ChunkSize = vec_load_elements<real_t, profile_t>();
  static constexpr Idx BlockSize = ChunkSize * SubgroupSize;
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
 * @tparam Config The compile-time kernel configuration
 * @tparam LocalViewT The type of the local memory view
 * @tparam GlobalViewT The type of the global memory view
 * @param global_data global data for the kernel
 * @param global View of global memory
 * @param local View of local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <transfer_direction TransferDirection, level Level, ct_profile Config, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE void global_local_contiguous_copy(detail::global_data_struct<1> global_data, GlobalViewT global,
                                                 LocalViewT local, Idx total_num_elems, IdxGlobal global_offset = 0,
                                                 Idx local_offset = 0) {
  using real_t = get_element_remove_cv_t<GlobalViewT>;
  static_assert(std::is_floating_point_v<real_t>, "Expecting floating-point data type");
  static_assert(std::is_same_v<real_t, get_element_remove_cv_t<LocalViewT>>, "Type mismatch between global and local");
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset);
  using kernel_config_t = kernel_spec<Config>;
  static constexpr int ChunkSize = vec_load_elements<real_t, kernel_config_t>();

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

  if constexpr (kernel_config_t::UseSgTransfers) {
    // Unaligned subgroup copies cause issues when writing to buffers in some circumstances for unknown reasons.
    Idx copied_by_sg = impl::subgroup_block_copy<TransferDirection, Level, Config>(
        global_data, global, global_offset, local, local_offset, total_num_elems);
    local_offset += copied_by_sg;
    global_offset += copied_by_sg;
    total_num_elems -= copied_by_sg;
  } else {
    // Each workitem loads a chunk of consecutive elements. Chunks loaded by a group are consecutive.
    Idx block_copied_elements = impl::vec_aligned_group_block_copy<TransferDirection, Level, ChunkSize>(
        global_data, global, global_offset, local, local_offset, total_num_elems);
    local_offset += block_copied_elements;
    global_offset += block_copied_elements;
    total_num_elems -= block_copied_elements;
  }
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
 * @tparam Config The compile-time kernel configuration
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
template <detail::level Level, detail::ct_profile Config, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE void global2local(detail::global_data_struct<1> global_data, GlobalViewT global, LocalViewT local,
                                 Idx total_num_elems, IdxGlobal global_offset = 0, Idx local_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::GLOBAL_TO_LOCAL, Level, Config>(
      global_data, global, local, total_num_elems, global_offset, local_offset);
}

/**
 * Copies data from local memory to global memory.
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam Config The compile-time kernel configuration
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
template <detail::level Level, detail::ct_profile Config, typename LocalViewT, typename GlobalViewT>
PORTFFT_INLINE void local2global(detail::global_data_struct<1> global_data, LocalViewT local, GlobalViewT global,
                                 Idx total_num_elems, Idx local_offset = 0, IdxGlobal global_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::LOCAL_TO_GLOBAL, Level, Config>(
      global_data, global, local, total_num_elems, global_offset, local_offset);
}

}  // namespace portfft

#endif
