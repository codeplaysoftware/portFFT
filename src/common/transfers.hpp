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

/** Helper class for copying between global and local memory.
 *
 * @tparam TransferDirection Direction of memory transfer
 * @tparam Level Is this a sub-group or work-group operation?
 * @tparam RealT The floating-point type to transfer
 * @tparam SubgroupSize The subgroup size
 * @tparam Pad Do or don't pad local memory
 * @tparam BankLinesPerPad Paramater for local memory padding
 */
template <transfer_direction TransferDirection, level Level, typename RealT, std::size_t SubgroupSize, pad Pad,
          std::size_t BankLinesPerPad>
class global_local_copy_helper {
 private:
  static constexpr std::size_t ChunkSizeRaw = PORTFFT_VEC_LOAD_BYTES / sizeof(RealT);
  static_assert(TransferDirection == transfer_direction::LOCAL_TO_GLOBAL ||
                TransferDirection == transfer_direction::GLOBAL_TO_LOCAL);
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  static_assert(std::is_same_v<std::remove_cv_t<RealT>, RealT>, "RealT should not be const or volatile qualified");

  /// Pad local with baked-in arguments.
  static PORTFFT_INLINE std::size_t padder(Idx local_idx) { return pad_local<Pad>(local_idx, BankLinesPerPad); }

 public:
  using real_type = RealT;
  // The number of reals to copy at once in a SYCL::vec<RealT, ChunkSize>
  static constexpr Idx ChunkSize = ChunkSizeRaw < 1 ? 1 : ChunkSizeRaw;
  // The sycl::vec type used for copying.
  using vec_t = sycl::vec<real_type, ChunkSize>;
  // The block size used for fast block copying with sub-groups.
  static constexpr Idx SgBlockCopyBlockSize = ChunkSize * SubgroupSize;
  // The group type associated with the Level
  using copy_group_t = std::conditional_t<Level == level::SUBGROUP, sycl::sub_group, sycl::group<1>>;

  /** Copy between index-contiguous global and local memory using sub-group loads/stores for real data.
   *  Works on a fixed size block. Data does not need to be aligned to alignof(vec_t). Arguments are
   *  expected to be the same for all values in the sub-group.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, SgBlockCopyBlockSize)
   *
   *  @tparam GlobalViewT The type of the global memory view
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @returns The number of reals copied
   */
  template <typename GlobalViewT, typename LocalViewT>
  static PORTFFT_INLINE Idx sg_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                          IdxGlobal global_offset, LocalViewT local, Idx local_offset) {
    // Is the local memory suitable for using Intel's subgroup copy extensions with?
    constexpr bool IsSgContiguous = PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || Pad == pad::DONT_PAD;
    const char* func_name = __func__;
    global_data.log_message_subgroup(func_name, "SgBlockCopyBlockSize", SgBlockCopyBlockSize, "global_offset",
                                     global_offset, "local_offset", local_offset, "IsSgContiguous", IsSgContiguous);
    Idx local_id = global_data.sg.get_local_linear_id();
    // A helper function to generate indexes in local memory.
    auto indexer = [=](Idx i) __attribute__((always_inline)) {
      return padder(local_offset + i * SubgroupSize + local_id);
    };
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      vec_t vec = global_data.sg.load<ChunkSize>(detail::get_global_multi_ptr(&global[global_offset]));
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) __attribute__((always_inline)) {
        if constexpr (IsSgContiguous) {
          global_data.sg.store(detail::get_local_multi_ptr(&local[padder(local_offset + j * SubgroupSize)]),
                               vec[static_cast<int>(j)]);
        } else {
          local[indexer(j)] = vec[static_cast<int>(j)];
        }
      });
    } else {
      vec_t vec;
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) __attribute__((always_inline)) {
        if constexpr (IsSgContiguous) {
          vec[static_cast<int>(j)] =
              global_data.sg.load(detail::get_local_multi_ptr(&local[padder(local_offset + j * SubgroupSize)]));
        } else {
          vec[static_cast<int>(j)] = local[indexer(j)];
        }
      });
      global_data.sg.store(detail::get_global_multi_ptr(&global[global_offset]), vec);
    }
    return SgBlockCopyBlockSize;
  }

  /** Copy between index-contiguous global and local memory using sub-groups loads/stores for real data.
   *  Data does not need to be aligned to alignof(vec_t). Arguments are expected to be the same for all values in the
   *  sub-group. Copies in multiples of SgBlockCopyBlockSize, and may copy less than the given item count n.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, m) where m <= n
   *
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param group The work-group of sub-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @param n The count of reals to copy
   *  @returns The number of reals copied. May be less than n.
   */
  template <typename GlobalViewT, typename LocalViewT>
  static PORTFFT_INLINE IdxGlobal sg_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                                IdxGlobal global_offset, LocalViewT local, Idx local_offset,
                                                IdxGlobal n) {
    const char* func_name = __func__;
    global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset, "n",
                                          n);
    if constexpr (Level == level::SUBGROUP) {
      static constexpr Idx BlockSize = SgBlockCopyBlockSize;
      Idx block_count = n / BlockSize;
      for (Idx block_idx{0}; block_idx < block_count; ++block_idx) {
        Idx offset = block_idx * BlockSize;
        sg_block_copy(global_data, global, global_offset + offset, local, local_offset + offset);
      }
      global_data.log_message_scoped<Level>(func_name, "copied_value_count", block_count * BlockSize);
      return block_count * BlockSize;
    } else {
      auto sg = global_data.sg;
      static constexpr Idx BlockSize = SgBlockCopyBlockSize;
      Idx subgroup_id = sg.get_group_id();
      Idx subgroup_count = sg.get_group_linear_range();
      Idx block_count = n / BlockSize;
      // NB: For work-groups this may lead to divergence between sub-groups on the final loop iteration.
      for (Idx block_idx{subgroup_id}; block_idx < block_count; block_idx += subgroup_count) {
        Idx offset = block_idx * BlockSize;
        sg_block_copy(global_data, global, global_offset + offset, local, local_offset + offset);
      }
      global_data.log_message_scoped<Level>(func_name, "copied_value_count", block_count * BlockSize);
      return block_count * BlockSize;
    }
  }

  /** Copy between index-contiguous global and local memory using work-group or sub-group. Global memory argument must
   * be aligned to alignof(vec_t). global[global_offset + i] <-> local[local_offset + i] for i in [0, ChunkSize *
   * group.get_local_range(0))
   *
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param group The sub-group or work-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @returns The number of reals copied
   */
  template <typename GlobalViewT, typename LocalViewT>
  static PORTFFT_INLINE IdxGlobal vec_aligned_block_copy(detail::global_data_struct global_data, copy_group_t group,
                                                         GlobalViewT global, IdxGlobal global_offset, LocalViewT local,
                                                         Idx local_offset) {
    const char* func_name = __func__;
    global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset,
                                          "copy_block_size", ChunkSize * group.get_local_range()[0]);
    Idx local_id = group.get_local_id()[0];
    Idx wi_offset = local_id * ChunkSize;
    auto indexer = [=](Idx i) PORTFFT_INLINE { return padder(local_offset + wi_offset + i); };
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      vec_t loaded;
      loaded = *reinterpret_cast<const vec_t*>(&global[global_offset + wi_offset]);
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) __attribute__((always_inline)) {
        local[indexer(j)] = loaded[static_cast<int>(j)];
      });
    } else {
      vec_t to_store;
      detail::unrolled_loop<0, ChunkSize, 1>([&](Idx j) __attribute__((always_inline)) {
        to_store[static_cast<int>(j)] = local[indexer(j)];
      });
      *reinterpret_cast<vec_t*>(&global[global_offset + wi_offset]) = to_store;
    }
    return ChunkSize * group.get_local_range()[0];
  }

  /** Copy between index-contiguous global and local memory for n values, where n < group.get_local_range()[0].
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
   *
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param group The sub-group or work-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @param n The number of reals to copy. Must be less than group.get_local_range()[0]
   *  @returns The number of reals copied
   */
  template <typename GlobalViewT, typename LocalViewT>
  static PORTFFT_INLINE IdxGlobal subrange_copy(detail::global_data_struct global_data, copy_group_t group,
                                                GlobalViewT global, IdxGlobal global_offset, LocalViewT local,
                                                Idx local_offset, IdxGlobal n) {
    const char* func_name = __func__;
    global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset,
                                          "group_size", group.get_local_range()[0], "n", n);
    Idx local_id = group.get_local_id()[0];
    if (local_id < n) {
      if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
        local[padder(local_offset + local_id)] = global[global_offset + local_id];
      } else {
        global[global_offset + local_id] = local[padder(local_offset + local_id)];
      }
    }
    return n;
  }

  /** Copy between index-contiguous global and local memory for n values. No particular requirements for alignment.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
   *
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
  template <typename GlobalViewT, typename LocalViewT>
  static PORTFFT_INLINE IdxGlobal naive_copy(detail::global_data_struct global_data, copy_group_t group,
                                             GlobalViewT global, IdxGlobal global_offset, LocalViewT local,
                                             Idx local_offset, IdxGlobal n) {
    Idx local_id = group.get_local_id()[0];
    Idx local_size = group.get_local_range()[0];
    Idx loop_iters = n / local_size;
    const char* func_name = __func__;
    global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset, "n",
                                          n);
    for (Idx j = 0; j < loop_iters; j++) {
      if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
        local[padder(local_offset + local_id + j * local_size)] = global[global_offset + local_id + j * local_size];
      } else {
        global[global_offset + local_id + j * local_size] = local[padder(local_offset + local_id + j * local_size)];
      }
    }
    Idx loop_copies = loop_iters * local_size;
    subrange_copy(global_data, group, global, global_offset + loop_copies, local, local_offset + loop_copies,
                  n - loop_copies);
    return n;
  }
};

/**
 * Copies data from global memory to local memory. Expects the value of most input arguments to be the
 * same for work-items in the group described by template parameter "Level".
 *
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam SubgroupSize size of the subgroup
 * @tparam Pad Whether to add a pad after each `PORTFFT_N_LOCAL_BANKS * BankLinesPerPad` elements in local memory to
 * avoid bank conflicts.
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam LocalViewT The type of the local memory view
 * @tparam GlobalViewT The type of the global memory view
 * @param global_data global data for the kernel
 * @param global pointer to global memory
 * @param local pointer to local memory
 * @param total_num_elems total number of values to copy per group
 * @param global_offset offset to the global pointer
 * @param local_offset offset to the local pointer
 */
template <transfer_direction TransferDirection, level Level, Idx SubgroupSize, detail::pad Pad, Idx BankLinesPerPad,
          typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE void global_local_contiguous_copy(detail::global_data_struct global_data, GlobalViewT global,
                                                 LocalViewT local, IdxGlobal total_num_elems,
                                                 IdxGlobal global_offset = 0, Idx local_offset = 0) {
  using elem_t = get_element_remove_cv_t<GlobalViewT>;
  static_assert(std::is_same_v<elem_t, get_element_remove_cv_t<LocalViewT>>, "Type mismatch between global and local");
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset);
  sycl::nd_item<1> it = global_data.it;
  auto group = [=]() {
    if constexpr (Level == level::SUBGROUP) {
      return it.get_sub_group();
    } else {
      return it.get_group();
    }
  }();
  using copy_helper_t = global_local_copy_helper<TransferDirection, Level, elem_t, SubgroupSize, Pad, BankLinesPerPad>;

#ifdef PORTFFT_USE_SG_TRANSFERS
  std::size_t copied_by_sg =
      copy_helper_t::sg_block_copy(global_data, global, global_offset, local, local_offset, total_num_elems);
  local_offset += copied_by_sg;
  global_offset += copied_by_sg;
  total_num_elems -= copied_by_sg;
#else
  const elem_t* global_ptr = &global[global_offset];
  const elem_t* global_aligned_ptr = reinterpret_cast<const elem_t*>(detail::round_up_to_multiple(
      reinterpret_cast<std::uintptr_t>(global_ptr), alignof(typename copy_helper_t::vec_t)));
  IdxGlobal unaligned_elements = static_cast<IdxGlobal>(global_aligned_ptr - global_ptr);

  // Load the first few unaligned elements. Assumes group size > alignof(vec_t) / sizeof(vec_t).
  copy_helper_t::subrange_copy(group, global, global_offset, local, local_offset, unaligned_elements);
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;
  total_num_elems -= unaligned_elements;

  // Each workitem loads a chunk of consecutive elements. Chunks loaded by a group are consecutive.
  Idx local_size = group.get_local_range()[0];
  IdxGlobal stride = static_cast<IdxGlobal>(local_size * copy_helper_t::ChunkSize);
  IdxGlobal rounded_down_num_elems = (total_num_elems / stride) * stride;
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    copy_helper_t::vec_aligned_block_copy(group, global, global_offset + i, local, local_offset + i);
  }
  local_offset += rounded_down_num_elems;
  global_offset += rounded_down_num_elems;
  total_num_elems -= rounded_down_num_elems;
#endif
  // We cannot load fixed-size blocks of data anymore, so we use naive copies.
  copy_helper_t::naive_copy(global_data, group, global, global_offset, local, local_offset, total_num_elems);
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
                                 IdxGlobal total_num_elems, IdxGlobal global_offset = 0, Idx local_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::GLOBAL_TO_LOCAL, Level, SubgroupSize, Pad,
                                       BankLinesPerPad>(global_data, global, local, total_num_elems, global_offset,
                                                        local_offset);
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
                                 IdxGlobal total_num_elems, Idx local_offset = 0, IdxGlobal global_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::LOCAL_TO_GLOBAL, Level, SubgroupSize, Pad,
                                       BankLinesPerPad>(global_data, global, local, total_num_elems, global_offset,
                                                        local_offset);
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
PORTFFT_INLINE void local2global_transposed(detail::global_data_struct global_data, Idx N, Idx M, Idx stride, T* local,
                                            T* global, IdxGlobal offset) {
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
                                            T* local_ptr, IdxGlobal offset, Idx num_complex, IdxGlobal stride_global,
                                            Idx stride_local) {
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
                                     Idx local_id, Idx workers_in_group, TDstIdx destination_offset = 0) {
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
    if constexpr (Pad == detail::pad::DO_PAD) {
      destination_idx = detail::pad_local<Pad>(destination_idx_unpadded, BankLinesPerPad);
    } else {
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
 * Transfer data between local and private memory, with 3 levels of transpositions / strides.
 * local_or_global_idx = s1 * (s2 * (s3 * i + o3) + o2) + o1
 * private_idx = 2 * i
 * output[out_idx] = input[in_idx]
 * output[out_idx + 1] = input[in_idx + 1]
 * where out_idx and in_idx are chosen according to TransferDirection
 * for i in [0, NumComplexElements) where loc is indexed repecting padding.
 *
 * @tparam TransferDirection Direction of Transfer
 * @tparam Pad Whether or not to pad
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
 * @param bank_lines_per_pad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 */
template <detail::transfer_direction TransferDirection, detail::pad Pad, Idx NumComplexElements, typename TDstIdx, typename InputT,
          typename DestT>
PORTFFT_INLINE void transfer_strided(detail::global_data_struct global_data, InputT input, DestT output, TDstIdx stride_1,
                                     TDstIdx offset_1, TDstIdx stride_2, TDstIdx offset_2, TDstIdx stride_3, TDstIdx offset_3,
                                     Idx bank_lines_per_pad) {
  static_assert(std::is_same_v<detail::get_element_remove_cv_t<InputT>, detail::get_element_t<DestT>>,
                "Type mismatch between local and private views");
  static_assert(((Pad == detail::pad::DO_PAD) && std::is_same_v<TDstIdx, Idx>) ||
                ((Pad == detail::pad::DONT_PAD) && std::is_same_v<TDstIdx, IdxGlobal>));
  const char* func_name = __func__;
  global_data.log_message_local(__func__, "stride_1", stride_1, "offset_1", offset_1, "stride_2", stride_2, "offset_2",
                                offset_2, "stride_3", stride_3, "offset_3", offset_3);
  detail::unrolled_loop<0, NumComplexElements, 1>([&](const Idx j) PORTFFT_INLINE {
    TDstIdx base_offset = stride_1 * (stride_2 * static_cast<TDstIdx>((j * stride_3 + offset_3)) + offset_2) + offset_1;
    if constexpr (TransferDirection == detail::transfer_direction::LOCAL_TO_PRIVATE) {
      global_data.log_message(func_name, "from", detail::pad_local<Pad>(base_offset, bank_lines_per_pad), "to", 2 * j,
                              "value", input[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)]);
      global_data.log_message(func_name, "from", detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad), "to",
                              2 * j + 1, "value", input[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)]);
      output[2 * j] = input[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)];
      output[2 * j + 1] = input[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_LOCAL) {
      global_data.log_message(func_name, "from", 2 * j, "to", detail::pad_local<Pad>(base_offset, bank_lines_per_pad),
                              "value", input[2 * j]);
      global_data.log_message(func_name, "from", 2 * j + 1, "to",
                              detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad), "value", input[2 * j + 1]);
      output[detail::pad_local<Pad>(base_offset, bank_lines_per_pad)] = input[2 * j];
      output[detail::pad_local<Pad>(base_offset + 1, bank_lines_per_pad)] = input[2 * j + 1];
    }
    if constexpr (TransferDirection == detail::transfer_direction::PRIVATE_TO_GLOBAL) {
      output[base_offset] = input[2 * j];
      output[base_offset + 1] = input[2 * j + 1];
    }
  });
}

/**
 * Views the data in the local memory as an NxM matrix, and stores data from the private memory along the column:
 * loc[2 * stride * (num_workers * i + thread_id) + 2 * col_num] := priv[i]
 * loc[2 * stride * (num_workers * i + thread_id) + 2 * col_num + 1] := priv[i + 1]
 * for i in [0, NumElementsPerWI) where loc is indexed repecting padding.
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
  transfer_strided<detail::transfer_direction::PRIVATE_TO_LOCAL, Pad, NumElementsPerWI>(
      global_data, priv, local, 1, 0, 2 * stride, 2 * col_num, num_workers, thread_id, BankLinesPerPad);
}

/**
 * Views the data in the local memory as an NxM matrix, and loads a column into the private memory
 * priv[2 * i] := loc[2 * stride * (i + thread_id * NumElementsPerWI) + 2 * col_num]
 * priv[2 * i + 1] := loc[2 * stride * (i + thread_id * NumElementsPerWI) + 2 * col_num + 1]
 * for i in [0, NumElementsPerWI) where loc is indexed repecting padding.
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
  transfer_strided<detail::transfer_direction::LOCAL_TO_PRIVATE, Pad, NumElementsPerWI>(
      global_data, local, priv, 1, 0, 2 * stride, 2 * col_num, 1, thread_id * NumElementsPerWI, BankLinesPerPad);
}

/**
 * Transfers data from local memory which is strided to global memory, which too is strided in a transposed fashion
 *
 * @tparam Pad Whether or not to pad local memory
 * @tparam T Scalar type
 *
 * @param global_data global data for the kernel
 * @param loc Pointer to local memory
 * @param global Pointer to global memory
 * @param global_offset Offset to global memory
 * @param local_stride stride value in local memory
 * @param N Number of rows
 * @param M Number of Columns
 * @param fft_size Size of the problem
 * @param bank_lines_per_pad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 */
template <detail::pad Pad, typename T>
PORTFFT_INLINE void local_strided_2_global_strided_transposed(detail::global_data_struct global_data, T* loc, T* global,
                                                              IdxGlobal global_offset, Idx local_stride, Idx N, Idx M,
                                                              Idx fft_size, Idx bank_lines_per_pad) {
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "global_offset", global_offset, "local_stride", local_stride, "N", N, "M", M,
                                "fft_size", fft_size);
  Idx batch_num = static_cast<Idx>(global_data.it.get_local_linear_id()) / 2;
  for (Idx i = 0; i < fft_size; i++) {
    Idx source_row = i / N;
    Idx source_col = i % N;
    Idx local_idx = detail::pad_local<Pad>(
        local_stride * (source_col * M + source_row) + static_cast<Idx>(global_data.it.get_local_id(0)),
        bank_lines_per_pad);
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
 * @tparam pad Whether or not to consider padding in local memory
 * @tparam Level Which level (subgroup or workgroup) does the transfer.
 * @tparam T Scalar Type
 *
 * @param global_data  global data for the kernel
 * @param global_base_ptr Global Pointer
 * @param local_ptr Local Pointer
 * @param offset Offset from which the strided loads would begin
 * @param num_complex Number of complex numbers per workitem
 * @param stride_global Stride Value for global memory
 * @param stride_local Stride Value for Local Memory
 */
template <detail::pad Pad, detail::level Level, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local_transposed2_global_transposed(detail::global_data_struct global_data, T* global_base_ptr,
                                                        T* local_ptr, IdxGlobal offset, Idx num_complex,
                                                        IdxGlobal stride_global, Idx stride_local) {
  global_data.log_message_local(__func__,
                                "Tranferring data from local to global memory with stride_global:", stride_global,
                                " global offset = ", offset, "number of elements per workitem = ", num_complex,
                                " and local stride:", stride_local);
  Idx local_id;
  if constexpr (Level == detail::level::SUBGROUP) {
    local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  } else {
    local_id = static_cast<Idx>(global_data.it.get_local_id(0));
  }

  for (Idx i = 0; i < num_complex; i++) {
    Idx local_index = detail::pad_local<Pad>(2 * i * stride_local + local_id, BankLinesPerPad);
    IdxGlobal global_index = offset + static_cast<IdxGlobal>(local_id) + static_cast<IdxGlobal>(2 * i) * stride_global;
    global_data.log_message(__func__, "from", local_index, "to", global_index, "value", local_ptr[local_index]);
    global_base_ptr[global_index] = local_ptr[local_index];
  }
}

/**
 * Transfers data from local memory (which is in contiguous layout) to global memory (which is in strided layout),
 * by interpreting the local memory as if strided. To be used specifically for workgroup FFTs, where input in PACKED but
 * output is BATCHED_INTERLEAVED
 *
 * @tparam Pad Whether or not Padding is to be applied
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 * @tparam T Pointer type to local and global
 *
 * @param global_data global data for the kernel
 * @param global_ptr Pointer to global memory
 * @param local_ptr Pointer to local memory
 * @param global_stride Stride applicable to global memory
 * @param global_offset  Offset applicable to global memory
 * @param num_elements Total number of elements to be transferred
 * @param N Viewing num_elements as product of two factors, N being the first factor
 * @param M Viewing num_elements as product of two factors, M being the second factor
 */
template <detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void localstrided_2global_strided(detail::global_data_struct global_data, T* global_ptr, T* local_ptr,
                                                 IdxGlobal global_stride, IdxGlobal global_offset, Idx num_elements,
                                                 Idx N, Idx M) {
  global_data.log_message_global(__func__, "transferring data with global_stride = ", global_stride,
                                 " global offset = ", global_offset);
  Idx start_index = static_cast<Idx>(global_data.it.get_local_linear_id());
  Idx index_stride = static_cast<Idx>(global_data.it.get_local_range(0));
  for (Idx idx = start_index; idx < num_elements; idx += index_stride) {
    Idx source_row = idx / N;
    Idx source_col = idx % N;
    Idx base_offset = detail::pad_local<Pad>(2 * source_col * M + 2 * source_row, BankLinesPerPad);
    IdxGlobal base_global_idx = static_cast<IdxGlobal>(idx) * global_stride + global_offset;
    global_data.log_message(__func__, "from (", base_offset, ",", base_offset, ") ", "to (", base_global_idx,
                            base_global_idx + 1, "values = (", local_ptr[base_offset], ",", local_ptr[base_offset + 1],
                            ")");
    global_ptr[base_global_idx] = local_ptr[base_offset];
    global_ptr[base_global_idx + 1] = local_ptr[base_offset + 1];
  }
}

/**
 * Transfers data from local memory (which is in strided layout) to global memory (which is in strided layout),
 * by adding another stride to local memory. To be used specifically for workgroup FFTs, where input is
 * BATCHED_INTERLEAVED and output is BATCHED_INTERLEAVED as well.
 * Call site is resposible for managing OOB accesses
 *
 * @tparam Pad Whether or not Padding is to be applied
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 * @tparam T Pointer type to local and global
 *
 * @param global_data global data for the kernel
 * @param global_ptr Pointer to global memory
 * @param local_ptr Pointer to local memory
 * @param global_stride Stride applicable to global memory
 * @param global_offset Offset applicable to global memory
 * @param local_stride Stride applicable to local memory
 * @param num_elements Total number of elements to be transferred per workitem
 * @param N Viewing num_elements as product of two factors, N being the first factor
 * @param M Viewing num_elements as product of two factors, M being the second factor
 */
template <detail::pad Pad, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void local2strides_2global_strided(detail::global_data_struct global_data, T* global_ptr, T* local_ptr,
                                                  IdxGlobal global_stride, IdxGlobal global_offset, Idx local_stride,
                                                  Idx num_elements, Idx N, Idx M) {
  global_data.log_message_global(__func__, "transferring data with global_stride = ", global_stride,
                                 " global offset = ", global_offset, " local stride = ", local_stride);
  for (Idx idx = 0; idx < num_elements; idx++) {
    Idx local_stride_2 = (idx % N) * M + (idx / N);
    Idx base_offset = detail::pad_local<Pad>(
        local_stride_2 * local_stride + static_cast<Idx>(global_data.it.get_local_id(0)), BankLinesPerPad);
    IdxGlobal global_idx = static_cast<IdxGlobal>(idx) * global_stride + global_offset +
                           static_cast<IdxGlobal>(global_data.it.get_local_id(0));
    global_data.log_message(__func__, "from", base_offset, "to", global_idx, "value", local_ptr[base_offset]);
    global_ptr[global_idx] = local_ptr[base_offset];
  }
}

};  // namespace portfft

#endif
