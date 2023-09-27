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
#include <traits.hpp>

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
template <transfer_direction TransferDirection, typename RealT, std::size_t SubgroupSize>
class global_local_copy_helper {
 private:
  static constexpr std::size_t chunk_size_raw = PORTFFT_VEC_LOAD_BYTES / sizeof(RealT);
  static_assert(TransferDirection == transfer_direction::LOCAL_TO_GLOBAL ||
                TransferDirection == transfer_direction::GLOBAL_TO_LOCAL);

 public:
  using real_type = RealT;

  // The number of reals to copy at once.
  static constexpr int chunk_size = chunk_size_raw < 1 ? 1 : chunk_size_raw;
  // The sycl::vec type used for copying.
  using vec_t = sycl::vec<real_type, chunk_size>;
  // The block size used for fast block copying with sub-groups.
  static constexpr int sg_block_copy_block_size = chunk_size * SubgroupSize;

  /** Copy between index-contiguous copy between global and local memory using sub-group loads/stores for real data.
   *  Works on a fixed size block. Data does not need to be aligned to alignof(vec_t). Arguments are
   *  expected to be the same for all values in the sub-group.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, sg_block_copy_block_size)
   *
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @returns The number of reals copied
   */
  template <typename GlobalViewT, typename LocalViewT>
  static std::size_t sg_block_copy(detail::global_data_struct global_data, GlobalViewT global,
                                   std::size_t global_offset, LocalViewT local, std::size_t local_offset) {
    // Is the local memory suitable for using Intel's subgroup copy extensions with?
    constexpr bool isSgContiguous = PORTFFT_N_LOCAL_BANKS % SubgroupSize == 0 || !LocalViewT::is_padded;
    const char* func_name = __func__;
    global_data.log_message_subgroup(func_name, "sg_block_copy_block_size", sg_block_copy_block_size, "global_offset",
                                     global_offset, "local_offset", local_offset, "isSgContiguous", isSgContiguous);
    // A helper function to generate indexes in local memory.
    std::size_t local_id = global_data.sg.get_local_linear_id();
    auto indexer = [=](std::size_t i) __attribute__((always_inline)) {
      return local_offset + i * SubgroupSize + local_id;
    };
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      vec_t vec = global_data.sg.load<chunk_size>(detail::get_global_multi_ptr(&global[global_offset]));
      detail::unrolled_loop<0, chunk_size, 1>([&](std::size_t j) __attribute__((always_inline)) {
        if constexpr (isSgContiguous) {
          global_data.sg.store(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]),
                               vec[static_cast<int>(j)]);
        } else {
          local[indexer(j)] = vec[static_cast<int>(j)];
        }
      });
    } else {
      vec_t vec;
      detail::unrolled_loop<0, chunk_size, 1>([&](std::size_t j) __attribute__((always_inline)) {
        if constexpr (isSgContiguous) {
          vec[static_cast<int>(j)] =
              global_data.sg.load(detail::get_local_multi_ptr(&local[local_offset + j * SubgroupSize]));
        } else {
          vec[static_cast<int>(j)] = local[indexer(j)];
        }
      });
      global_data.sg.store(detail::get_global_multi_ptr(&global[global_offset]), vec);
    }
    return sg_block_copy_block_size;
  }

  /** Copy between index-contiguous copy between global and local memory using sub-groups loads/stores for real data.
   *  Data does not need to be aligned to alignof(vec_t). Arguments are expected to be the same for all values in the
   *  sub-group. Copies in multiples of sg_block_copy_block_size, and may copy less than the given item count n.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, m) where m <= n
   *
   *  @tparam GroupT The type of group to do the work with
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param group The work-group of sub-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @param n The count of elements to copy
   *  @returns The number of reals copied. May be less than n.
   */
  template <typename GroupT, typename GlobalViewT, typename LocalViewT>
  static std::size_t sg_block_copy(detail::global_data_struct global_data, GroupT /*group*/, GlobalViewT global,
                                   std::size_t global_offset, LocalViewT local, std::size_t local_offset,
                                   std::size_t n) {
    const char* func_name = __func__;
    global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "global_offset", global_offset, "local_offset",
                                                        local_offset, "n", n);
    if constexpr (std::is_same_v<GroupT, sycl::sub_group>) {
      static constexpr std::size_t block_size = sg_block_copy_block_size;
      std::size_t block_count = n / block_size;
      for (std::size_t block_idx{0}; block_idx < block_count; ++block_idx) {
        std::size_t offset = block_idx * block_size;
        sg_block_copy(global_data, global, global_offset + offset, local, local_offset + offset);
      }
      global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "copied_value_count", block_count * block_size);
      return block_count * block_size;
    } else {
      auto sg = global_data.sg;
      static constexpr std::size_t block_size = sg_block_copy_block_size;
      std::size_t subgroup_id = sg.get_group_id();
      std::size_t subgroup_count = sg.get_group_linear_range();
      std::size_t block_count = n / block_size;
      for (std::size_t block_idx{subgroup_id}; block_idx < block_count; block_idx += subgroup_count) {
        std::size_t offset = block_idx * block_size;
        sg_block_copy(global_data, global, global_offset + offset, local, local_offset + offset);
      }
      global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "copied_value_count", block_count * block_size);
      return block_count * block_size;
    }
  }

  /** Copy between index-contiguous copy between global and local memory using work-group or sub-group.
   *  Global memory argument must be aligned to alignof(vec_t).
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, chunk_size * group.get_local_range(0))
   *
   *  @tparam GroupT A SYCL work-group or sub-group
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param wg The sub-group or work-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @returns The number of reals copied
   */
  template <typename GroupT, typename GlobalViewT, typename LocalViewT>
  static std::size_t vec_aligned_block_copy(detail::global_data_struct global_data, GroupT wg, GlobalViewT global,
                                            std::size_t global_offset, LocalViewT local, std::size_t local_offset) {
    const char* func_name = __func__;
    global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "global_offset", global_offset, "local_offset",
                                                        local_offset, "copy_block_size",
                                                        chunk_size * wg.get_local_range()[0]);
    // A helper function to generate indexes in local memory.
    std::size_t local_id = wg.get_local_id()[0];
    std::size_t wi_offset = local_id * chunk_size;
    auto indexer = [=](std::size_t i) __attribute__((always_inline)) { return local_offset + wi_offset + i; };
    if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
      vec_t loaded;
      loaded = *reinterpret_cast<const vec_t*>(&global[global_offset + wi_offset]);
      detail::unrolled_loop<0, chunk_size, 1>([&](std::size_t j) __attribute__((always_inline)) {
        local[indexer(j)] = loaded[static_cast<int>(j)];
      });
    } else {
      vec_t to_store;
      detail::unrolled_loop<0, chunk_size, 1>([&](std::size_t j) __attribute__((always_inline)) {
        to_store[static_cast<int>(j)] = local[indexer(j)];
      });
      *reinterpret_cast<vec_t*>(&global[global_offset + wi_offset]) = to_store;
    }
    return chunk_size * wg.get_local_range()[0];
  }

  /** Copy between index-contiguous copy between global and local memory for n values, where n <
   *  group.get_local_range()[0].
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
   *
   *  @tparam GroupT A SYCL work-group or sub-group
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param wg The sub-group or work-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @param n The number of reals to copy. Must be less than the group range
   *  @returns The number of reals copied
   */
  template <typename GroupT, typename GlobalViewT, typename LocalViewT>
  static std::size_t subrange_copy(detail::global_data_struct global_data, GroupT wg, GlobalViewT global,
                                   std::size_t global_offset, LocalViewT local, std::size_t local_offset,
                                   std::size_t n) {
    const char* func_name = __func__;
    global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "global_offset", global_offset, "local_offset",
                                                        local_offset, "group_size", wg.get_local_range()[0], "n", n);
    std::size_t local_id = wg.get_local_id()[0];
    if (local_id < n) {
      if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
        local[local_offset + local_id] = global[global_offset + local_id];
      } else {
        global[global_offset + local_id] = local[local_offset + local_id];
      }
    }
    return n;
  }

  /** Copy between index-contiguous copy between global and local memory for n values.
   *  No particular requirements for alignment.
   *  global[global_offset + i] <-> local[local_offset + i] for i in [0, n)
   *
   *  @tparam GroupT A SYCL work-group or sub-group
   *  @tparam GlobalViewT The view of local memory
   *  @tparam LocalViewT The type of the local memory view
   *  @param global_data global data for the kernel
   *  @param wg The sub-group or work-group
   *  @param global The global memory view to copy to/from. Expects to be real element type
   *  @param global_offset The offset into global memory to start copying at
   *  @param local The local memory view. Expects to be real element type
   *  @param local_offset The offset into local memory to start copying at
   *  @param n The number of reals to copy
   *  @returns The number of reals copied
   */
  template <typename GroupT, typename GlobalViewT, typename LocalViewT>
  static std::size_t naive_copy(detail::global_data_struct global_data, GroupT wg, GlobalViewT global,
                                std::size_t global_offset, LocalViewT local, std::size_t local_offset, std::size_t n) {
    std::size_t local_id = wg.get_local_id()[0];
    std::size_t local_size = wg.get_local_range()[0];
    std::size_t loop_iters = n / local_size;
    const char* func_name = __func__;
    global_data.log_message_scoped<get_level_v<GroupT>>(func_name, "global_offset", global_offset, "local_offset",
                                                        local_offset);
    for (std::size_t j = 0; j < loop_iters; j++) {
      // NOTE: This looks like no-coalesced global memory access.
      if constexpr (TransferDirection == transfer_direction::GLOBAL_TO_LOCAL) {
        local[local_offset + local_id * loop_iters + j] = global[global_offset + local_id * loop_iters + j];
      } else {
        global[global_offset + local_id * loop_iters + j] = local[local_offset + local_id * loop_iters + j];
      }
    }
    std::size_t loop_copies = loop_iters * local_size;
    subrange_copy(global_data, wg, global, global_offset + loop_copies, local, local_offset + loop_copies,
                  n - loop_copies);
    return n;
  }
};

/**
 * Copies data from global memory to local memory. Expects the value of all input arguments except "it" to be the
 * same for work-items in the group described by template parameter "Level".
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
template <transfer_direction TransferDirection, level Level, int SubgroupSize, typename GlobalViewT,
          typename LocalViewT>
PORTFFT_INLINE inline void global_local_contiguous_copy(detail::global_data_struct global_data, GlobalViewT global,
                                                        LocalViewT local, std::size_t total_num_elems,
                                                        std::size_t global_offset = 0, std::size_t local_offset = 0) {
  using elem_t = std::remove_cv_t<typename LocalViewT::element_type>;
  static_assert(std::is_same_v<std::remove_cv_t<typename GlobalViewT::element_type>, elem_t>,
                "Different source / destination types.");
  static_assert(Level == detail::level::SUBGROUP || Level == detail::level::WORKGROUP,
                "Only implemented for subgroup and workgroup levels!");
  const char* func_name = __func__;
  global_data.log_message_scoped<Level>(func_name, "global_offset", global_offset, "local_offset", local_offset);
  sycl::nd_item<1> it = global_data.it;
  auto group = [=]() {
    if constexpr (Level == detail::level::SUBGROUP) {
      return it.get_sub_group();
    } else {
      return it.get_group();
    }
  }();
  using copy_helper_t =
      detail::global_local_copy_helper<TransferDirection, typename LocalViewT::element_type, SubgroupSize>;

#ifdef PORTFFT_USE_SG_TRANSFERS
  std::size_t copied_by_sg =
      copy_helper_t::sg_block_copy(it, group, global, global_offset, local, local_offset, total_num_elems);
  local_offset += copied_by_sg;
  global_offset += copied_by_sg;
  total_num_elems -= copied_by_sg;
#else
  const elem_t* global_ptr = &global[global_offset];
  const elem_t* global_aligned_ptr = reinterpret_cast<const elem_t*>(detail::round_up_to_multiple(
      reinterpret_cast<std::uintptr_t>(global_ptr), alignof(typename copy_helper_t::vec_t)));
  std::size_t unaligned_elements = static_cast<std::size_t>(global_aligned_ptr - global_ptr);

  // load the first few unaligned elements
  copy_helper_t::subrange_copy(group, global, global_offset, local, local_offset, unaligned_elements);
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;
  total_num_elems -= unaligned_elements;

  // Each workitem loads a chunk of `ChunkSize` consecutive elements. Chunks loaded by a group are consecutive.
  std::size_t local_size = group.get_local_range()[0];
  std::size_t stride = local_size * copy_helper_t::chunk_size;
  std::size_t rounded_down_num_elems = (total_num_elems / stride) * stride;
  for (std::size_t i = 0; i < rounded_down_num_elems; i += stride) {
    copy_helper_t::vec_aligned_block_copy(group, global, global_offset + i, local, local_offset + i);
  }
  local_offset += unaligned_elements;
  global_offset += unaligned_elements;
  total_num_elems -= unaligned_elements;
#endif
  // We can not load `ChunkSize`-sized chunks anymore, so we use more naive copy methods.
  copy_helper_t::naive_copy(global_data, group, global, global_offset, local, local_offset, total_num_elems);
}

}  // namespace detail

/**
 * Copies data from global memory to local memory. Expects the value of all input arguments except "it" to be the
 * same for work-items in the group described by template parameter "Level".
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
template <detail::level Level, int SubgroupSize, typename GlobalViewT, typename LocalViewT>
PORTFFT_INLINE inline void global2local(detail::global_data_struct global_data, GlobalViewT global, LocalViewT local,
                                        std::size_t total_num_elems, std::size_t global_offset = 0,
                                        std::size_t local_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::GLOBAL_TO_LOCAL, Level, SubgroupSize, GlobalViewT,
                                       LocalViewT>(global_data, global, local, total_num_elems, global_offset,
                                                   local_offset);
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
template <detail::level Level, int SubgroupSize, typename LocalViewT, typename GlobalViewT>
PORTFFT_INLINE inline void local2global(detail::global_data_struct global_data, LocalViewT local, GlobalViewT global,
                                        std::size_t total_num_elems, std::size_t local_offset = 0,
                                        std::size_t global_offset = 0) {
  detail::global_local_contiguous_copy<detail::transfer_direction::LOCAL_TO_GLOBAL, Level, SubgroupSize, GlobalViewT,
                                       LocalViewT>(global_data, global, local, total_num_elems, global_offset,
                                                   local_offset);
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
template <int NumElemsPerWI, typename PrivViewT, typename DestViewT>
PORTFFT_INLINE void store_transposed(detail::global_data_struct global_data, const PrivViewT priv,
                                     DestViewT destination, std::size_t local_id, std::size_t workers_in_group,
                                     std::size_t destination_offset = 0) {
  static_assert(std::is_same_v<typename PrivViewT::element_type, typename DestViewT::element_type>,
                "Source / destination element type mismatch.");
  const char* func_name = __func__;
  global_data.log_message_local(func_name, "local_id", local_id, "workers_in_group", workers_in_group,
                                "destination_offset", destination_offset);
  constexpr int VecSize = 2;  // each workitem stores 2 consecutive values (= one complex value)
  using T_vec = sycl::vec<typename PrivViewT::real_type, VecSize>;
  const T_vec* priv_vec = reinterpret_cast<const T_vec*>(priv.data);
  T_vec* destination_vec = reinterpret_cast<T_vec*>(destination.data);

  detail::unrolled_loop<0, NumElemsPerWI, 1>([&](int i) PORTFFT_INLINE {
    std::size_t destination_idx = destination_offset + local_id + static_cast<std::size_t>(i) * workers_in_group;
    global_data.log_message(func_name, "from", i, "to", destination_idx, "value", priv[i]);
    // if the destination address is aligned, we can use vector store:
    if (!DestViewT::is_padded && (reinterpret_cast<std::uintptr_t>(destination.data) % alignof(T_vec) == 0)) {
      destination_vec[destination_idx] = priv_vec[i];
    } else {
      destination[destination_idx] = priv[i];
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
