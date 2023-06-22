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

#ifndef SYCL_FFT_GENERAL_DISPATCHER_HPP
#define SYCL_FFT_GENERAL_DISPATCHER_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <common/transfers.hpp>
#include <common/workgroup.hpp>
#include <common/workitem.hpp>
#include <enums.hpp>

namespace sycl_fft {

namespace detail {

/**
 * Implementation of FFT for sizes that can be done by independent work items.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam transpose_in whether input is transposed (interpreting it as a matrix of batch size times FFT size)
 * @tparam N size of each transform
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local memory pointer. Must have enough space for 2*N*subgroup_size
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, detail::transpose transpose_in, int N, typename T>
__attribute__((always_inline)) inline void workitem_impl(const T* input, T* output, T* loc, std::size_t n_transforms,
                                                         sycl::nd_item<1> it, T scaling_factor) {
  constexpr std::size_t N_reals = 2 * N;

  T priv[N_reals];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t global_id = it.get_global_id(0);
  std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  std::size_t global_size = it.get_global_range(0);
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t local_offset = N_reals * subgroup_size * subgroup_id;

  for (std::size_t i = global_id; i < roundUpToMultiple(n_transforms, subgroup_size); i += global_size) {
    bool working = i < n_transforms;
    std::size_t n_working = sycl::min(subgroup_size, n_transforms - i + subgroup_local_id);

    if constexpr (transpose_in == detail::transpose::NOT_TRANSPOSED) {
      global2local<pad::DO_PAD, level::SUBGROUP>(it, input, loc, N_reals * n_working, N_reals * (i - subgroup_local_id),
                                                 local_offset);
      sycl::group_barrier(sg);
    }
    if (working) {
      if constexpr (transpose_in == detail::transpose::TRANSPOSED) {
        unrolled_loop<0, N_reals, 2>([&](const std::size_t j) __attribute__((always_inline)) {
          using T_vec = sycl::vec<T, 2>;
          reinterpret_cast<T_vec*>(&priv[j])->load(0, detail::get_global_multi_ptr(&input[i * 2 + j * n_transforms]));
        });
      } else {
        local2private<N_reals, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals, local_offset);
      }
      wi_dft<dir, N, 1, 1>(priv, priv);
      unrolled_loop<0, N_reals, 2>([&](int i) __attribute__((always_inline)) {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      private2local<N_reals, pad::DO_PAD>(priv, loc, subgroup_local_id, N_reals, local_offset);
    }
    sycl::group_barrier(sg);
    local2global<pad::DO_PAD, level::SUBGROUP>(it, loc, output, N_reals * n_working, local_offset,
                                               N_reals * (i - subgroup_local_id));
    sycl::group_barrier(sg);
  }
}

/**
 * Implementation of FFT for sizes that can be done by a subgroup.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam factor_wi factor of the FFT size. How many elements per FFT are processed by one workitem
 * @tparam factor_sg factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*factor_wi*factor_sg*subgroup_size
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, int factor_wi, int factor_sg, typename T>
__attribute__((always_inline)) inline void subgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                         std::size_t n_transforms, sycl::nd_item<1> it,
                                                         const T* twiddles, T scaling_factor) {
  constexpr int N_reals_per_wi = 2 * factor_wi;

  T priv[N_reals_per_wi];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  std::size_t subgroup_id = sg.get_group_id();
  constexpr std::size_t n_sgs_in_wg = SYCLFFT_SGS_IN_WG;
  std::size_t id_of_sg_in_kernel = subgroup_id + it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  std::size_t n_ffts_per_sg = subgroup_size / factor_sg;
  std::size_t max_wis_working = n_ffts_per_sg * factor_sg;
  std::size_t n_reals_per_fft = factor_sg * N_reals_per_wi;
  std::size_t n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  std::size_t id_of_fft_in_sg = subgroup_local_id / factor_sg;
  std::size_t id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
  std::size_t n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  std::size_t id_of_wi_in_fft = subgroup_local_id % factor_sg;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      roundUpToMultiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  global2local<pad::DONT_PAD, level::WORKGROUP>(it, twiddles, loc_twiddles, N_reals_per_wi * factor_sg);
  sycl::group_barrier(it.get_group());

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    std::size_t n_ffts_worked_on_by_sg = sycl::min(n_transforms - (i - id_of_fft_in_sg), n_ffts_per_sg);

    global2local<pad::DO_PAD, level::SUBGROUP>(it, input, loc, n_ffts_worked_on_by_sg * n_reals_per_fft,
                                               n_reals_per_fft * (i - id_of_fft_in_sg), subgroup_id * n_reals_per_sg);

    sycl::group_barrier(sg);
    if (working) {
      local2private<N_reals_per_wi, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals_per_wi,
                                                 subgroup_id * n_reals_per_sg);
    }
    sg_dft<dir, factor_wi, factor_sg>(priv, sg, loc_twiddles);
    unrolled_loop<0, N_reals_per_wi, 2>([&](int i) __attribute__((always_inline)) {
      priv[i] *= scaling_factor;
      priv[i + 1] *= scaling_factor;
    });
    if constexpr (factor_sg == SYCLFFT_TARGET_SUBGROUP_SIZE) {
      // in this case we get fully coalesced memory access even without going through local memory
      // TODO we may want to tune maximal `factor_sg` for which we use direct stores.
      if (working) {
        store_transposed<N_reals_per_wi, pad::DONT_PAD>(priv, output, id_of_wi_in_fft, factor_sg,
                                                        i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
      }
    } else {
      if (working) {
        store_transposed<N_reals_per_wi, pad::DO_PAD>(priv, loc, id_of_wi_in_fft, factor_sg,
                                                      subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
      }
      sycl::group_barrier(sg);
      local2global<pad::DO_PAD, level::SUBGROUP>(it, loc, output, n_ffts_worked_on_by_sg * n_reals_per_fft,
                                                 subgroup_id * n_reals_per_sg, n_reals_per_fft * (i - id_of_fft_in_sg));
      sycl::group_barrier(sg);
    }
  }
}

/**
 * Implementation that can be handled by a workgroup
 *
 * @tparam dir Direction of the FFt
 * @tparam fft_size Problem size of the FFT
 * @tparam T_in Input pointer type
 * @tparam T_out Output pointer type
 * @tparam T Scalar type
 * @tparam T_twiddles Twiddles to be used by the FFT
 *
 * @param input Input pointer
 * @param output Output pointer
 * @param loc Local Accessor containing the inupts
 * @param loc_twiddles local accessor to to twiddles to be used by sub FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param n_transforms Batch size
 * @param it Associated nd_item
 * @param twiddles Pointer to the global memory containing twiddles for sub FFTs
 * @param scaling_factor factor by which the result will be scaled
 */
template <direction dir, std::size_t fft_size, typename T>
__attribute__((always_inline)) inline void workgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                          std::size_t n_transforms, sycl::nd_item<1> it,
                                                          const T* twiddles, T scaling_factor) {
  std::size_t num_workgroups = it.get_group_range(0);
  std::size_t wg_id = it.get_group(0);
  std::size_t max_global_offset = 2 * (n_transforms - 1) * fft_size;
  std::size_t global_offset = 2 * fft_size * wg_id;
  std::size_t offset_increment = 2 * fft_size * num_workgroups;
  constexpr std::size_t N = detail::factorize(fft_size);
  constexpr std::size_t M = fft_size / N;
  const T* wg_twiddles = twiddles + 2 * (M + N);

  global2local<pad::DONT_PAD, level::WORKGROUP>(it, twiddles, loc_twiddles, 2 * (M + N));
  sycl::group_barrier(it.get_group());

  for (std::size_t offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    global2local<pad::DO_PAD, level::WORKGROUP>(it, input, loc, 2 * fft_size, offset);
    sycl::group_barrier(it.get_group());
    wg_dft<dir, fft_size, N, M>(loc, loc_twiddles, wg_twiddles, it, scaling_factor);
    local2global_transposed<N, M, SYCLFFT_SGS_IN_WG, SYCLFFT_TARGET_SUBGROUP_SIZE, detail::pad::DO_PAD>(it, loc, output,
                                                                                                        offset);
    sycl::group_barrier(it.get_group());
  }
}

/**
 * Selects appropriate template instantiation of workitem implementations for
 * given size of DFT.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam transpose_in whether input is transposed (interpreting it as a matrix of batch size times FFT size)
 * @tparam factor_sg factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size
 * values
 * @param fft_size size of each transform
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, detail::transpose transpose_in, typename T>
__attribute__((always_inline)) inline void workitem_dispatcher(const T* input, T* output, T* loc, std::size_t fft_size,
                                                               std::size_t n_transforms, sycl::nd_item<1> it,
                                                               T scaling_factor) {
  switch (fft_size) {
#define SYCL_FFT_WI_DISPATCHER_IMPL(N)                                                           \
  case N:                                                                                        \
    if constexpr (fits_in_wi<T>(N)) {                                                            \
      workitem_impl<dir, transpose_in, N>(input, output, loc, n_transforms, it, scaling_factor); \
    }                                                                                            \
    break;
    SYCL_FFT_WI_DISPATCHER_IMPL(1)
    SYCL_FFT_WI_DISPATCHER_IMPL(2)
    SYCL_FFT_WI_DISPATCHER_IMPL(4)
    SYCL_FFT_WI_DISPATCHER_IMPL(8)
    SYCL_FFT_WI_DISPATCHER_IMPL(16)
    SYCL_FFT_WI_DISPATCHER_IMPL(32)
    // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_WI_DISPATCHER_IMPL
  }
}

/**
 * Selects appropriate template instantiation of subgroup implementation for
 * given factor_sg.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam N size of each transform
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*factor_wi*factor_sg*subgroup_size
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, int factor_wi, typename T>
__attribute__((always_inline)) void cross_sg_dispatcher(int factor_sg, const T* input, T* output, T* loc,
                                                        T* loc_twiddles, std::size_t n_transforms, sycl::nd_item<1> it,
                                                        const T* twiddles, T scaling_factor) {
  switch (factor_sg) {
#define SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(M)                                                                          \
  case M:                                                                                                             \
    if constexpr (M <= SYCLFFT_TARGET_SUBGROUP_SIZE && !fits_in_wi<T>(M * factor_wi)) {                               \
      subgroup_impl<dir, factor_wi, M>(input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor); \
    }                                                                                                                 \
    break;
    // cross-sg size 1 cases are supported by workitem implementation
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(2)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(4)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(8)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(16)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(32)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(64)
    // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_CROSS_SG_DISPATCHER_IMPL
  }
}

/**
 * Selects appropriate template instantiation of subgroup implementation for
 * given factor_wi.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam T type of the scalar used for computations
 * @param factor_wi factor of the FFT size. How many elements per FFT are processed by one workitem
 * @param factor_sg factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles twiddle factors to use
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, typename T>
__attribute__((always_inline)) inline void subgroup_dispatcher(int factor_wi, int factor_sg, const T* input, T* output,
                                                               T* loc, T* loc_twiddles, std::size_t n_transforms,
                                                               sycl::nd_item<1> it, const T* twiddles,
                                                               T scaling_factor) {
  switch (factor_wi) {
#define SYCL_FFT_SG_WI_DISPATCHER_IMPL(N)                                                                  \
  case N:                                                                                                  \
    if constexpr (fits_in_wi<T>(N)) {                                                                      \
      cross_sg_dispatcher<dir, N>(factor_sg, input, output, loc, loc_twiddles, n_transforms, it, twiddles, \
                                  scaling_factor);                                                         \
    }                                                                                                      \
    break;
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(1)
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(2)
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(4)
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(8)
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(16)
    SYCL_FFT_SG_WI_DISPATCHER_IMPL(32)
    // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_SG_WI_DISPATCHER_IMPL
  }
}

template <direction dir, typename T>
__attribute__((always_inline)) inline void workgroup_dispatcher(const T* input, T* output, std::size_t fft_size, T* loc,
                                                                T* loc_twiddles, std::size_t n_transforms,
                                                                sycl::nd_item<1> it, const T* twiddles,
                                                                T scaling_factor) {
  switch (fft_size) {
#define SYCL_FFT_WG_DISPATCHER_IMPL(N)                                                                    \
  case N:                                                                                                 \
    workgroup_impl<dir, N>(input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor); \
    break;
    SYCL_FFT_WG_DISPATCHER_IMPL(256)
    SYCL_FFT_WG_DISPATCHER_IMPL(512)
    SYCL_FFT_WG_DISPATCHER_IMPL(1024)
    SYCL_FFT_WG_DISPATCHER_IMPL(2048)
    SYCL_FFT_WG_DISPATCHER_IMPL(4096)
    SYCL_FFT_WG_DISPATCHER_IMPL(8192)
    // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_WG_DISPATCHER_IMPL
  }
}

/**
 * Selects appropriate implementation for given problem size.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam transpose_in whether input is transposed (interpreting it as a matrix of batch size times FFT size)
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size
 * values if the subgroup implementation is used
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*factor_wi*factor_sg
 * values
 * @param fft_size size of each transform
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles twiddle factors to use
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, detail::transpose transpose_in, typename T>
__attribute__((always_inline)) inline void dispatcher(const T* input, T* output, T* loc, T* loc_twiddles,
                                                      std::size_t fft_size, std::size_t n_transforms,
                                                      sycl::nd_item<1> it, const T* twiddles, T scaling_factor) {
  // TODO: should decision which implementation to use and factorization be done
  // on host?
  if (fits_in_wi_device<T>(fft_size)) {
    workitem_dispatcher<dir, transpose_in>(input, output, loc, fft_size, n_transforms, it, scaling_factor);
    return;
  }
  std::size_t sg_size = it.get_sub_group().get_local_linear_range();
  // Check that fft_size can be represented as an int
  if (fft_size <= MAX_FFT_SIZE_WI * sg_size) {
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), static_cast<int>(sg_size));
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (fits_in_wi_device<T>(static_cast<std::size_t>(factor_wi))) {
      subgroup_dispatcher<dir>(factor_wi, factor_sg, input, output, loc, loc_twiddles, n_transforms, it, twiddles,
                               scaling_factor);
      return;
    }
  } else {
    workgroup_dispatcher<dir>(input, output, fft_size, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor);
    return;
  }
}

/**
 * Calculates twiddle factors needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param q queue
 * @param subgroup_size size of subgroup used by the compute kernel
 * @return T* pointer to device memory containing twiddle factors
 */
template <typename T>
T* calculate_twiddles(sycl::queue& q, std::size_t fft_size, int subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return nullptr;
  }

  // Check that fft_size can be represented as an int
  if (fft_size <= MAX_FFT_SIZE_WI * static_cast<std::size_t>(subgroup_size)) {
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), subgroup_size);
    int factor_wi = static_cast<int>(fft_size) / factor_sg;
    if (fits_in_wi<T>(factor_wi)) {
      T* res = sycl::malloc_device<T>(fft_size * 2, q);
      sycl::range<2> kernel_range({static_cast<std::size_t>(factor_sg), static_cast<std::size_t>(factor_wi)});
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(kernel_range, [=](sycl::item<2> it) {
          int n = static_cast<int>(it.get_id(0));
          int k = static_cast<int>(it.get_id(1));
          sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
        });
      });
      q.wait();  // waiting once here can be better than depending on the event
                 // for all future calls to compute
      return res;
    }
  } else {
    std::size_t N = detail::factorize(fft_size);
    std::size_t M = fft_size / N;
    int factor_sg_M = detail::factorize_sg(static_cast<int>(M), subgroup_size);
    int factor_wi_M = static_cast<int>(M) / factor_sg_M;
    if (!fits_in_wi<T>(factor_wi_M)) {
      throw std::runtime_error("FFT size " + std::to_string(M) + " is not supported for subgroup_size " +
                               std::to_string(subgroup_size));
    }
    int factor_sg_N = (detail::factorize_sg(static_cast<int>(N), subgroup_size));
    int factor_wi_N = static_cast<int>(N) / factor_sg_N;
    if (!fits_in_wi<T>(factor_wi_N)) {
      throw std::runtime_error("FFT size " + std::to_string(N) + " is not supported for subgroup_size " +
                               std::to_string(subgroup_size));
    }
    T* res = sycl::malloc_device<T>(2 * (M + N + fft_size), q);
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_M), static_cast<std::size_t>(factor_wi_M)}),
                       [=](sycl::item<2> it) {
                         int n = static_cast<int>(it.get_id(0));
                         int k = static_cast<int>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_M, factor_wi_M, n, k, res);
                       });
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_N), static_cast<std::size_t>(factor_wi_N)}),
                       [=](sycl::item<2> it) {
                         int n = static_cast<int>(it.get_id(0));
                         int k = static_cast<int>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_N, factor_wi_N, n, k, res + (2 * M));
                       });
    });
    q.wait();
    return res;
  }
  return nullptr;
}

template <typename T>
void populate_wg_twiddles(std::size_t fft_size, T* global_pointer, sycl::queue& queue) {
  std::size_t N = detail::factorize(fft_size);
  std::size_t M = fft_size / N;
  if (fits_in_wi<T>(M)) {
    return;
  }

  T* temp_host = sycl::malloc_host<T>(2 * fft_size, queue);

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      std::size_t index = 2 * (i * M + j);
      temp_host[index] =
          static_cast<T>(std::cos((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size)));
      temp_host[index + 1] =
          static_cast<T>(std::sin((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size)));
    }
  }
  queue.copy(temp_host, global_pointer, 2 * fft_size).wait();
  sycl::free(temp_host, queue);
}

/**
 * Calculates the amount of local memory needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param subgroup_size size of subgroup used by the compute kernel
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t num_scalars_in_local_mem(std::size_t fft_size, std::size_t subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return detail::pad_local(2 * fft_size * subgroup_size) * SYCLFFT_SGS_IN_WG;
  } else {
    if (fft_size <= MAX_FFT_SIZE_WI * subgroup_size) {
      int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size));
      int fact_wi = static_cast<int>(fft_size) / factor_sg;
      if (fits_in_wi<T>(fact_wi)) {
        std::size_t n_ffts_per_sg = subgroup_size / static_cast<std::size_t>(factor_sg);
        return detail::pad_local(2 * fft_size * n_ffts_per_sg) * SYCLFFT_SGS_IN_WG;
      }
    } else {
      return detail::pad_local(2 * fft_size);
    }
  }
  return 0;
}

template <typename T>
std::size_t num_scalars_in_twiddles(std::size_t fft_size, std::size_t subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return 0;
  } else {
    int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size));
    if (fits_in_wi<T>(static_cast<int>(fft_size) / factor_sg)) {
      return 2 * fft_size;
    } else {
      std::size_t N = detail::factorize(fft_size);
      std::size_t M = fft_size / N;
      return 2 * (M + N);
    }
  }

  return 0;
}

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param n_compute_units number fo compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t get_global_size(std::size_t fft_size, std::size_t n_transforms, std::size_t subgroup_size,
                            std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t n_sgs_we_can_utilize = (n_transforms + subgroup_size - 1) / subgroup_size;
  if (fits_in_wi<T>(fft_size)) {
    return subgroup_size * detail::roundUpToMultiple(sycl::min(maximum_n_sgs, n_sgs_we_can_utilize),
                                                     static_cast<std::size_t>(SYCLFFT_SGS_IN_WG));
  }
  // Check that fft_size can be represented as an int
  if (fft_size <= MAX_FFT_SIZE_WI * subgroup_size) {
    std::size_t factor_sg =
        static_cast<std::size_t>(detail::factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size)));
    std::size_t n_ffts_per_sg = subgroup_size / factor_sg;
    n_sgs_we_can_utilize = divideCeil(n_transforms, n_ffts_per_sg);
    // Less subgroups launched seems to be optimal for subgroup implementation.
    // This is a temporary solution until we have tuning
    maximum_n_sgs /= 4;
    return subgroup_size * detail::roundUpToMultiple(sycl::min(maximum_n_sgs, n_sgs_we_can_utilize),
                                                     static_cast<std::size_t>(SYCLFFT_SGS_IN_WG));
  } else {
    return subgroup_size * detail::roundUpToMultiple(sycl::min(maximum_n_sgs, n_sgs_we_can_utilize),
                                                     static_cast<std::size_t>(SYCLFFT_SGS_IN_WG));
  }
  return 0;
}

}  // namespace detail
}  // namespace sycl_fft

#endif
