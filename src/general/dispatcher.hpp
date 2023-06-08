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
 * @tparam N size of each transform
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*N*subgroup_size
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction dir, int N, typename T_in, typename T_out, typename T>
__attribute__((always_inline)) inline void workitem_impl(T_in input, T_out output,
                                                         const sycl::local_accessor<T, 1>& loc,
                                                         std::size_t n_transforms, sycl::nd_item<1> it,
                                                         T scaling_factor) {
  constexpr int N_reals = 2 * N;

  T priv[N_reals];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t global_id = it.get_global_id(0);
  std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  std::size_t global_size = it.get_global_range(0);
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t local_offset = N_reals * subgroup_size * subgroup_id;

  for (size_t i = global_id; i < roundUpToMultiple(n_transforms, subgroup_size); i += global_size) {
    bool working = i < n_transforms;
    int n_working = sycl::min(subgroup_size, n_transforms - i + subgroup_local_id);

    global2local<true>(input, loc, N_reals * n_working, subgroup_size, subgroup_local_id,
                       N_reals * (i - subgroup_local_id), local_offset);
    sycl::group_barrier(sg);
    if (working) {
      local2private<N_reals, true>(loc, priv, subgroup_local_id, N_reals, local_offset);
      wi_dft<dir, N, 1, 1>(priv, priv);
      unrolled_loop<0, N_reals, 2>([&](const int i) __attribute__((always_inline)) {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      private2local<N_reals, true>(priv, loc, subgroup_local_id, N_reals, local_offset);
    }
    sycl::group_barrier(sg);
    local2global<true>(loc, output, N_reals * n_working, subgroup_size, subgroup_local_id, local_offset,
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
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
 * @tparam T type of the scalar used for computations
 * @tparam T_twiddles pointer containing the twiddles
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
template <direction dir, int factor_wi, int factor_sg, typename T_in, typename T_out, typename T, typename T_twiddles>
__attribute__((always_inline)) inline void subgroup_impl(T_in input, T_out output,
                                                         const sycl::local_accessor<T, 1>& loc,
                                                         const sycl::local_accessor<T, 1>& loc_twiddles,
                                                         std::size_t n_transforms, sycl::nd_item<1> it,
                                                         T_twiddles twiddles, T scaling_factor) {
  constexpr int N_reals_per_wi = 2 * factor_wi;

  T priv[N_reals_per_wi];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t workgroup_local_id = it.get_local_id(0);
  std::size_t workgroup_size = it.get_local_range(0);
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  std::size_t subgroup_id = sg.get_group_id();
  constexpr std::size_t n_sgs_in_wg = SYCLFFT_SGS_IN_WG;
  std::size_t id_of_sg_in_kernel = subgroup_id + it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  int n_ffts_per_sg = subgroup_size / factor_sg;
  int max_wis_working = n_ffts_per_sg * factor_sg;
  int n_reals_per_fft = factor_sg * N_reals_per_wi;
  int n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  int id_of_fft_in_sg = subgroup_local_id / factor_sg;
  std::size_t id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
  std::size_t n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  int id_of_wi_in_fft = subgroup_local_id % factor_sg;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      roundUpToMultiple<size_t>(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  global2local<false>(twiddles, loc_twiddles, N_reals_per_wi * factor_sg, workgroup_size, workgroup_local_id);
  sycl::group_barrier(it.get_group());

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    int n_ffts_worked_on_by_sg = sycl::min(static_cast<int>(n_transforms - (i - id_of_fft_in_sg)), n_ffts_per_sg);

    global2local<true>(input, loc, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_size, subgroup_local_id,
                       n_reals_per_fft * (i - id_of_fft_in_sg), subgroup_id * n_reals_per_sg);

    sycl::group_barrier(sg);
    if (working) {
      local2private<N_reals_per_wi, true>(loc, priv, subgroup_local_id, N_reals_per_wi, subgroup_id * n_reals_per_sg);
    }
    sg_dft<dir, factor_wi, factor_sg>(priv, sg, loc_twiddles);
    unrolled_loop<0, N_reals_per_wi, 2>([&](const int i) __attribute__((always_inline)) {
      priv[i] *= scaling_factor;
      priv[i + 1] *= scaling_factor;
    });
    if (working) {
      private2local_transposed<N_reals_per_wi, true>(priv, loc, id_of_wi_in_fft, factor_sg,
                                                     subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
    }
    sycl::group_barrier(sg);

    local2global<true>(loc, output, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_size, subgroup_local_id,
                       subgroup_id * n_reals_per_sg, n_reals_per_fft * (i - id_of_fft_in_sg));

    sycl::group_barrier(sg);
  }
}

/**
 * @brief Entire workgroup calculates an one FFT.
 *
 * @tparam dir Direction of the FFT
 * @tparam fft_size size of the fft_problem'
 * @tparam N First Factor
 * @tparam M Second Factor
 * @tparam loc_size local memory size in bytes
 * @tparam T_in
 * @tparam T_out
 * @tparam T
 * @tparam T_twiddles
 */
template <direction dir, int fft_size, typename T_in, typename T_out, typename T, typename T_twiddles>
__attribute__((always_inline)) inline void workgroup_impl(
    T_in input, T_out output, const sycl::local_accessor<T, 1>& loc, const sycl::local_accessor<T, 1>& loc_twiddles,
    std::size_t n_transforms, sycl::nd_item<1> it,
    T_twiddles twiddles, T scaling_factor) {
  constexpr int N = detail::factorize(fft_size);
  constexpr int M = fft_size / N;
  constexpr int fact_sg_N = detail::factorize_sg(N, SYCLFFT_TARGET_SUBGROUP_SIZE);
  constexpr int fact_wi_N = N / fact_sg_N;
  constexpr int fact_sg_M = detail::factorize_sg(M, SYCLFFT_TARGET_SUBGROUP_SIZE);
  constexpr int fact_wi_M = M / fact_sg_M;
  constexpr int sg_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
  constexpr int m_ffts_in_sg = sg_size / fact_sg_M;
  constexpr int n_ffts_in_sg = sg_size / fact_sg_N;
  constexpr int m_reals_per_fft = 2 * M;
  constexpr int n_reals_per_fft = 2 * N;
  constexpr int num_threads_per_fft_in_sg_m = m_ffts_in_sg / SYCLFFT_TARGET_SUBGROUP_SIZE;
  constexpr int num_threads_per_fft_in_sg_n = n_ffts_in_sg / SYCLFFT_TARGET_SUBGROUP_SIZE;
  constexpr int private_mem_size = fact_wi_M > fact_wi_N ? 2 * fact_wi_M : 2 * fact_wi_N;
  T priv[private_mem_size];
  T scratch[private_mem_size];

  sycl::sub_group sg = it.get_sub_group();
  int workgroup_size = it.get_local_range(0);
  int num_workgroups = it.get_group_range(0);
  int sg_id = sg.get_group_id();
  int num_sgs = workgroup_size / sg_size;
  int wg_id = it.get_group(0);
  int id_of_thread_in_wg = it.get_local_linear_id();
  int id_of_thread_in_sg = sg.get_local_linear_id();

  int max_global_offset = 2 * (n_transforms - 1) * fft_size;
  int global_offset = 2 * fft_size * wg_id;
  int offset_increment = 2 * fft_size * num_workgroups;

  constexpr int max_working_tid_in_sg_m = m_ffts_in_sg * fact_sg_M;
  constexpr int max_working_tid_in_sg_n = n_ffts_in_sg * fact_sg_N;

  int m_sg_offset = sg_id * m_ffts_in_sg + sg.get_local_linear_id() / fact_sg_M;
  int m_sg_increment = num_sgs * m_ffts_in_sg;
  int max_m_sg_offset =
      detail::roundUpToMultiple<size_t>(N, m_ffts_in_sg) + (sg.get_local_linear_id() >= max_working_tid_in_sg_m);

  int n_sg_offset = sg_id * n_ffts_in_sg + sg.get_local_linear_id() / fact_sg_N;
  int n_sg_increment = num_sgs * n_ffts_in_sg;
  int max_n_sg_offset =
      detail::roundUpToMultiple<size_t>(M, n_ffts_in_sg) + (sg.get_local_linear_id() >= max_working_tid_in_sg_m);

  global2local<true>(twiddles, loc_twiddles, 2 * (M + N), workgroup_size, id_of_thread_in_wg);
  sycl::group_barrier(it.get_group());

  for (int offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    global2local<true>(input, loc, 2 * fft_size, workgroup_size, id_of_thread_in_wg, offset);
    sycl::group_barrier(it.get_group());
    wg_dft<dir, fact_wi_M, fact_sg_M, fact_wi_N, fact_sg_N, m_ffts_in_sg, n_ffts_in_sg, fft_size, N, M>(
        priv, scratch, loc, loc_twiddles, it, m_sg_offset, max_m_sg_offset, m_sg_increment, n_sg_offset,
        max_n_sg_offset, n_sg_increment, num_threads_per_fft_in_sg_m, scaling_factor, out);
    sycl::group_barrier(it.get_group());
    local2global<true>(loc, output, 2 * fft_size, workgroup_size, id_of_thread_in_wg, 0, offset);
    sycl::group_barrier(it.get_group());
  }
}

/**
 * Selects appropriate template instantiation of workitem implementations for
 * given size of DFT.
 *
 * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam factor_sg factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
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
template <direction dir, typename T_in, typename T_out, typename T>
__attribute__((always_inline)) inline void workitem_dispatcher(T_in input, T_out output,
                                                               const sycl::local_accessor<T, 1>& loc,
                                                               std::size_t fft_size, std::size_t n_transforms,
                                                               sycl::nd_item<1> it, T scaling_factor) {
  switch (fft_size) {
#define SYCL_FFT_WI_DISPATCHER_IMPL(N)                                             \
  case N:                                                                          \
    if constexpr (fits_in_wi<T>(N)) {                                              \
      workitem_impl<dir, N>(input, output, loc, n_transforms, it, scaling_factor); \
    }                                                                              \
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
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
 * @tparam T type of the scalar used for computations
 * @tparam T_twiddles pointer containing the twiddles
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
template <direction dir, int factor_wi, typename T_in, typename T_out, typename T, typename T_twiddles>
__attribute__((always_inline)) void cross_sg_dispatcher(int factor_sg, T_in input, T_out output,
                                                        const sycl::local_accessor<T, 1>& loc,
                                                        const sycl::local_accessor<T, 1>& loc_twiddles,
                                                        std::size_t n_transforms, sycl::nd_item<1> it,
                                                        T_twiddles twiddles, T scaling_factor) {
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
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
 * @tparam T type of the scalar used for computations
 * @tparam T_twiddles type of the accessor or pointer to global memory
 * containing twiddle factors
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
template <direction dir, typename T_in, typename T_out, typename T, typename T_twiddles>
__attribute__((always_inline)) inline void subgroup_dispatcher(int factor_wi, int factor_sg, T_in input, T_out output,
                                                               const sycl::local_accessor<T, 1>& loc,
                                                               const sycl::local_accessor<T, 1>& loc_twiddles,
                                                               std::size_t n_transforms, sycl::nd_item<1> it,
                                                               T_twiddles twiddles, T scaling_factor) {
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

template <direction dir, typename T_in, typename T_out, typename T, typename T_twiddles>
__attribute__((always_inline)) inline void workgroup_dispatcher(
    T_in input, T_out output, int fft_size, const sycl::local_accessor<T, 1>& loc,
    const sycl::local_accessor<T, 1>& loc_twiddles,
    std::size_t n_transforms, sycl::nd_item<1> it, T_twiddles twiddles, T scaling_factor) {
  switch (fft_size) {
#define SYCL_FFT_WG_DISPATCHER_IMPL(N)                                                                         \
  case N:                                                                                                      \
    \            
    workgroup_impl<dir, N>(input, output, loc, loc_twiddles, scratch_loc_accessor, n_transforms, it, twiddles, \
                           scaling_factor, out);                                                               \
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
 * @tparam T_in type of the accessor or pointer to global memory containing
 * input data
 * @tparam T_out type of the accessor or pointer to global memory for output
 * data
 * @tparam T type of the scalar used for computations
 * @tparam T_twiddles type of the accessor or pointer to global memory
 * containing twiddle factors
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
template <direction dir, typename T_in, typename T_out, typename T, typename T_twiddles>
void dispatcher(T_in input, T_out output, const sycl::local_accessor<T, 1>& loc,
                const sycl::local_accessor<T, 1>& loc_twiddles,
                std::size_t fft_size, std::size_t n_transforms, sycl::nd_item<1> it, T_twiddles twiddles,
                T scaling_factor) {
  // TODO: should decision which implementation to use and factorization be done
  // on host?
  if (fits_in_wi_device<T>(fft_size)) {
    workitem_dispatcher<dir>(input, output, loc, fft_size, n_transforms, it, scaling_factor);
  } else {
    int factor_sg = detail::factorize_sg(fft_size, it.get_sub_group().get_local_linear_range());
    int factor_wi = fft_size / factor_sg;
    if (fits_in_wi_device<T>(factor_wi)) {
      subgroup_dispatcher<dir>(factor_wi, factor_sg, input, output, loc, loc_twiddles, n_transforms, it, twiddles,
                               scaling_factor);
    } else {
      workgroup_dispatcher<dir>(input, output, fft_size, loc, loc_twiddles, loc_scratch_space, n_transforms, it,
                                twiddles, scaling_factor);
    }
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
T* calculate_twiddles(std::size_t fft_size, sycl::queue& q, std::size_t subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return nullptr;
  } else {
    std::size_t factor_sg = detail::factorize_sg(fft_size, subgroup_size);
    std::size_t factor_wi = fft_size / factor_sg;
    if (fits_in_wi<T>(factor_wi)) {
      T* res = sycl::malloc_device<T>(fft_size * 2, q);
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>({factor_sg, factor_wi}), [=](sycl::item<2> it) {
          int n = it.get_id(0);
          int k = it.get_id(1);
          sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
        });
      });
      q.wait();  // waiting once here can be better than depending on the event
                 // for all future calls to compute
      return res;
    } else {
      std::size_t N = detail::factorize(fft_size);
      std::size_t M = fft_size / N;
      std::size_t factor_sg_M = detail::factorize_sg(M, subgroup_size);
      std::size_t factor_wi_M = N / factor_sg;
      if (!fits_in_wi<T>(factor_wi_M)) {
        throw std::runtime_error("FFT size " + std::to_string(N) + " is not supported for subgroup_size " +
                                 std::to_string(subgroup_size));
      }
      std::size_t factor_sg_N = detail::factorize_sg(N, subgroup_size);
      std::size_t factor_wi_N = N / factor_sg;
      if (!fits_in_wi<T>(factor_wi_N)) {
        throw std::runtime_error("FFT size " + std::to_string(N) + " is not supported for subgroup_size " +
                                 std::to_string(subgroup_size));
      }
      T* res = sycl::malloc_device<T>(2 * (M + N), q);

      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>({factor_sg_M, factor_wi_M}), [=](sycl::item<2> it) {
          int n = it.get_id(0);
          int k = it.get_id(1);
          sg_calc_twiddles(factor_sg_M, factor_wi_M, n, k, res);
        });
      });
      q.wait();
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>({factor_sg_N, factor_wi_N}), [=](sycl::item<2> it) {
          int n = it.get_id(0);
          int k = it.get_id(1);
          sg_calc_twiddles(factor_sg_N, factor_wi_N, n, k, res + (2 * M));
        });
      });
      q.wait();
      return res;
    }
  }
}

/**
 * Calculates the amount of local memory needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param subgroup_size size of subgroup used by the compute kernel
 * @return int number of elements of size T that need to fit into local memory
 */
template <typename T>
int num_scalars_in_local_mem(std::size_t fft_size, std::size_t subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return detail::pad_local(2 * fft_size * subgroup_size) * SYCLFFT_SGS_IN_WG;
  } else {
    int factor_sg = detail::factorize_sg(fft_size, subgroup_size);
    int n_ffts_per_sg = subgroup_size / factor_sg;
    return detail::pad_local(2 * fft_size * n_ffts_per_sg);
  }
}

template <typename T>
int num_scalars_in_twiddles(std::size_t fft_size, std::size_t subgroup_size) {
  if (fits_in_wi<T>(fft_size)) {
    return 1;
  } else {
    int factor_sg = detail::factorize_sg(fft_size, subgroup_size);
    if (fits_in_wi<T>(fft_size / factor_sg)) {
      return 2 * fft_size;
    } else {
      int N = detail::factorize(fft_size);
      int M = fft_size / N;
      return 2 * (M + N);
    }
  }
}

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param n_compute_units number fo compute units on target device
 * @return int number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t get_global_size(std::size_t fft_size, std::size_t n_transforms, std::size_t subgroup_size,
                            size_t n_compute_units) {
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t n_sgs_we_can_utilize;
  if (fits_in_wi<T>(fft_size)) {
    n_sgs_we_can_utilize = (n_transforms + subgroup_size - 1) / subgroup_size;
  } else {
    int factor_sg = detail::factorize_sg(fft_size, subgroup_size);
    std::size_t n_ffts_per_sg = subgroup_size / factor_sg;
    n_sgs_we_can_utilize = divideCeil(n_transforms, n_ffts_per_sg);
    // Less subgroups launched seems to be optimal for subgroup implementation.
    // This is a temporary solution until we have tunning
    maximum_n_sgs /= 4;
  }
  return subgroup_size *
         detail::roundUpToMultiple<std::size_t>(std::min(maximum_n_sgs, n_sgs_we_can_utilize), SYCLFFT_SGS_IN_WG);
}
}  // namespace detail
}  // namespace sycl_fft

#endif
