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

#ifndef SYCL_FFT_COMMON_WORKGROUP_HPP
#define SYCL_FFT_COMMON_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

namespace sycl_fft {

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam FFTSize Problem Size
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam T Scalar Type
 * @tparam T_twiddles_ptr Type of twiddle pointer utilized by subgroup ffts
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 */
template <direction Dir, int FFTSize, int N, int M, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor) {
  constexpr int fact_sg_N = detail::factorize_sg(N, SubgroupSize);
  constexpr int fact_wi_N = N / fact_sg_N;
  constexpr int fact_sg_M = detail::factorize_sg(M, SubgroupSize);
  constexpr int fact_wi_M = M / fact_sg_M;
  constexpr int private_mem_size = fact_wi_M > fact_wi_N ? 2 * fact_wi_M : 2 * fact_wi_N;
  T priv[private_mem_size];

  sycl::sub_group sg = it.get_sub_group();
  constexpr int m_ffts_in_sg = SubgroupSize / fact_sg_M;
  constexpr int n_ffts_in_sg = SubgroupSize / fact_sg_N;
  int sg_id = static_cast<int>(sg.get_group_id());
  int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;

  constexpr int max_working_tid_in_sg_m = m_ffts_in_sg * fact_sg_M;
  constexpr int max_working_tid_in_sg_n = n_ffts_in_sg * fact_sg_N;

  int m_sg_offset = sg_id * m_ffts_in_sg + static_cast<int>(sg.get_local_linear_id()) / fact_sg_M;
  int m_sg_increment = num_sgs * m_ffts_in_sg;
  int max_m_sg_offset = detail::roundUpToMultiple<int>(N, m_ffts_in_sg) +
                        (static_cast<int>(sg.get_local_linear_id()) >= max_working_tid_in_sg_m);

  int n_sg_offset = sg_id * n_ffts_in_sg + static_cast<int>(sg.get_local_linear_id()) / fact_sg_N;
  int n_sg_increment = num_sgs * n_ffts_in_sg;
  int max_n_sg_offset = detail::roundUpToMultiple<int>(M, n_ffts_in_sg) +
                        (static_cast<int>(sg.get_local_linear_id()) >= max_working_tid_in_sg_n);

  for (int sub_batch = n_sg_offset; sub_batch <= max_n_sg_offset; sub_batch += n_sg_increment) {
    bool working = sub_batch < M && static_cast<int>(sg.get_local_linear_id()) < max_working_tid_in_sg_n;
    if (working) {
      local2private_transposed<fact_wi_N, M, detail::pad::DO_PAD>(
          loc, priv, static_cast<int>(sg.get_local_linear_id()) % fact_sg_N, sub_batch);
    }
    sg_dft<Dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles + (2 * M));
    if (working) {
      private2local_transposed<fact_wi_N, M, detail::pad::DO_PAD>(
          priv, loc, static_cast<int>(sg.get_local_linear_id()) % fact_sg_N, fact_sg_N, sub_batch);
    }
  }

  sycl::group_barrier(it.get_group());
  for (int sub_batch = m_sg_offset; sub_batch <= max_m_sg_offset; sub_batch += m_sg_increment) {
    bool working = sub_batch < N && sg.get_local_linear_id() < max_working_tid_in_sg_m;
    if (working) {
      local2private<2 * fact_wi_M, detail::pad::DO_PAD>(
          loc, priv, sg.get_local_linear_id() % static_cast<std::size_t>(fact_sg_M),
          static_cast<std::size_t>(2 * fact_wi_M), static_cast<std::size_t>(2 * M * sub_batch));
    }
    detail::unrolled_loop<0, fact_wi_M, 1>([&](const int i) __attribute__((always_inline)) {
      int twiddle_n_index = sub_batch;
      int twiddle_m_index = (static_cast<int>(sg.get_local_linear_id()) % fact_sg_M) * fact_wi_M + i;
      int twiddle_index = 2 * M * twiddle_n_index + (2 * twiddle_m_index);
      T twiddle_real = wg_twiddles[twiddle_index];
      T twiddle_imag = wg_twiddles[twiddle_index + 1];
      if constexpr (Dir == direction::BACKWARD) {
        twiddle_imag = -twiddle_imag;
      }
      T tmp_real = priv[2 * i];
      priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
      priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
    });

    sg_dft<Dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles);
    detail::unrolled_loop<0, fact_wi_M, 1>([&](const int i) __attribute__((always_inline)) {
      priv[2 * i] *= scaling_factor;
      priv[2 * i + 1] *= scaling_factor;
    });

    if (working) {
      store_transposed<2 * fact_wi_M, detail::pad::DO_PAD>(
          priv, loc, sg.get_local_linear_id() % static_cast<std::size_t>(fact_sg_M),
          static_cast<std::size_t>(fact_sg_M), static_cast<std::size_t>(2 * M * sub_batch));
    }
  }
}

}  // namespace sycl_fft

#endif
