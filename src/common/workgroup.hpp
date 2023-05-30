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
// TODO: refactor code here
template <direction dir, int fact_wi_M, int fact_sg_M, int fact_wi_N, int fact_sg_N, int m_ffts_in_sg, int n_ffts_in_sg,
          int fft_size, int N, int M, typename T_ptr, typename T, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void wg_dft(T_ptr priv, const sycl::local_accessor<T, 1>& loc,
                                                  T_twiddles_ptr loc_twiddles, sycl::nd_item<1> it, int m_sg_offset,
                                                  int max_m_sg_offset, int m_sg_increment, int n_sg_offset,
                                                  int max_n_sg_offset, int n_sg_increment,
                                                  int num_threads_per_fft_in_sg_m, T scaling_factor) {
  sycl::sub_group sg = it.get_sub_group();
  constexpr int max_working_tid_in_sg_m = m_ffts_in_sg * fact_sg_M;
  constexpr int max_working_tid_in_sg_n = n_ffts_in_sg * fact_sg_N;
  constexpr int twiddle_offset_N = 2 * fft_size;
  constexpr int twiddle_offset_M = 2 * (fft_size + N);
  constexpr int m_threads_per_fft_in_sg = SYCLFFT_TARGET_SUBGROUP_SIZE / m_ffts_in_sg;

  for (int sg_m_offset = m_sg_offset; sg_m_offset <= max_m_sg_offset; sg_m_offset += m_sg_increment) {
    local2private_transposed<2 * fact_wi_M, true>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_M, sg_m_offset);

    sg_dft<dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles.get_pointer() + twiddle_offset_M);
    sycl::group_barrier(sg);

    detail::unrolled_loop<0, 2 * fact_wi_M, 2>([&](const int idx) __attribute__((always_inline)) {
      T tmp_real = priv[idx];
      int twiddle_n_idx = sg.get_group_id() * m_ffts_in_sg + sg.get_local_linear_id() / fact_sg_M;
      int twiddle_k_idx =
          (sg.get_local_linear_id() - ((sg.get_local_linear_id() / fact_sg_M) * m_threads_per_fft_in_sg)) *
              m_threads_per_fft_in_sg +
          idx;
      T twiddle_real = loc_twiddles[2 * (twiddle_n_idx * N + twiddle_k_idx)];
      T twiddle_imag = loc_twiddles[2 * (twiddle_n_idx * N + twiddle_k_idx) + 1];
      if constexpr (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;
      priv[idx] = tmp_real * twiddle_real - priv[idx + 1] * twiddle_imag;
      priv[idx + 1] = tmp_real * twiddle_imag + priv[idx + 1] * twiddle_real;
    });

    private2local_transposed<2 * fact_wi_M, true>(
        priv, loc, sg.get_local_linear_id() % fact_sg_M, fact_sg_M,
        sg_m_offset + (sg.get_local_linear_id() / fact_sg_M) * 2 * m_ffts_in_sg * M);
  }

  sycl::group_barrier(it.get_group());

  for (std::size_t sg_n_offset = n_sg_offset; sg_n_offset <= max_n_sg_offset; sg_n_offset += n_sg_increment) {
    local2private<2 * fact_wi_N, false>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_N, sg_n_offset);

    sg_dft<dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles.get_pointer() + twiddle_offset_N);
    sycl::group_barrier(sg);

    detail::unrolled_loop<0, 2 * fact_wi_N, 2>([&](const int idx) __attribute__((always_inline)) {
      priv[idx] *= scaling_factor;
      priv[idx + 1] *= scaling_factor;
    });

    private2local<2 * fact_wi_N, true>(priv, loc, sg.get_local_linear_id(), 2 * fact_wi_N, sg_n_offset);
  }
}
}  // namespace sycl_fft
#endif