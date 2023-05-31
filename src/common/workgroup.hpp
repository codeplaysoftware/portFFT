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
  int id_of_wi_in_fft = sg.get_local_linear_id() % fact_sg_M;

  for (int sg_m_offset = m_sg_offset; sg_m_offset <= max_m_sg_offset; sg_m_offset += m_sg_increment) {
    bool working = sg_m_offset < M && sg.get_local_linear_id() < max_working_tid_in_sg_m;

    int twiddle_n_idx = sg_m_offset  + (sg.get_local_linear_id() / fact_sg_M);
    int twiddle_k_idx = id_of_wi_in_fft * fact_wi_M;

    if(working)
      local2private<2 * fact_wi_M, true>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_M, 2 * sg_m_offset * M);

    sg_dft<dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles);
    sycl::group_barrier(sg);

    detail::unrolled_loop<0, fact_wi_M, 1>([&](const int idx) __attribute__((always_inline)) {
      constexpr T TWOPI_OVER_N = (-2 * M_PI) / (N * M);
      T twiddle_real = sycl::cos(static_cast<T>(twiddle_n_idx * (twiddle_k_idx + idx)) * TWOPI_OVER_N);
      T twiddle_imag = sycl::sin(static_cast<T>(twiddle_n_idx * (twiddle_k_idx + idx)) * TWOPI_OVER_N);
      if constexpr (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;

      T tmp_real = priv[2 * idx];

      priv[2 * idx] = tmp_real * twiddle_real - priv[2 * idx + 1] * twiddle_imag;
      priv[2 * idx + 1] = priv[2 * idx + 1] * twiddle_real + tmp_real * twiddle_imag;
    });
    if(working)
      private2local_transposed<2 * fact_wi_M, true>(priv, loc, id_of_wi_in_fft, fact_sg_M, 2 * M * sg_m_offset);
  }

  sycl::group_barrier(it.get_group());

  for (std::size_t sg_n_offset = n_sg_offset; sg_n_offset <= max_n_sg_offset; sg_n_offset += n_sg_increment) {
    bool working = sg_n_offset < N && sg.get_local_linear_id() < max_working_tid_in_sg_n;
    if(working)
      local2private<2 * fact_wi_N, true>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_N, 2 * sg_n_offset * N);

    sg_dft<dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles.get_pointer() + (2 * M));

    detail::unrolled_loop<0, 2 * fact_wi_N, 2>([&](const int idx) __attribute__((always_inline)) {
      priv[idx] *= scaling_factor;
      priv[idx + 1] *= scaling_factor;
    });
    if(working)
      private2local<2 * fact_wi_N, true>(priv, loc, sg.get_local_linear_id(), 2 * fact_wi_N, 2 * N * sg_n_offset);
  }
  sycl::group_barrier(sg);
}
}  // namespace sycl_fft
#endif