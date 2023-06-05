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

  for (int sub_batch = n_sg_offset; sub_batch <= max_n_sg_offset; sub_batch += n_sg_increment) {
    bool working = sub_batch < M && sg.get_local_linear_id() < max_working_tid_in_sg_n;
    if (working) local2private_transposed<fact_wi_N, M>(loc, priv, sg.get_local_linear_id() % fact_sg_N, sub_batch);
    sg_dft<dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles.get_pointer() + (2 * M));
    sycl::group_barrier(sg);
    detail::unrolled_loop<0, fact_wi_N, 1>([&](const int i) __attribute__((always_inline)) {
      T twiddle_m_index = sub_batch;
      T twiddle_n_index = (sg.get_local_linear_id() % fact_sg_N) * fact_wi_N + i;
      constexpr T MINUS_TWO_PI = -2 * M_PI;
      T twiddle_real = sycl::cos((MINUS_TWO_PI * twiddle_n_index * twiddle_m_index) / fft_size);
      T twiddle_imag = sycl::sin((MINUS_TWO_PI * twiddle_n_index * twiddle_m_index) / fft_size);
      if (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;
      T tmp_real = priv[2 * i];
      priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
      priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
    });
    if (working) private2local_transposed<fact_wi_N, M>(loc, priv, sg.get_local_linear_id() % fact_sg_N, sub_batch);
  }
  sycl::group_barrier(it.get_group());
  for (int sub_batch = m_sg_offset; sub_batch <= max_m_sg_offset; sub_batch += m_sg_increment) {
    bool working = sub_batch < N && sg.get_local_linear_id() < max_working_tid_in_sg_m;
    if (working)
      local2private<2 * fact_wi_M, false>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_M, 2 * M * sub_batch);
    sg_dft<dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles);
    sycl::group_barrier(sg);
    detail::unrolled_loop<0, 2 * fact_wi_M, 2>([&](const int i) __attribute__((always_online)) {
      priv[i] *= scaling_factor;
      priv[i + 1] *= scaling_factor;
    });
    if (working)
      private2local<2 * fact_wi_M, false>(priv, loc, sg.get_local_linear_id(), 2 * fact_wi_M, 2 * M * sub_batch);
  }
}
}  // namespace sycl_fft
#endif