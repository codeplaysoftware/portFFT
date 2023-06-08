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

template<int num_complex_per_wi, int num_threads_per_fft, int subgroup_size, typename T_ptr>
__attribute__((always_inline)) inline void transpose(T_ptr priv, T_ptr output, sycl::sub_group sg) {

    using T = detail::remove_ptr<T_ptr>;
    int id_of_thread_in_fft = sg.get_local_linear_id() % num_threads_per_fft;
    int current_simd_lane = sg.get_local_linear_id() & (subgroup_size - 1);
    int batch_start_simd_lane = (sg.get_local_linear_id() - id_of_thread_in_fft) & (subgroup_size - 1);
    int simd_lane_relative_to_batch = id_of_thread_in_fft & (num_threads_per_fft - 1);

    detail::unrolled_loop<0, num_complex_per_wi, 1>([&](const int id_of_element_in_wi) __attribute__((always_inline)) {
      int relative_target_simd_lane =
          ((simd_lane_relative_to_batch + id_of_element_in_wi) & (num_complex_per_wi - 1)) * (num_threads_per_fft / num_complex_per_wi) +
          (simd_lane_relative_to_batch / num_complex_per_wi);
      int target_simd_lane = batch_start_simd_lane + relative_target_simd_lane;
      int store_address = (current_simd_lane + id_of_element_in_wi) & (num_complex_per_wi - 1);
      int target_address = ((num_complex_per_wi - id_of_element_in_wi) + (current_simd_lane / (num_threads_per_fft / num_complex_per_wi))) & (num_complex_per_wi - 1);
      T& real_value = priv[2*target_address];
      T& complex_value = priv[2*target_address + 1];
      output[2 * store_address] = sycl::select_from_group(sg, real_value, target_simd_lane);
      output[2 * store_address + 1] = sycl::select_from_group(sg, complex_value, target_simd_lane);
    });
}


template <direction dir, int fact_wi_M, int fact_sg_M, int fact_wi_N, int fact_sg_N, int m_ffts_in_sg, int n_ffts_in_sg,
          int fft_size, int N, int M, typename T_ptr, typename T, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void wg_dft(T_ptr priv, T_ptr scratch, const sycl::local_accessor<T, 1>& loc,
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
    if (working) local2private_transposed<fact_wi_N, M, true>(loc, priv, sg.get_local_linear_id() % fact_sg_N, sub_batch);
    sg_dft<dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles.get_pointer() + (2 * M));
    transpose<fact_wi_N, fact_sg_N, SYCLFFT_TARGET_SUBGROUP_SIZE>(priv, scratch, sg);

    detail::unrolled_loop<0, fact_wi_N, 1>([&](const int i) __attribute__((always_inline)) {
      T& curr_real = scratch[2 * i];
      T& curr_imag = scratch[2 * i + 1];
      
      // TODO: L2 cache latency vs sin,cos (compare SFU latency vs sequence of FFMAD).
      T twiddle_m_index = sub_batch;
      T twiddle_n_index = (sg.get_local_linear_id() % fact_sg_N) * fact_wi_N + i;
      constexpr T MINUS_TWO_PI = -2 * M_PI;
      T twiddle_real = sycl::cos((MINUS_TWO_PI * twiddle_n_index * twiddle_m_index) / fft_size);
      T twiddle_imag = sycl::sin((MINUS_TWO_PI * twiddle_n_index * twiddle_m_index) / fft_size);
      if (dir == direction::BACKWARD) twiddle_imag = -twiddle_imag;
      T tmp_real = scratch[2 * i];
      curr_real = tmp_real * twiddle_real - curr_imag * twiddle_imag;
      curr_imag = tmp_real * twiddle_imag + curr_imag * twiddle_real;
    });

    if (working) private2local_transposed<fact_wi_N, M, true>(loc, scratch, sg.get_local_linear_id() % fact_sg_N, sub_batch);
  }

  sycl::group_barrier(it.get_group());

  for (int sub_batch = m_sg_offset; sub_batch <= max_m_sg_offset; sub_batch += m_sg_increment) {
    bool working = sub_batch < N && sg.get_local_linear_id() < max_working_tid_in_sg_m;
    if (working)
      local2private<2 * fact_wi_M, true>(loc, priv, sg.get_local_linear_id(), 2 * fact_wi_M, 2 * M * sub_batch);

    sg_dft<dir, fact_wi_M, fact_sg_M>(priv, sg, loc_twiddles);
    transpose<fact_wi_M, fact_sg_M, SYCLFFT_TARGET_SUBGROUP_SIZE>(priv, scratch, sg); 

    detail::unrolled_loop<0, 2 * fact_wi_M, 2>([&](const int i) __attribute__((always_inline)) {
      T& curr_real = scratch[2 * i];
      T& curr_imag = scratch[2 * i + 1];

      curr_real *= scaling_factor;
      curr_imag *= scaling_factor;
    });
    if (working)
      private2local_transposed<fact_wi_M, M, true>(loc, priv, sg.get_local_linear_id() % fact_sg_M, sub_batch);
  }
}

}  // namespace sycl_fft
#endif