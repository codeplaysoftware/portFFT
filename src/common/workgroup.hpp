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
template <direction dir, int N, int fact_sg1, int fact_wi1, int fact_sg2, int fact_wi2, int fft_size, typename T_ptr,
          typename T, typename T_twiddles_ptr>
__attribute__((always_inline)) inline void wg_dft(T_ptr priv, const sycl::local_accessor<T, 1>& loc,
                                                  T_twiddles_ptr loc_twiddles, sycl::nd_item<1> it,
                                                  int num_batches_per_sg_N, int num_batches_per_sg_M,
                                                  int n_reals_per_sg_M, int n_reals_per_sg_N, int max_wis_working_M,
                                                  int max_wis_working_N) {
  sycl::sub_group sg = it.get_sub_group();
  bool working_N = sg.get_local_linear_id() < max_wis_working_N;
  bool working_M = sg.get_local_linear_id() < max_wis_working_M;
  for (int j = 0; j < num_batches_per_sg_N; j++) {
    if (working_M)
      local2private_transposed<2 * fact_wi2, true>(
          loc, priv, sg.get_local_linear_id(), n_reals_per_sg_M,
          sg.get_group_id() * num_batches_per_sg_N * n_reals_per_sg_M + j * n_reals_per_sg_M);

    sg_dft<dir, fact_wi2, fact_sg2>(priv, sg, loc_twiddles.get_pointer() + 2 * fft_size + 2 * N);
    // multiply twiddles;
    detail::unrolled_loop<0, 2 * fact_wi2, 2>([&](const int i) {
      auto twiddle_position = N * (sg.get_group_id() * num_batches_per_sg_N + j) + sg.get_local_linear_id() * fact_wi2;
      T twiddle_real = loc_twiddles[2 * twiddle_position];
      T twiddle_imag = loc_twiddles[2 * twiddle_position + 1];
      T tmp_imag = priv[i + 1];
      priv[i + 1] = priv[i] * twiddle_imag + priv[i + 1] * twiddle_real;
      priv[i] = priv[i] * twiddle_real - tmp_imag * twiddle_imag;
    });
    private2local_transposed<2 * fact_wi2, true>(
        priv, loc, sg.get_local_linear_id(), max_wis_working_M,
        sg.get_group_id() * num_batches_per_sg_N * n_reals_per_sg_M + j * n_reals_per_sg_M);
  }
  sycl::group_barrier(it.get_group());
  for (int j = 0; j < num_batches_per_sg_M; j++) {
    if (working_M)
      local2private_transposed<2 * fact_wi1, true>(
          loc, priv, sg.get_local_linear_id(), n_reals_per_sg_N,
          sg.get_group_id() * num_batches_per_sg_M * n_reals_per_sg_N + j * n_reals_per_sg_N);

    sg_dft<dir, fact_wi1, fact_sg1>(priv, sg, loc_twiddles.get_pointer() + 2 * fft_size);
    private2local<2 * fact_wi1, true>(
        priv, loc, sg.get_local_linear_id(), max_wis_working_N,
        sg.get_group_id() * num_batches_per_sg_M * n_reals_per_sg_N + j * n_reals_per_sg_N);
  }
}

}  // namespace sycl_fft
#endif