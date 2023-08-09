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

#ifndef PORTFFT_COMMON_WORKGROUP_HPP
#define PORTFFT_COMMON_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

namespace portfft {

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
template <direction Dir, detail::transpose TransposeIn, int FFTSize, int N, int M, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor, std::size_t max_num_batches_in_local_mem,
                                                  std::size_t current_batch) {
  constexpr int FactSgN = detail::factorize_sg(N, SubgroupSize);
  constexpr int FactWiN = N / FactSgN;
  constexpr int FactSgM = detail::factorize_sg(M, SubgroupSize);
  constexpr int FactWiM = M / FactSgM;
  constexpr int PrivateMemSize = FactWiM > FactWiN ? 2 * FactWiM : 2 * FactWiN;
  T priv[PrivateMemSize];

  sycl::sub_group sg = it.get_sub_group();
  constexpr int MFftsInSg = SubgroupSize / FactSgM;
  constexpr int NFftsInSg = SubgroupSize / FactSgN;
  int sg_id = static_cast<int>(sg.get_group_id());
  int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;

  constexpr int MaxWorkingTidInSgM = MFftsInSg * FactSgM;
  constexpr int MaxWorkingTidInSgN = NFftsInSg * FactSgN;

  int m_sg_offset = sg_id * MFftsInSg + static_cast<int>(sg.get_local_linear_id()) / FactSgM;
  int m_sg_increment = num_sgs * MFftsInSg;
  int max_m_sg_offset = detail::round_up_to_multiple<int>(N, MFftsInSg) +
                        (static_cast<int>(sg.get_local_linear_id()) >= MaxWorkingTidInSgM);

  int n_sg_offset = sg_id * NFftsInSg + static_cast<int>(sg.get_local_linear_id()) / FactSgN;
  int n_sg_increment = num_sgs * NFftsInSg;
  int max_n_sg_offset = detail::round_up_to_multiple<int>(M, NFftsInSg) +
                        (static_cast<int>(sg.get_local_linear_id()) >= MaxWorkingTidInSgN);

  for (int sub_batch = n_sg_offset; sub_batch < max_n_sg_offset; sub_batch += n_sg_increment) {
    bool working = sub_batch < M && static_cast<int>(sg.get_local_linear_id()) < MaxWorkingTidInSgN;
    if (working) {
      if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
        local2private_transposed<FactWiN, detail::pad::DO_PAD>(
            loc, priv, static_cast<int>(sg.get_local_linear_id()) % FactSgN, sub_batch, M);
      } else {
        detail::unrolled_loop<0, FactWiN, 1>([&](const std::size_t j) {
          priv[j] = loc[detail::pad_local(2 * max_num_batches_in_local_mem *
                                              (j * FactSgN + sg.get_local_linear_id() % FactSgN) +
                                          2 * current_batch)];
          priv[j + 1] = loc[detail::pad_local(2 * max_num_batches_in_local_mem *
                                                  (j * FactSgN + sg.get_local_linear_id() % FactSgN) +
                                              2 * current_batch + 1)];
        });
      }
    }
    sg_dft<Dir, FactWiN, FactSgN>(priv, sg, loc_twiddles + (2 * M));
    if (working) {
      if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
        private2local_transposed<FactWiN, detail::pad::DO_PAD>(
            priv, loc, static_cast<int>(sg.get_local_linear_id()) % FactSgN, FactSgN, sub_batch, M);
      } else {
        detail::unrolled_loop<0, FactWiN, 1>([&](const std::size_t j) {
          loc[detail::pad_local(2 * max_num_batches_in_local_mem * (j * FactSgN + sg.get_local_linear_id() % FactSgN) +
                                2 * current_batch)] = priv[j];
          loc[detail::pad_local(2 * max_num_batches_in_local_mem * (j * FactSgN + sg.get_local_linear_id() % FactSgN) +
                                2 * current_batch + 1)] = priv[j + 1];
        });
      }
    }
  }

  sycl::group_barrier(it.get_group());
  for (int sub_batch = m_sg_offset; sub_batch < max_m_sg_offset; sub_batch += m_sg_increment) {
    bool working = sub_batch < N && sg.get_local_linear_id() < MaxWorkingTidInSgM;
    if (working) {
      if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
        local2private<2 * FactWiM, detail::pad::DO_PAD>(
            loc, priv, sg.get_local_linear_id() % static_cast<std::size_t>(FactSgM),
            static_cast<std::size_t>(2 * FactWiM), static_cast<std::size_t>(2 * M * sub_batch));
      } else {
        local2private_transposed<FactWiM, detail::pad::DO_PAD>(loc, priv, sg.get_local_linear_id() % FactSgM, sub_batch,
                                                               max_num_batches_in_local_mem);
      }
    }
    detail::unrolled_loop<0, FactWiM, 1>([&](const int i) __attribute__((always_inline)) {
      int twiddle_n_index = sub_batch;
      int twiddle_m_index = (static_cast<int>(sg.get_local_linear_id()) % FactSgM) * FactWiM + i;
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

    sg_dft<Dir, FactWiM, FactSgM>(priv, sg, loc_twiddles);
    detail::unrolled_loop<0, FactWiM, 1>([&](const int i) __attribute__((always_inline)) {
      priv[2 * i] *= scaling_factor;
      priv[2 * i + 1] *= scaling_factor;
    });

    if (working) {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        private2local_transposed<FactWiM, detail::pad::DO_PAD>(priv, loc, sg.get_local_linear_id() % FactSgM, FactSgM,
                                                               sub_batch, max_num_batches_in_local_mem);
      } else {
        store_transposed<2 * FactWiM, detail::pad::DO_PAD>(
            priv, loc, sg.get_local_linear_id() % static_cast<std::size_t>(FactSgM), static_cast<std::size_t>(FactSgM),
            static_cast<std::size_t>(2 * M * sub_batch));
      }
    }
  }
}

}  // namespace portfft

#endif
