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
 * Calculate the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 * e.g. If there are 64 elements in a row, then the column values are 128 float apart.
 * There are 32 banks, each the size of a float, so we only want a padding float every 128/32=4 groups to read along the
 * column with without bank conflicts.
 *
 * @param row_size the number of complex values in a row
 * @return constexpr std::size_t the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
constexpr std::size_t bank_groups_per_pad_wg(std::size_t row_size) {
  // 2*row_size is the number of floats between each successive read for the column dfts
  // we only need 1 pad for each of those
  return (2 * row_size) / PORTFFT_N_LOCAL_BANKS;
}

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam FFTSize Problem Size
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankGroupsPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T Scalar Type
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 */
template <direction Dir, int FFTSize, int N, int M, int SubgroupSize, std::size_t BankGroupsPerPad, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor) {
  // the number of work-items involved in every row subgroup fft
  constexpr int fact_sg_N = detail::factorize_sg(N, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr int fact_wi_N = N / fact_sg_N;
  // the number of work-items involved in every column subgroup fft
  constexpr int fact_sg_M = detail::factorize_sg(M, SubgroupSize);
  // the number of values held in by a work-item in a column subgroup dft
  constexpr int fact_wi_M = M / fact_sg_M;

  constexpr int private_mem_size = fact_wi_M > fact_wi_N ? 2 * fact_wi_M : 2 * fact_wi_N;
  T priv[private_mem_size];
  const int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;

  sycl::sub_group sg = it.get_sub_group();
  {  // column ffts
    constexpr int ffts_per_sg = SubgroupSize / fact_sg_N;
    constexpr bool excess_sgs = M % ffts_per_sg > 0;
    constexpr bool excess_wis = SubgroupSize % fact_sg_N > 0;

    // only needed when there are excess work-items
    constexpr std::size_t max_working_tid_in_sg = ffts_per_sg * fact_sg_N;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / fact_sg_N;
    // id of the work-item in the fft
    const int fft_local_id = static_cast<int>(sg.get_local_linear_id()) % fact_sg_N;

    const int column_begin = static_cast<int>(sg.get_group_id()) * ffts_per_sg + fft_in_subgroup;
    const int column_step = num_sgs * ffts_per_sg;
    int column_end;
    if constexpr (excess_sgs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of ffts_per_sg anyway
      column_end = detail::roundUpToMultiple(M, ffts_per_sg);
    } else {
      column_end = M;
    }

    if constexpr (excess_wis) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      column_end += (fft_in_subgroup == ffts_per_sg) ? 1 : 0;
    }

    for (int column = column_begin; column < column_end; column += column_step) {
      bool working = true;
      if constexpr (excess_sgs) {
        working = column < M;
      }
      if constexpr (excess_wis) {
        working = working && sg.get_local_linear_id() < max_working_tid_in_sg;
      }
      if (working) {
        local2private_transposed<fact_wi_N, detail::pad::DO_PAD, BankGroupsPerPad>(loc, priv, fft_local_id, column, M);
      }
      sg_dft<Dir, fact_wi_N, fact_sg_N>(priv, sg, loc_twiddles + (2 * M));
      if (working) {
        private2local_transposed<fact_wi_N, detail::pad::DO_PAD, BankGroupsPerPad>(priv, loc, fft_local_id, fact_sg_N,
                                                                                   column, M);
      }
    }
  }

  sycl::group_barrier(it.get_group());

  {  // row ffts
    constexpr int ffts_per_sg = SubgroupSize / fact_sg_M;
    constexpr bool excess_sgs = M % ffts_per_sg > 0;
    constexpr bool excess_wis = SubgroupSize % fact_sg_M > 0;

    // only needed when there are excess work-items
    constexpr int max_working_tid_in_sg = ffts_per_sg * fact_sg_M;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / fact_sg_M;
    // id of the work-item in the fft
    const int fft_local_id = static_cast<int>(sg.get_local_linear_id()) % fact_sg_M;

    const int row_begin = static_cast<int>(sg.get_group_id()) * ffts_per_sg + fft_in_subgroup;
    const int row_step = num_sgs * ffts_per_sg;
    int row_end;
    if constexpr (excess_sgs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of ffts_per_sg anyway
      row_end = detail::roundUpToMultiple(N, ffts_per_sg);
    } else {
      row_end = N;
    }

    if constexpr (excess_wis) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      row_end += (fft_in_subgroup == ffts_per_sg) ? 1 : 0;
    }

    for (int row = row_begin; row < row_end; row += row_step) {
      bool working = true;
      if constexpr (excess_sgs) {
        working = row < N;
      }
      if constexpr (excess_wis) {
        working = working && sg.get_local_linear_id() < max_working_tid_in_sg;
      }
      if (working) {
        local2private<2 * fact_wi_M, detail::pad::DO_PAD, BankGroupsPerPad>(
            loc, priv, static_cast<std::size_t>(fft_local_id), static_cast<std::size_t>(2 * fact_wi_M),
            static_cast<std::size_t>(2 * M * row));
      }
      detail::unrolled_loop<0, fact_wi_M, 1>([&](const int i) __attribute__((always_inline)) {
        int element = fft_local_id * fact_wi_M + i;
        int twiddle_index = M * row + element;
        sycl::vec<T, 2> twiddles = reinterpret_cast<const sycl::vec<T, 2>*>(wg_twiddles)[twiddle_index];
        T twiddle_real = twiddles[0];
        T twiddle_imag = twiddles[1];
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
        store_transposed<2 * fact_wi_M, detail::pad::DO_PAD, BankGroupsPerPad>(
            priv, loc, static_cast<std::size_t>(fft_local_id), static_cast<std::size_t>(fact_sg_M),
            static_cast<std::size_t>(2 * M * row));
      }
    }
  }
}

}  // namespace portfft

#endif
