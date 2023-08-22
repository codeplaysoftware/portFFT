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
 * Calculate the number of groups or bank lines of PORTFFT_N_LOCAL_BANKS between each padding in local memory,
 * specifically for reducing bank conflicts when reading values from the columns of a 2D data layout. e.g. If there are
 * 64 complex elements in a row, then the consecutive values in the same column are 128 floats apart. There are 32
 * banks, each the size of a float, so we only want a padding float every 128/32=4 bank lines to read along the column
 * without bank conflicts.
 *
 * @param row_size the size in bytes of the row. 32 std::complex<float> values would probably have a size of 256 bytes.
 * @return constexpr std::size_t the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
constexpr std::size_t bank_lines_per_pad_wg(std::size_t row_size) {
  constexpr std::size_t BankLineSize = sizeof(float) * PORTFFT_N_LOCAL_BANKS;
  if (row_size % BankLineSize == 0) {
    return row_size / BankLineSize;
  }
  // There is room for improvement here. E.G if row_size was half of BankLineSize then maybe you would still want 1
  // pad every bank group.
  return 0;
}

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T Scalar Type
 *
 * @param factor_n Smaller factor of the problem size. Problem size is factor_n * factor_m.
 * @param factor_m Larger factor of the problem size. Problem size is factor_n * factor_m.
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 */
template <direction Dir, int SubgroupSize, std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void wg_dft(std::size_t factor_n, std::size_t factor_m, T* loc, T* loc_twiddles,
                                                  const T* wg_twiddles, sycl::nd_item<1> it, T scaling_factor) {
  // the number of work-items involved in every row subgroup fft. Is less than SubgroupSize.
  std::size_t fact_sg_n = detail::factorize_sg(factor_n, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  std::size_t fact_wi_n = factor_n / fact_sg_n;
  // the number of work-items involved in every column subgroup fft. Is less than SubgroupSize.
  std::size_t fact_sg_m = detail::factorize_sg(factor_m, SubgroupSize);
  // the number of values held in by a work-item in a column subgroup dft
  std::size_t fact_wi_m = factor_m / fact_sg_m;

  constexpr size_t PrivateMemSize = 2 * detail::MaxFftSizeWi;
  T priv[PrivateMemSize];
  const std::size_t num_sgs = it.get_local_range(0) / SubgroupSize;

  sycl::sub_group sg = it.get_sub_group();
  {  // column ffts
    const std::size_t ffts_per_sg = SubgroupSize / fact_sg_n;
    const bool excess_wis = SubgroupSize % fact_sg_n > 0;
    const bool excess_sgs = factor_m % ffts_per_sg > 0;

    // only needed when there are excess work-items
    const std::size_t max_working_tid_in_sg = ffts_per_sg * fact_sg_n;

    const std::size_t fft_in_subgroup = sg.get_local_linear_id() / fact_sg_n;
    // id of the work-item in the fft
    const std::size_t fft_local_id = sg.get_local_linear_id() % fact_sg_n;

    const std::size_t column_begin = sg.get_group_id() * ffts_per_sg + fft_in_subgroup;
    const std::size_t column_step = num_sgs * ffts_per_sg;
    std::size_t column_end;
    if (excess_sgs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      // NOLINTNEXTLINE
      column_end = detail::round_up_to_multiple(factor_m, ffts_per_sg);
    } else {
      column_end = factor_m;
    }

    if (excess_wis) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      column_end += (fft_in_subgroup == ffts_per_sg) ? 1 : 0;
    }

    for (std::size_t column = column_begin; column < column_end; column += column_step) {
      bool working = true;
      if (excess_sgs) {
        working = column < factor_m;
      }
      if (excess_wis) {
        working = working && sg.get_local_linear_id() < max_working_tid_in_sg;
      }
      if (working) {
        local2private_transposed<detail::pad::DO_PAD, BankLinesPerPad>(fact_wi_n, loc, priv, fft_local_id, column,
                                                                       factor_m);
      }
      T wi_private_scratch[detail::wi_temps(detail::MaxFftSizeWi)];
      sg_dft<Dir>(fact_wi_n, fact_sg_n, priv, sg, loc_twiddles + (2 * factor_m), wi_private_scratch);
      if (working) {
        private2local_transposed<detail::pad::DO_PAD, BankLinesPerPad>(fact_wi_n, priv, loc, fft_local_id, fact_sg_n,
                                                                       column, factor_m);
      }
    }
  }

  sycl::group_barrier(it.get_group());

  {  // row ffts
    const std::size_t ffts_per_sg = SubgroupSize / fact_sg_m;
    const bool excess_wis = SubgroupSize % fact_sg_m > 0;
    const bool excess_sgs = factor_n % ffts_per_sg > 0;

    // only needed when there are excess work-items
    const std::size_t max_working_tid_in_sg = ffts_per_sg * fact_sg_m;

    const std::size_t fft_in_subgroup = sg.get_local_linear_id() / fact_sg_m;
    // id of the work-item in the fft
    const std::size_t fft_local_id = sg.get_local_linear_id() % fact_sg_m;

    const std::size_t row_begin = sg.get_group_id() * ffts_per_sg + fft_in_subgroup;
    const std::size_t row_step = num_sgs * ffts_per_sg;
    std::size_t row_end;
    if (excess_sgs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      // NOLINTNEXTLINE
      row_end = detail::round_up_to_multiple(factor_n, ffts_per_sg);
    } else {
      row_end = factor_n;
    }

    if (excess_wis) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      row_end += (fft_in_subgroup == ffts_per_sg) ? 1 : 0;
    }

    for (std::size_t row = row_begin; row < row_end; row += row_step) {
      bool working = true;
      if (excess_sgs) {
        working = row < factor_n;
      }
      if (excess_wis) {
        working = working && sg.get_local_linear_id() < max_working_tid_in_sg;
      }
      if (working) {
        local2private<detail::pad::DO_PAD, BankLinesPerPad>(2 * fact_wi_m, loc, priv, fft_local_id, 2 * fact_wi_m,
                                                            2 * factor_m * row);
      }
#pragma clang loop unroll(full)
      for (std::size_t i{0}; i < fact_wi_m; ++i) {
        std::size_t element = fft_local_id * fact_wi_m + i;
        std::size_t twiddle_index = factor_m * row + element;
        sycl::vec<T, 2> twiddles = reinterpret_cast<const sycl::vec<T, 2>*>(wg_twiddles)[twiddle_index];
        T twiddle_real = twiddles[0];
        T twiddle_imag = twiddles[1];
        if constexpr (Dir == direction::BACKWARD) {
          twiddle_imag = -twiddle_imag;
        }
        T tmp_real = priv[2 * i];
        priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
        priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
      }

      T wi_private_scratch[2 * detail::wi_temps(detail::MaxFftSizeWi)];
      sg_dft<Dir>(fact_wi_m, fact_sg_m, priv, sg, loc_twiddles, wi_private_scratch);
#pragma clang loop unroll(full)
      for (std::size_t i{0}; i < fact_wi_m; ++i) {
        priv[2 * i] *= scaling_factor;
        priv[2 * i + 1] *= scaling_factor;
      }

      if (working) {
        store_transposed<detail::pad::DO_PAD, BankLinesPerPad>(2 * fact_wi_m, priv, loc, fft_local_id, fact_sg_m,
                                                               2 * factor_m * row);
      }
    }
  }
}

}  // namespace portfft

#endif
