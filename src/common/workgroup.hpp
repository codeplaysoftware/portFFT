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
 * @tparam TransposeIn Whether or not the input is transposed
 * @tparam FFTSize Problem Size
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankLinesPerPad the number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad.
 * @tparam T Scalar Type
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 * @param max_num_batches_in_local_mem Maximum possible number of batches in local memory
 * @param sub_batch_num Batch that is stored in the local memory currently being computed
 */
template <direction Dir, detail::transpose TransposeIn, int FFTSize, int N, int M, int SubgroupSize,
          std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor, std::size_t max_num_batches_in_local_mem,
                                                  std::size_t sub_batch_num) {
  // the number of work-items involved in every row subgroup fft
  constexpr int FactSgN = detail::factorize_sg(N, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr int FactWiN = N / FactSgN;
  // the number of work-items involved in every column subgroup fft
  constexpr int FactSgM = detail::factorize_sg(M, SubgroupSize);
  // the number of values held in by a work-item in a column subgroup dft
  constexpr int FactWiM = M / FactSgM;

  constexpr int PrivateMemSize = FactWiM > FactWiN ? 2 * FactWiM : 2 * FactWiN;
  T priv[PrivateMemSize];
  const int num_sgs = static_cast<int>(it.get_local_range(0)) / SubgroupSize;

  sycl::sub_group sg = it.get_sub_group();
  {  // column ffts
    constexpr int FFTsPerSG = SubgroupSize / FactSgN;
    constexpr bool ExcessWIs = SubgroupSize % FactSgN > 0;
    constexpr bool ExcessSGs = M % FFTsPerSG > 0;

    // only needed when there are excess work-items
    constexpr std::size_t MaxWorkingTidInSg = FFTsPerSG * FactSgN;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / FactSgN;
    // id of the work-item in the fft
    const int wi_in_fft = static_cast<int>(sg.get_local_linear_id()) % FactSgN;

    const int column_begin = static_cast<int>(sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
    const int column_step = num_sgs * FFTsPerSG;
    int column_end;
    if constexpr (ExcessSGs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      column_end = detail::round_up_to_multiple(M, FFTsPerSG);
    } else {
      column_end = M;
    }

    if constexpr (ExcessWIs) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      column_end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
    }

    for (int column = column_begin; column < column_end; column += column_step) {
      bool working = true;
      if constexpr (ExcessSGs) {
        working = column < M;
      }
      if constexpr (ExcessWIs) {
        working = working && sg.get_local_linear_id() < MaxWorkingTidInSg;
      }
      if (working) {
        if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
          /**
           * Load data from the column corresponsing to the sub_batch_being computed,
           * in a transposed fashion, viewing each column as N x M Matrix.
           */
          detail::unrolled_loop<0, FactWiN, 1>([&](const int j) __attribute__((always_inline)) {
            std::size_t base_offset =
                2 * max_num_batches_in_local_mem * static_cast<std::size_t>((M * (wi_in_fft * FactWiN + j)) + column) +
                2 * sub_batch_num;
            priv[2 * j] = loc[detail::pad_local(base_offset, BankLinesPerPad)];
            priv[2 * j + 1] = loc[detail::pad_local(base_offset + 1, BankLinesPerPad)];
          });
        } else {
          local2private_transposed<FactWiN, detail::pad::DO_PAD, BankLinesPerPad>(loc, priv, wi_in_fft, column, M);
        }
      }
      sg_dft<Dir, FactWiN, FactSgN>(priv, sg, loc_twiddles + (2 * M));
      if (working) {
        if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
          /**
           * Store back the  data to the column corresponsing to the sub_batch_being computed,
           * in a transposed fashion, viewing each column as N x M Matrix, given the result from
           * sg_dft is also transposed in the registers.
           */
          detail::unrolled_loop<0, FactWiN, 1>([&](const int j) __attribute__((always_inline)) {
            std::size_t base_offset =
                2 * max_num_batches_in_local_mem * static_cast<std::size_t>((M * (j * FactSgN + wi_in_fft) + column)) +
                2 * sub_batch_num;
            loc[detail::pad_local(base_offset, BankLinesPerPad)] = priv[2 * j];
            loc[detail::pad_local(base_offset + 1, BankLinesPerPad)] = priv[2 * j + 1];
          });
        } else {
          private2local_transposed<FactWiN, detail::pad::DO_PAD, BankLinesPerPad>(priv, loc, wi_in_fft, FactSgN, column,
                                                                                  M);
        }
      }
    }
  }

  sycl::group_barrier(it.get_group());
  {  // row ffts
    constexpr int FFTsPerSG = SubgroupSize / FactSgM;
    constexpr bool ExcessWIs = SubgroupSize % FactSgM > 0;
    constexpr bool ExcessSGs = N % FFTsPerSG > 0;

    // only needed when there are excess work-items
    constexpr int MaxWorkingTidInSg = FFTsPerSG * FactSgM;

    const int fft_in_subgroup = static_cast<int>(sg.get_local_linear_id()) / FactSgM;
    // id of the work-item in the fft
    const int wi_in_fft = static_cast<int>(sg.get_local_linear_id()) % FactSgM;

    const int row_begin = static_cast<int>(sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
    const int row_step = num_sgs * FFTsPerSG;
    int row_end;
    if constexpr (ExcessSGs) {
      // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
      // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
      row_end = detail::round_up_to_multiple(N, FFTsPerSG);
    } else {
      row_end = N;
    }

    if constexpr (ExcessWIs) {
      // also allow these work-items to enter the loop, without making other work-items do another loop.
      row_end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
    }

    for (int row = row_begin; row < row_end; row += row_step) {
      bool working = true;
      if constexpr (ExcessSGs) {
        working = row < N;
      }
      if constexpr (ExcessWIs) {
        working = working && sg.get_local_linear_id() < MaxWorkingTidInSg;
      }
      if (working) {
        if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
          /**
           * Load FactWiM contiguous elements per column corresponding to the sub batch being processed.
           */
          detail::unrolled_loop<0, FactWiM, 1>([&](const int j) __attribute__((always_inline)) {
            std::size_t base_index =
                2 * max_num_batches_in_local_mem * static_cast<std::size_t>((row * M + wi_in_fft * FactWiM + j)) +
                2 * sub_batch_num;
            priv[2 * j] = loc[detail::pad_local(base_index, BankLinesPerPad)];
            priv[2 * j + 1] = loc[detail::pad_local(base_index + 1, BankLinesPerPad)];
          });
        } else {
          local2private<2 * FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(
              loc, priv, static_cast<std::size_t>(wi_in_fft), static_cast<std::size_t>(2 * FactWiM),
              static_cast<std::size_t>(2 * M * row));
        }
      }
      detail::unrolled_loop<0, FactWiM, 1>([&](const int i) __attribute__((always_inline)) {
        int element = wi_in_fft * FactWiM + i;
        int twiddle_index = M * row + element;
        sycl::vec<T, 2> twiddles = *reinterpret_cast<const sycl::vec<T, 2>*>(&wg_twiddles[2 * twiddle_index]);
        T twiddle_real = twiddles[0];
        T twiddle_imag = twiddles[1];
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
          /**
           * Store back FactWiM contiguous elements per column corresponding to the sub batch being processed,
           * un-transposing the transposed result obtained from sg_dft
           */
          detail::unrolled_loop<0, FactWiM, 1>([&](const int j) __attribute__((always_inline)) {
            std::size_t base_index =
                2 * max_num_batches_in_local_mem * static_cast<std::size_t>((j * FactSgN + wi_in_fft + M * row)) +
                2 * sub_batch_num;
            loc[detail::pad_local(base_index, BankLinesPerPad)] = priv[2 * j];
            loc[detail::pad_local(base_index + 1, BankLinesPerPad)] = priv[2 * j + 1];
          });
        } else {
          store_transposed<2 * FactWiM, detail::pad::DO_PAD, BankLinesPerPad>(
              priv, loc, static_cast<std::size_t>(wi_in_fft), static_cast<std::size_t>(FactSgM),
              static_cast<std::size_t>(2 * M * row));
        }
      }
    }
  }
}

}  // namespace portfft

#endif
