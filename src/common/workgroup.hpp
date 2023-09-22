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
#include <common/logging.hpp>
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

namespace detail {
/**
 * Calculate all dfts in one dimension of the data stored in local memory.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input layout
 * @tparam DFTSize Size of each DFT to calculate
 * @tparam StrideWithinDFT Stride between elements of each DFT - also the number of the DFTs in the inner dimension
 * @tparam NDFTsInOuterDimension Number of DFTs in outer dimension
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankLinesPerPad The number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 * @tparam T Scalar type
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param scaling_factor Scalar factor with which the result is to be scaled
 * @param max_num_batches_in_local_mem Number of batches local memory is allocated for
 * @param sub_batch_num Id of the local memory batch to work on
 * @param global_data global data for the kernel
 */
template <direction Dir, detail::layout LayoutIn, int DFTSize, int StrideWithinDFT, int NDFTsInOuterDimension,
          int SubgroupSize, std::size_t BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void dimension_dft(T* loc, T* loc_twiddles, const T* wg_twiddles,
                                                         T scaling_factor, std::size_t max_num_batches_in_local_mem,
                                                         std::size_t sub_batch_num, global_data_struct global_data) {
  global_data.log_message_global(__func__, "entered", "DFTSize", DFTSize, "StrideWithinDFT", StrideWithinDFT,
                                 "NDFTsInOuterDimension", NDFTsInOuterDimension, "max_num_batches_in_local_mem",
                                 max_num_batches_in_local_mem, "sub_batch_num", sub_batch_num);
  constexpr int OuterStride = DFTSize * StrideWithinDFT;
  // the number of work-items involved in every subgroup fft
  constexpr int FactSg = detail::factorize_sg(DFTSize, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr int FactWi = DFTSize / FactSg;

  constexpr int FFTsPerSG = SubgroupSize / FactSg;
  constexpr bool ExcessWIs = SubgroupSize % FactSg > 0;
  constexpr bool ExcessSGs = StrideWithinDFT % FFTsPerSG > 0;
  // only needed when there are excess work-items
  constexpr std::size_t MaxWorkingTidInSg = FFTsPerSG * FactSg;

  const int num_sgs = static_cast<int>(global_data.it.get_local_range(0)) / SubgroupSize;
  const int fft_in_subgroup = static_cast<int>(global_data.sg.get_local_linear_id()) / FactSg;
  // id of the work-item in the fft
  const int wi_id_in_fft = static_cast<int>(global_data.sg.get_local_linear_id()) % FactSg;

  T priv[2 * FactWi];

  const int begin = static_cast<int>(global_data.sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
  const int step = num_sgs * FFTsPerSG;
  int end;
  constexpr int TotalDFTs = StrideWithinDFT * NDFTsInOuterDimension;
  if constexpr (ExcessSGs) {
    // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
    // it is safe to increase column_end for all work-items since they are all taking steps of FFTsPerSG anyway
    end = detail::round_up_to_multiple(TotalDFTs, FFTsPerSG);
  } else {
    end = TotalDFTs;
  }

  if constexpr (ExcessWIs) {
    // also allow these work-items to enter the loop, without making other work-items do another loop.
    end += (fft_in_subgroup == FFTsPerSG) ? 1 : 0;
  }
  for (int j = begin; j < end; j += step) {
    int j_inner = j % StrideWithinDFT;
    int j_outer = j / StrideWithinDFT;
    auto loc_view = make_padded_view<detail::pad::DO_PAD, BankLinesPerPad>(loc);
    T* loc_start = loc + detail::pad_local(static_cast<std::size_t>(2 * j_outer * OuterStride), BankLinesPerPad);
    auto loc_start_view = make_padded_view<detail::pad::DO_PAD, BankLinesPerPad>(loc_start);
    bool working = true;
    if constexpr (ExcessSGs) {
      working = j < TotalDFTs;
    }
    if constexpr (ExcessWIs) {
      working = working && global_data.sg.get_local_linear_id() < MaxWorkingTidInSg;
    }
    if (working) {
      if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "loading transposed data from local to private memory");
        transfer_strided<detail::transfer_direction::LOCAL_TO_PRIVATE, FactWi>(
            global_data, make_complex_complex_view(priv), make_complex_complex_view(loc_view),
            max_num_batches_in_local_mem, sub_batch_num, static_cast<std::size_t>(StrideWithinDFT),
            static_cast<std::size_t>(j_inner + j_outer * OuterStride), 1L,
            static_cast<std::size_t>(wi_id_in_fft * FactWi));
      } else {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        // transposition due to working on columns
        local2private_transposed<FactWi>(global_data, make_complex_complex_view(loc_start_view),
                                         make_complex_complex_view(priv), wi_id_in_fft, j_inner, StrideWithinDFT);
      }
      global_data.log_dump_private("data loaded in registers:", priv, 2 * FactWi);

      if (wg_twiddles) {
        detail::unrolled_loop<0, FactWi, 1>([&](const int i) PORTFFT_INLINE {
          // Unintuitive indexing to ensure coalesced access
          int twiddle_i = i * FactSg + wi_id_in_fft;
          int twiddle_j = j_outer;
          int twiddle_index = twiddle_j * DFTSize + twiddle_i;
          sycl::vec<T, 2> twiddles = reinterpret_cast<const sycl::vec<T, 2>*>(wg_twiddles)[twiddle_index];
          T twiddle_real = twiddles[0];
          T twiddle_imag = twiddles[1];
          if constexpr (Dir == direction::BACKWARD) {
            twiddle_imag = -twiddle_imag;
          }
          multiply_complex(priv[2 * i], priv[2 * i + 1], twiddle_real, twiddle_imag, priv[2 * i], priv[2 * i + 1]);
        });
        global_data.log_dump_private("data in registers after twiddle multiplication:", priv, 2 * FactWi);
      }
      if (scaling_factor != static_cast<T>(1)) {
        detail::unrolled_loop<0, FactWi, 1>([&](const int i) PORTFFT_INLINE {
          priv[2 * i] *= scaling_factor;
          priv[2 * i + 1] *= scaling_factor;
        });
        global_data.log_dump_private("data in registers after scaling:", priv, 2 * FactWi);
      }
    }
    sg_dft<Dir, FactWi, FactSg>(priv, global_data.sg, loc_twiddles);
    if (working) {
      global_data.log_dump_private("data in registers after computation:", priv, 2 * FactWi);
      if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "storing transposed data from private to local memory");
        transfer_strided<detail::transfer_direction::PRIVATE_TO_LOCAL, FactWi>(
            global_data, make_complex_complex_view(priv), make_complex_complex_view(loc_view),
            max_num_batches_in_local_mem, sub_batch_num, static_cast<std::size_t>(StrideWithinDFT),
            static_cast<std::size_t>(j_inner + j_outer * OuterStride), static_cast<std::size_t>(FactSg),
            static_cast<std::size_t>(wi_id_in_fft));
      } else {
        global_data.log_message_global(__func__, "storing non-transposed data from private to local memory");
        // transposition due to working on columns AND transposition for SG dft
        private2local_transposed<FactWi>(global_data, make_complex_complex_view(priv),
                                         make_complex_complex_view(loc_view), wi_id_in_fft, FactSg,
                                         j_inner + j_outer * OuterStride, StrideWithinDFT);
      }
    }
  }
  global_data.log_message_global(__func__, "exited");
}
};

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input layout
 * @tparam FFTSize Problem Size
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam BankLinesPerPad The number of groups of PORTFFT_N_LOCAL_BANKS to have between each local pad
 * @tparam T Scalar type
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param global_data global data for the kernel
 * @param scaling_factor Scalar factor with which the result is to be scaled
 * @param max_num_batches_in_local_mem Number of batches local memory is allocated for
 * @param sub_batch_num Id of the local memory batch to work on
 */
template <direction Dir, detail::layout LayoutIn, int FFTSize, int N, int M, int SubgroupSize,
          std::size_t BankLinesPerPad, typename T>
PORTFFT_INLINE void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, detail::global_data_struct global_data,
                           T scaling_factor, std::size_t max_num_batches_in_local_mem, std::size_t sub_batch_num) {
  global_data.log_message_global(__func__, "entered", "FFTSize", FFTSize, "N", N, "M", M,
                                 "max_num_batches_in_local_mem", max_num_batches_in_local_mem, "sub_batch_num",
                                 sub_batch_num);
  // column-wise DFTs
  detail::dimension_dft<Dir, LayoutIn, N, M, 1, SubgroupSize, BankLinesPerPad, T>(
      loc, loc_twiddles + (2 * M), nullptr, 1, max_num_batches_in_local_mem, sub_batch_num, global_data);
  sycl::group_barrier(global_data.it.get_group());
  // row-wise DFTs, including twiddle multiplications and scaling
  detail::dimension_dft<Dir, LayoutIn, M, 1, N, SubgroupSize, BankLinesPerPad, T>(
      loc, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, sub_batch_num, global_data);
  global_data.log_message_global(__func__, "exited");
}

}  // namespace portfft

#endif
