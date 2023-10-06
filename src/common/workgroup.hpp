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
#include <defines.hpp>
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
 * @return the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
constexpr Idx bank_lines_per_pad_wg(Idx row_size) {
  constexpr Idx BankLineSize = sizeof(float) * PORTFFT_N_LOCAL_BANKS;
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
 * @tparam LayoutIn Input Layout
 * @tparam MultiplyOnLoad Whether the input data is multiplied with some data array before fft computation.
 * @tparam MultiplyOnStore Whether the input data is multiplied with some data array after fft computation.
 * @tparam ApplyScaleFactor Whether or not the scale factor is applied
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
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 * @param absolute_base_batch Absosulte batch from which batches loaded in local memory will be computed
 */
template <direction Dir, detail::layout LayoutIn, detail::elementwise_multiply MultiplyOnLoad,
          detail::elementwise_multiply MultiplyOnStore, detail::apply_scale_factor ApplyScaleFactor, Idx DFTSize,
          Idx StrideWithinDFT, Idx NDFTsInOuterDimension, Idx SubgroupSize, Idx BankLinesPerPad, typename T>
__attribute__((always_inline)) inline void dimension_dft(T* loc, T* loc_twiddles, const T* wg_twiddles,
                                                         T scaling_factor, Idx max_num_batches_in_local_mem,
                                                         Idx sub_batch_num, const T* load_modifier_data,
                                                         const T* store_modifier_data, IdxGlobal absolute_base_batch,
                                                         global_data_struct global_data) {
  global_data.log_message_global(__func__, "entered", "DFTSize", DFTSize, "StrideWithinDFT", StrideWithinDFT,
                                 "NDFTsInOuterDimension", NDFTsInOuterDimension, "max_num_batches_in_local_mem",
                                 max_num_batches_in_local_mem, "sub_batch_num", sub_batch_num);
  constexpr Idx OuterStride = DFTSize * StrideWithinDFT;
  // the number of work-items involved in every subgroup fft
  constexpr Idx FactSg = detail::factorize_sg(DFTSize, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  constexpr Idx FactWi = DFTSize / FactSg;

  constexpr Idx FFTsPerSG = SubgroupSize / FactSg;
  constexpr bool ExcessWIs = SubgroupSize % FactSg > 0;
  constexpr bool ExcessSGs = StrideWithinDFT % FFTsPerSG > 0;
  // only needed when there are excess work-items
  constexpr Idx MaxWorkingTidInSg = FFTsPerSG * FactSg;

  const Idx num_sgs = static_cast<Idx>(global_data.it.get_local_range(0)) / SubgroupSize;
  const Idx fft_in_subgroup = static_cast<Idx>(global_data.sg.get_local_linear_id()) / FactSg;
  // id of the work-item in the fft
  const Idx wi_id_in_fft = static_cast<Idx>(global_data.sg.get_local_linear_id()) % FactSg;

  T priv[2 * FactWi];

  const Idx begin = static_cast<Idx>(global_data.sg.get_group_id()) * FFTsPerSG + fft_in_subgroup;
  const Idx step = num_sgs * FFTsPerSG;
  Idx end;
  constexpr Idx TotalDFTs = StrideWithinDFT * NDFTsInOuterDimension;
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
  for (Idx j = begin; j < end; j += step) {
    Idx j_inner = j % StrideWithinDFT;
    Idx j_outer = j / StrideWithinDFT;
    T* loc_start = loc + detail::pad_local(2 * j_outer * OuterStride, BankLinesPerPad);
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
        transfer_strided<detail::transfer_direction::LOCAL_TO_PRIVATE, detail::pad::DO_PAD, FactWi>(
            global_data, loc, priv, 2 * max_num_batches_in_local_mem, 2 * sub_batch_num, StrideWithinDFT,
            j_inner + j_outer * OuterStride, 1, wi_id_in_fft * FactWi, BankLinesPerPad);
      } else {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        // transposition due to working on columns
        local2private_transposed<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(global_data, loc_start, priv,
                                                                               wi_id_in_fft, j_inner, StrideWithinDFT);
      }
      global_data.log_dump_private("data loaded in registers:", priv, 2 * FactWi);

      if (wg_twiddles) {
        detail::unrolled_loop<0, FactWi, 1>([&](const Idx i) PORTFFT_INLINE {
          // Unintuitive indexing to ensure coalesced access
          Idx twiddle_i = i * FactSg + wi_id_in_fft;
          Idx twiddle_j = j_outer;
          Idx twiddle_index = twiddle_j * DFTSize + twiddle_i;
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
      if constexpr (ApplyScaleFactor == detail::apply_scale_factor::APPLIED) {
        if (scaling_factor != static_cast<T>(1)) {
          detail::unrolled_loop<0, FactWi, 1>([&](const Idx i) PORTFFT_INLINE {
            priv[2 * i] *= scaling_factor;
            priv[2 * i + 1] *= scaling_factor;
          });
          global_data.log_dump_private("data in registers after scaling:", priv, 2 * FactWi);
        }
      }
      if constexpr (MultiplyOnLoad == detail::elementwise_multiply::APPLIED) {
        detail::unrolled_loop<0, FactWi, 1>([&](const Idx idx) PORTFFT_INLINE {
          // load modifier needs to be tensor shape : n_transforms x M x FacWi x FactSG
          IdxGlobal base_offset =
              2 * (absolute_base_batch + static_cast<IdxGlobal>(sub_batch_num)) * static_cast<IdxGlobal>(DFTSize) +
              static_cast<IdxGlobal>(2 * FactWi * FactSg + 2 * idx * FactSg + 2 * wi_id_in_fft);
          sycl::vec<T, 2> priv_modifier = *reinterpret_cast<sycl::vec<T, 2>*>(&load_modifier_data[base_offset]);
          multiply_complex(priv[2 * idx], priv[2 * idx + 1], priv_modifier[0], priv_modifier[1], priv[2 * idx],
                           priv[2 * idx + 1]);
        });
      }
    }
    sg_dft<Dir, FactWi, FactSg>(priv, global_data.sg, loc_twiddles);

    if (working) {
      if constexpr (MultiplyOnStore == detail::elementwise_multiply::APPLIED) {
        // Store modifier data layout in global memory - n_transforms x N x FactorSG x FactorWI
        detail::unrolled_loop<0, FactWi, 1>([&](const Idx idx) PORTFFT_INLINE {
          IdxGlobal base_offset =
              2 * (absolute_base_batch + static_cast<IdxGlobal>(sub_batch_num)) * static_cast<IdxGlobal>(DFTSize) +
              static_cast<IdxGlobal>(2 * j * FactWi * FactSg + 2 * idx * FactSg + 2 * wi_id_in_fft);
          sycl::vec<T, 2> priv_modifier = *reinterpret_cast<sycl::vec<T, 2>*>(&store_modifier_data[base_offset]);
          multiply_complex(priv[2 * idx], priv[2 * idx + 1], priv_modifier[0], priv_modifier[1], priv[2 * idx],
                           priv[2 * idx + 1]);
        });
      }
      global_data.log_dump_private("data in registers after computation:", priv, 2 * FactWi);
      if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "storing transposed data from private to local memory");
        transfer_strided<detail::transfer_direction::PRIVATE_TO_LOCAL, detail::pad::DO_PAD, FactWi>(
            global_data, priv, loc, 2 * max_num_batches_in_local_mem, 2 * sub_batch_num, StrideWithinDFT,
            j_inner + j_outer * OuterStride, FactSg, wi_id_in_fft, BankLinesPerPad);
      } else {
        global_data.log_message_global(__func__, "storing non-transposed data from private to local memory");
        // transposition due to working on columns AND transposition for SG dft
        private2local_2strides<FactWi, detail::pad::DO_PAD, BankLinesPerPad>(
            global_data, priv, loc, wi_id_in_fft, FactSg * StrideWithinDFT, j_inner + j_outer * OuterStride,
            StrideWithinDFT);
      }
    }
  }
  global_data.log_message_global(__func__, "exited");
}
}  // namespace detail

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Whether or not the input is transposed
 * @tparam MultiplyOnLoad Whether the input data is multiplied with some data array before fft computation.
 * @tparam MultiplyOnStore Whether the input data is multiplied with some data array after fft computation.
 * @tparam ApplyScaleFactor Whether or not the scale factor is applied
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
 * @param absolute_base_batch Absosulte batch from which batches loaded in local memory will be computed
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <direction Dir, detail::layout LayoutIn, detail::elementwise_multiply MultiplyOnLoad,
          detail::elementwise_multiply MultiplyOnStore, detail::apply_scale_factor ApplyScaleFactor, Idx FFTSize, Idx N,
          Idx M, Idx SubgroupSize, Idx BankLinesPerPad, typename T>
PORTFFT_INLINE void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, T scaling_factor,
                           Idx max_num_batches_in_local_mem, Idx sub_batch_num, IdxGlobal absolute_base_batch,
                           const T* load_modifier_data, const T* store_modifier_data,
                           detail::global_data_struct global_data) {
  global_data.log_message_global(__func__, "entered", "FFTSize", FFTSize, "N", N, "M", M,
                                 "max_num_batches_in_local_mem", max_num_batches_in_local_mem, "sub_batch_num",
                                 sub_batch_num);
  // column-wise DFTs
  detail::dimension_dft<Dir, LayoutIn, MultiplyOnLoad, detail::elementwise_multiply::NOT_APPLIED,
                        detail::apply_scale_factor::NOT_APPLIED, N, M, 1, SubgroupSize, BankLinesPerPad, T>(
      loc, loc_twiddles + (2 * M), nullptr, 1, max_num_batches_in_local_mem, sub_batch_num, load_modifier_data,
      store_modifier_data, absolute_base_batch, global_data);
  sycl::group_barrier(global_data.it.get_group());
  // row-wise DFTs, including twiddle multiplications and scaling
  detail::dimension_dft<Dir, LayoutIn, detail::elementwise_multiply::NOT_APPLIED, MultiplyOnStore, ApplyScaleFactor, M,
                        1, N, SubgroupSize, BankLinesPerPad, T>(
      loc, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, sub_batch_num, load_modifier_data,
      store_modifier_data, absolute_base_batch, global_data);
  global_data.log_message_global(__func__, "exited");
}

}  // namespace portfft

#endif
