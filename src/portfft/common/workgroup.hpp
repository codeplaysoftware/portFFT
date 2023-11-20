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

#include "helpers.hpp"
#include "logging.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "portfft/traits.hpp"
#include "subgroup.hpp"

namespace portfft {

/**
 * Calculate the number of groups or bank lines of PORTFFT_N_LOCAL_BANKS between each padding in local memory,
 * specifically for reducing bank conflicts when reading values from the columns of a 2D data layout. e.g. If there are
 * 64 complex elements in a row, then the consecutive values in the same column are 128 floats apart. There are 32
 * banks, each the size of a float, so we only want a padding float every 128/32=4 bank lines to read along the column
 * without bank conflicts.
 *
 * @tparam T Input type to the function
 * @param row_size the size in bytes of the row. 32 std::complex<float> values would probably have a size of 256 bytes.
 * @return the number of groups of PORTFFT_N_LOCAL_BANKS between each padding in local memory.
 */
template <typename T>
constexpr T bank_lines_per_pad_wg(T row_size) {
  constexpr T BankLineSize = sizeof(float) * PORTFFT_N_LOCAL_BANKS;
  if (row_size % BankLineSize == 0) {
    return row_size / BankLineSize;
  }
  // There is room for improvement here. E.G if row_size was half of BankLineSize then maybe you would still want 1
  // pad every bank group.
  return 1;
}

namespace detail {
/**
 * Calculate all dfts in one dimension of the data stored in local memory.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input Layout
 * @tparam SubgroupSize Size of the subgroup
 * @tparam LocalT The type of the local view
 * @tparam T Scalar type
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param scaling_factor Scalar factor with which the result is to be scaled
 * @param max_num_batches_in_local_mem Number of batches local memory is allocated for
 * @param batch_num_in_local Id of the local memory batch to work on
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 * @param batch_num_in_kernel Absosulte batch from which batches loaded in local memory will be computed
 * @param dft_size Size of each DFT to calculate
 * @param stride_within_dft Stride between elements of each DFT - also the number of the DFTs in the inner dimension
 * @param ndfts_in_outer_dimension Number of DFTs in outer dimension
 * @param layout_in Input Layout
 * @param multiply_on_load Whether the input data is multiplied with some data array before fft computation.
 * @param MultiplyOnStore Whether the input data is multiplied with some data array after fft computation.
 * @param ApplyScaleFactor Whether or not the scale factor is applied
 * @param global_data global data for the kernel
 */
template <direction Dir, Idx SubgroupSize, typename LocalT, typename T>
__attribute__((always_inline)) inline void dimension_dft(
    LocalT loc, T* loc_twiddles, const T* wg_twiddles, T scaling_factor, Idx max_num_batches_in_local_mem,
    Idx batch_num_in_local, const T* load_modifier_data, const T* store_modifier_data, IdxGlobal batch_num_in_kernel,
    Idx dft_size, Idx stride_within_dft, Idx ndfts_in_outer_dimension, detail::layout layout_in,
    detail::elementwise_multiply multiply_on_load, detail::elementwise_multiply multiply_on_store,
    detail::apply_scale_factor apply_scale_factor, global_data_struct<1> global_data) {
  static_assert(std::is_same_v<detail::get_element_t<LocalT>, T>, "Real type mismatch");
  global_data.log_message_global(__func__, "entered", "DFTSize", dft_size, "stride_within_dft", stride_within_dft,
                                 "ndfts_in_outer_dimension", ndfts_in_outer_dimension, "max_num_batches_in_local_mem",
                                 max_num_batches_in_local_mem, "batch_num_in_local", batch_num_in_local);
  const Idx outer_stride = dft_size * stride_within_dft;
  // the number of work-items involved in every subgroup fft
  const Idx fact_sg = detail::factorize_sg(dft_size, SubgroupSize);
  // the number of values held in by a work-item in a row subgroup dft
  const Idx fact_wi = dft_size / fact_sg;

  const Idx ffts_per_sg = SubgroupSize / fact_sg;
  const bool excess_wis = SubgroupSize % fact_sg > 0;
  const bool excess_sgs = stride_within_dft % ffts_per_sg > 0;
  // only needed when there are excess work-items
  const Idx max_working_tid_in_sg = ffts_per_sg * fact_sg;

  const Idx num_sgs = static_cast<Idx>(global_data.it.get_local_range(0)) / SubgroupSize;
  const Idx fft_in_subgroup = static_cast<Idx>(global_data.sg.get_local_linear_id()) / fact_sg;
  // id of the work-item in the fft
  const Idx wi_id_in_fft = static_cast<Idx>(global_data.sg.get_local_linear_id()) % fact_sg;

#ifdef PORTFFT_USE_SCLA
  T wi_private_scratch[detail::SpecConstWIScratchSize];
  T priv[detail::SpecConstNumRealsPerFFT];
#else
  T wi_private_scratch[2 * wi_temps(detail::MaxComplexPerWI)];
  T priv[2 * MaxComplexPerWI];
#endif

  const Idx begin = static_cast<Idx>(global_data.sg.get_group_id()) * ffts_per_sg + fft_in_subgroup;
  const Idx step = num_sgs * ffts_per_sg;
  Idx end;
  const Idx total_dfts = stride_within_dft * ndfts_in_outer_dimension;
  if (excess_sgs) {
    // sg_dft uses subgroup operations, so all of the subgroup must enter the loop
    // it is safe to increase column_end for all work-items since they are all taking steps of ffts_per_sg anyway
    end = detail::round_up_to_multiple(total_dfts, ffts_per_sg);
  } else {
    end = total_dfts;
  }

  if (excess_wis) {
    // also allow these work-items to enter the loop, without making other work-items do another loop.
    end += (fft_in_subgroup == ffts_per_sg) ? 1 : 0;
  }
  for (Idx j = begin; j < end; j += step) {
    Idx j_inner = j % stride_within_dft;
    Idx j_outer = j / stride_within_dft;
    auto loc_start_view = offset_view(loc, 2 * j_outer * outer_stride);
    bool working = true;
    if (excess_sgs) {
      working = j < total_dfts;
    }
    if (excess_wis) {
      working = working && static_cast<Idx>(global_data.sg.get_local_linear_id()) < max_working_tid_in_sg;
    }
    if (working) {
      if (layout_in == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "loading transposed data from local to private memory");
        detail::strided_view local_view{
            loc, std::array{1, stride_within_dft, max_num_batches_in_local_mem},
            std::array{2 * wi_id_in_fft * fact_wi, 2 * (j_inner + j_outer * outer_stride), 2 * batch_num_in_local}};
        copy_wi<2>(global_data, local_view, priv, fact_wi);
      } else {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        // transposition due to working on columns
        detail::strided_view local_view{loc_start_view, std::array{1, stride_within_dft},
                                        std::array{2 * fact_wi * wi_id_in_fft, 2 * j_inner}};
        copy_wi<2>(global_data, local_view, priv, fact_wi);
      }
      global_data.log_dump_private("data loaded in registers:", priv, 2 * fact_wi);

      if (wg_twiddles) {
        PORTFFT_UNROLL
        for (Idx i = 0; i < fact_wi; i++) {
          // Unintuitive indexing to ensure coalesced access
          Idx twiddle_i = i * fact_sg + wi_id_in_fft;
          Idx twiddle_j = j_outer;
          Idx twiddle_index = twiddle_j * dft_size + twiddle_i;
          sycl::vec<T, 2> twiddles = reinterpret_cast<const sycl::vec<T, 2>*>(wg_twiddles)[twiddle_index];
          T twiddle_real = twiddles[0];
          T twiddle_imag = twiddles[1];
          if constexpr (Dir == direction::BACKWARD) {
            twiddle_imag = -twiddle_imag;
          }
          multiply_complex(priv[2 * i], priv[2 * i + 1], twiddle_real, twiddle_imag, priv[2 * i], priv[2 * i + 1]);
        }
        global_data.log_dump_private("data in registers after twiddle multiplication:", priv, 2 * fact_wi);
      }
      if (apply_scale_factor == detail::apply_scale_factor::APPLIED) {
        PORTFFT_UNROLL
        for (Idx i = 0; i < fact_wi; i++) {
          priv[2 * i] *= scaling_factor;
          priv[2 * i + 1] *= scaling_factor;
        }
        global_data.log_dump_private("data in registers after scaling:", priv, 2 * fact_wi);
      }
      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        PORTFFT_UNROLL
        for (Idx idx = 0; idx < fact_wi; idx++) {
          // load modifier needs to be tensor shape : n_transforms x M x FacWi x fact_sg
          IdxGlobal base_offset = 2 * (batch_num_in_kernel + static_cast<IdxGlobal>(batch_num_in_local)) *
                                      static_cast<IdxGlobal>(dft_size) +
                                  static_cast<IdxGlobal>(2 * fact_wi * fact_sg + 2 * idx * fact_sg + 2 * wi_id_in_fft);
          const sycl::vec<T, 2> priv_modifier =
              *reinterpret_cast<const sycl::vec<T, 2>*>(&load_modifier_data[base_offset]);
          multiply_complex(priv[2 * idx], priv[2 * idx + 1], priv_modifier[0], priv_modifier[1], priv[2 * idx],
                           priv[2 * idx + 1]);
        }
      }
    }

    sg_dft<Dir, SubgroupSize>(priv, global_data.sg, fact_wi, fact_sg, loc_twiddles, wi_private_scratch);

    if (working) {
      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        // Store modifier data layout in global memory - n_transforms x N x FactorSG x FactorWI
        PORTFFT_UNROLL
        for (Idx idx = 0; idx < fact_wi; idx++) {
          IdxGlobal base_offset =
              2 * (batch_num_in_kernel + static_cast<IdxGlobal>(batch_num_in_local)) *
                  static_cast<IdxGlobal>(dft_size) +
              static_cast<IdxGlobal>(2 * j * fact_wi * fact_sg + 2 * idx * fact_sg + 2 * wi_id_in_fft);
          const sycl::vec<T, 2> priv_modifier =
              *reinterpret_cast<const sycl::vec<T, 2>*>(&store_modifier_data[base_offset]);
          multiply_complex(priv[2 * idx], priv[2 * idx + 1], priv_modifier[0], priv_modifier[1], priv[2 * idx],
                           priv[2 * idx + 1]);
        }
      }
      global_data.log_dump_private("data in registers after computation:", priv, 2 * fact_wi);
      if (layout_in == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "storing transposed data from private to local memory");
        detail::strided_view local_view{
            loc, std::array{fact_sg, stride_within_dft, max_num_batches_in_local_mem},
            std::array{2 * wi_id_in_fft, 2 * (j_inner + j_outer * outer_stride), 2 * batch_num_in_local}};
        copy_wi<2>(global_data, priv, local_view, fact_wi);
      } else {
        global_data.log_message_global(__func__, "storing non-transposed data from private to local memory");
        // transposition due to working on columns AND transposition for SG dft
        detail::strided_view local_view{loc, std::array{fact_sg, stride_within_dft},
                                        std::array{2 * wi_id_in_fft, 2 * (j_inner + j_outer * outer_stride)}};
        copy_wi<2>(global_data, priv, local_view, fact_wi);
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
 * @tparam SubgroupSize Size of the subgroup
 * @tparam LocalT Local memory view type
 * @tparam T Scalar type
 *
 * @param loc A view of a local accessor containing input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param scaling_factor Scalar factor with which the result is to be scaled
 * @param max_num_batches_in_local_mem Number of batches local memory is allocated for
 * @param batch_num_in_local Id of the local memory batch to work on
 * @param batch_num_in_kernel Absosulte batch from which batches loaded in local memory will be computed
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 * @param fft_size Problem Size
 * @param N Smaller factor of the Problem size
 * @param M Larger factor of the problem size
 * @param layout_in Whether or not the input is transposed
 * @param multiply_on_load Whether the input data is multiplied with some data array before fft computation.
 * @param multiply_on_store Whether the input data is multiplied with some data array after fft computation.
 * @param apply_scale_factor Whether or not the scale factor is applied
 * @param global_data global data for the kernel
 */
template <direction Dir, Idx SubgroupSize, typename LocalT, typename T>
PORTFFT_INLINE void wg_dft(LocalT loc, T* loc_twiddles, const T* wg_twiddles, T scaling_factor,
                           Idx max_num_batches_in_local_mem, Idx batch_num_in_local, IdxGlobal batch_num_in_kernel,
                           const T* load_modifier_data, const T* store_modifier_data, Idx fft_size, Idx N, Idx M,
                           detail::layout layout_in, detail::elementwise_multiply multiply_on_load,
                           detail::elementwise_multiply multiply_on_store,
                           detail::apply_scale_factor apply_scale_factor, detail::global_data_struct<1> global_data) {
  global_data.log_message_global(__func__, "entered", "FFTSize", fft_size, "N", N, "M", M,
                                 "max_num_batches_in_local_mem", max_num_batches_in_local_mem, "batch_num_in_local",
                                 batch_num_in_local);
  // column-wise DFTs
  detail::dimension_dft<Dir, SubgroupSize, LocalT, T>(
      loc, loc_twiddles + (2 * M), nullptr, 1, max_num_batches_in_local_mem, batch_num_in_local, load_modifier_data,
      store_modifier_data, batch_num_in_kernel, N, M, 1, layout_in, multiply_on_load,
      detail::elementwise_multiply::NOT_APPLIED, detail::apply_scale_factor::NOT_APPLIED, global_data);
  sycl::group_barrier(global_data.it.get_group());
  // row-wise DFTs, including twiddle multiplications and scaling
  detail::dimension_dft<Dir, SubgroupSize, LocalT, T>(
      loc, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, batch_num_in_local,
      load_modifier_data, store_modifier_data, batch_num_in_kernel, M, 1, N, layout_in,
      detail::elementwise_multiply::NOT_APPLIED, multiply_on_store, apply_scale_factor, global_data);
  global_data.log_message_global(__func__, "exited");
}

}  // namespace portfft

#endif
