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

#ifndef PORTFFT_COMMON_SUBGROUP_BLUESTEIN_HPP
#define PORTFFT_COMMON_SUBGROUP_BLUESTEIN_HPP

#include "helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/subgroup_ct.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"

namespace portfft {

/**
 * Function to copy data between local and global memory as required by the subgroup level Bluestein algorithm,
 * when the data in both local and global memory is in packed format,when the storage scheme is INTERLEAVED_COMPLEX
 *
 * @tparam SubgroupSize Subgroup size
 * @tparam Direction  Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam TIn Global memory Type
 * @tparam LocView Type of the view constructed for local memory
 * @param global_ptr global memory pointer
 * @param loc_view View of the local memory
 * @param committed_size Size of the DFT as committed, also the number of complex elements in each transform present in
 * global memory
 * @param fft_size The padded DFT size, also the number of elements of complex elements in each transform that resides
 * in local memory
 * @param global_ptr_offset Offset to be applied to the global memory pointer
 * @param loc_offset Offset to be applied to the local memory view
 * @param n_ffts_in_sg Number of ffts that can be calculated by a single subgroup
 * @param transform_id Id of the transform in the kernel
 * @param n_transforms Total number of transforms in the kernel
 * @param global_data global_data_struct associated with the kernel launch
 */
template <Idx SubgroupSize, detail::transfer_direction Direction, typename TIn, typename LocView>
PORTFFT_INLINE void subgroup_impl_bluestein_local_global_packed_copy(
    TIn global_ptr, LocView& loc_view, Idx committed_size, Idx fft_size, IdxGlobal global_ptr_offset, Idx loc_offset,
    Idx n_ffts_in_sg, IdxGlobal transform_id, IdxGlobal n_transforms, detail::global_data_struct<1>& global_data) {
  PORTFFT_UNROLL
  for (Idx i = 0; i < n_ffts_in_sg && ((i + transform_id) < n_transforms); i++) {
    local_global_packed_copy<detail::level::SUBGROUP, Direction, SubgroupSize>(
        global_ptr, loc_view, global_ptr_offset + static_cast<IdxGlobal>(2 * i * committed_size),
        2 * i * fft_size + loc_offset, 2 * committed_size, global_data);
  }
}

/**
 * Function to copy data between local and global memory as required by the subgroup level Bluestein algorithm,
 * when the data in both local and global memory is in packed format,when the storage scheme is SPLIT_COMPLEX
 *
 * @tparam SubgroupSize Subgroup size
 * @tparam Direction  Direction Direction of the copy, expected to be either transfer_direction::LOCAL_TO_GLOBAL or
 * transfer_direction::GLOBAL_TO_LOCAL
 * @tparam TIn Global memory Type
 * @tparam LocView Type of the view constructed for local memory
 * @param global_ptr global memory pointer containing the real part of the data
 * @param global_imag_ptr global memory pointer containing the imaginary  part of the data
 * @param loc_view View of the local memory
 * @param committed_size Size of the DFT as committed, also the number of complex elements in each transform present in
 * global memory
 * @param fft_size The padded DFT size, also the number of elements of complex elements in each transform that resides
 * in local memory
 * @param global_ptr_offset Offset to be applied to the global memory pointer
 * @param loc_offset Offset to be applied to the local memory view
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param n_ffts_in_sg Number of ffts that can be calculated by a single subgroup
 * @param transform_id Id of the transform in the kernel
 * @param n_transforms Total number of transforms in the kernel
 * @param global_data global_data_struct associated with the kernel launch
 */
template <Idx SubgroupSize, detail::transfer_direction Direction, typename TIn, typename LocView>
PORTFFT_INLINE void subgroup_impl_bluestein_local_global_packed_copy(
    TIn global_ptr, TIn global_imag_ptr, LocView& loc_view, Idx committed_size, Idx fft_size,
    IdxGlobal global_ptr_offset, Idx loc_offset, Idx local_imag_offset, Idx n_ffts_in_sg, IdxGlobal transform_id,
    IdxGlobal n_transforms, detail::global_data_struct<1>& global_data) {
  PORTFFT_UNROLL
  for (Idx i = 0; i < n_ffts_in_sg && (i + transform_id < n_transforms); i++) {
    local_global_packed_copy<detail::level::SUBGROUP, Direction, SubgroupSize>(
        global_ptr, global_imag_ptr, loc_view, static_cast<IdxGlobal>(i * committed_size) + global_ptr_offset,
        i * fft_size + loc_offset, local_imag_offset, committed_size, global_data);
  }
}

/**
 * Implements the Subgroup level Bluestein algorithm when the layout of the data
 * in local memory is in BATCH_INTERLEAVED format
 *
 * @tparam SubgroupSize Subgroup Size
 * @tparam T Scalar Type
 * @tparam LocTwiddlesView Type of view of the local memory containing the twiddles
 * @tparam LocView Type of view of the local memory which stores the data
 * @param priv private memory array on which the computations will be done
 * @param private_scratch Scratch private memory to be passed to the wi_dft as a part of sg_dft
 * @param loc_view view of the local memory to store the data
 * @param load_modifier Global memory pointer containing the load modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param store_modifier Global memory pointer containing the store modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param twiddles_loc view of the local memory containing the twiddles
 * @param conjugate_on_load Whether or not conjugation of the input is to be done before the fft computation
 * @param conjugate_on_store Whether or not conjugation of the input is to be done after the fft computation
 * @param scale_applied Whether or not scale factor is applied
 * @param scale_factor Value of the scaling factor
 * @param id_of_wi_in_fft Id of the workitem in the FFT
 * @param factor_sg Number of workitems participating for one transform
 * @param factor_wi Number of complex elements per workitem for each transform
 * @param storage storage scheme of complex values in local memory, SPLIT_COMPLEX or INTERLEAVED_COMPLEX
 * @param wi_working Whether or not the workitem participates in the data transfers
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param max_num_batches_local_mem Maximum number of transforms that can be stored in local memory
 * @param fft_idx_in_local Id of the transform in local memory
 * @param global_data global_data_struct associated with kernel launch
 */
template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
PORTFFT_INLINE void sg_bluestein_batch_interleaved(
    T* priv, T* priv_scratch, LocView& loc_view, const T* load_modifier, const T* store_modifier,
    LocTwiddlesView& twiddles_loc, detail::complex_conjugate conjugate_on_load,
    detail::complex_conjugate conjugate_on_store, detail::apply_scale_factor scale_applied, T scale_factor,
    Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi, complex_storage storage, bool wi_working, Idx local_imag_offset,
    Idx max_num_batches_local_mem, Idx fft_idx_in_local, detail::global_data_struct<1>& global_data) {
  global_data.log_message_global(__func__, "computing forward FFT and applying scaling factor for the backward phase");
  sg_cooley_tukey<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::APPLIED, load_modifier,
      store_modifier, twiddles_loc, static_cast<T>(1. / (static_cast<T>(factor_sg * factor_wi))), 0, id_of_wi_in_fft,
      factor_sg, factor_wi, global_data);

  // TODO: Currently local memory is being used to load the data back in natural order for the backward phase, as the
  // result of sg_dft is transposed. However, the ideal way to this is using shuffles. Implement a batched matrix
  // transpose to transpose a matrix stored in the private memory of workitems of a subgroup using shuffles only. his we
  // way can even avoid the 2 sg_bluestein functions that we have today
  if (wi_working) {
    global_data.log_message(__func__, "storing result of the forward phase back to local memory");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<2, Idx>(
          loc_view, priv, {{factor_sg, max_num_batches_local_mem}, {2 * id_of_wi_in_fft, 2 * fft_idx_in_local}},
          factor_wi, detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    } else {
      local_private_strided_copy<2, Idx>(
          loc_view, loc_view, priv, {{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local}},
          {{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local + local_imag_offset}}, factor_wi,
          detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    }
  }

  sycl::group_barrier(global_data.sg);
  if (wi_working) {
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      global_data.log_message(__func__, "loading back the result from local memory for the backward phase");
      const Idx fft_element = 2 * id_of_wi_in_fft * factor_wi;
      local_private_strided_copy<1, Idx>(
          loc_view, priv,
          {{max_num_batches_local_mem}, {fft_element * max_num_batches_local_mem + 2 * fft_idx_in_local}}, factor_wi,
          detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    } else {
      local_private_strided_copy<2, Idx>(
          loc_view, loc_view, priv, {{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local}},
          {{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local + local_imag_offset}},
          factor_wi, detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    }
  }
  global_data.log_message(__func__, "computing backward FFT and applying user provided scale value");
  sg_cooley_tukey<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                                detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                                detail::complex_conjugate::APPLIED, scale_applied, static_cast<const T*>(nullptr),
                                load_modifier, twiddles_loc, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi,
                                global_data);

  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate on the output");
    detail::conjugate_inplace(priv, factor_wi);
  }
}

/**
 *
 * Implements the Subgroup level Bluestein algorithm when the layout of the data
 * in local memory is in BATCH_INTERLEAVED format
 *
 * @tparam SubgroupSize Subgroup Size
 * @tparam T Scalar Type
 * @tparam LocTwiddlesView Type of view of the local memory containing the twiddles
 * @tparam LocView Type of view of the local memory which stores the data
 * @param priv private memory array on which the computations will be done
 * @param private_scratch Scratch private memory to be passed to the wi_dft as a part of sg_dft
 * @param loc_view view of the local memory to store the data
 * @param load_modifier Global memory pointer containing the load modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param store_modifier Global memory pointer containing the store modifier data, assumed aligned to at least
 * sycl::vec<T, 2>
 * @param loc_twiddles view of the local memory containing the twiddles
 * @param conjugate_on_load Whether or not conjugation of the input is to be done before the fft computation
 * @param conjugate_on_store Whether or not conjugation of the input is to be done after the fft computation
 * @param scale_applied Whether or not scale factor is applied
 * @param scale_factor Value of the scaling factor
 * @param id_of_wi_in_fft Id of the workitem in the FFT
 * @param factor_sg Number of workitems participating for one transform
 * @param factor_wi Number of complex elements per workitem for each transform
 * @param storage storage scheme of complex values in local memory, SPLIT_COMPLEX or INTERLEAVED_COMPLEX
 * @param wi_working Whether or not the workitem participates in the data transfers
 * @param loc_view_store_offset Offset to be applied to local memory view when storing the data back to local memory
 * after forward fft phase
 * @param loc_view_load_offset offset to be applied to local memory view when loading the data back to local memory for
 * backward fft phase
 * @param local_imag_offset Number of elements in local memory after which the imaginary component of the values is
 * stored
 * @param global_data  global_data_struct associated with kernel launch
 */
template <Idx SubgroupSize, typename T, typename LocTwiddlesView, typename LocView>
void sg_bluestein_packed(T* priv, T* priv_scratch, LocView& loc_view, LocTwiddlesView& loc_twiddles,
                         const T* load_modifier, const T* store_modifier, detail::complex_conjugate conjugate_on_load,
                         detail::complex_conjugate conjugate_on_store, detail::apply_scale_factor scale_applied,
                         T scale_factor, Idx id_of_wi_in_fft, Idx factor_sg, Idx factor_wi, complex_storage storage,
                         bool wi_working, Idx loc_view_store_offset, Idx loc_view_load_offset, Idx local_imag_offset,
                         detail::global_data_struct<1>& global_data) {
  global_data.log_message_global(__func__, "computing forward FFT and applying scaling factor for the backward phase");
  sg_cooley_tukey<SubgroupSize>(
      priv, priv_scratch, detail::elementwise_multiply::APPLIED, detail::elementwise_multiply::APPLIED,
      conjugate_on_load, detail::complex_conjugate::NOT_APPLIED, detail::apply_scale_factor::APPLIED, load_modifier,
      store_modifier, loc_twiddles, static_cast<T>(1. / static_cast<T>(factor_sg * factor_wi)), 0, id_of_wi_in_fft,
      factor_sg, factor_wi, global_data);

  if (wi_working) {
    global_data.log_message(__func__, "storing result of the forward phase back to local memory");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<1, Idx>(loc_view, priv, {{factor_sg}, {loc_view_store_offset}}, factor_wi,
                                         detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    } else {
      local_private_strided_copy<1, Idx>(loc_view, loc_view, priv, {{factor_sg}, {loc_view_store_offset}},
                                         {{factor_sg}, {loc_view_store_offset + local_imag_offset}}, factor_wi,
                                         detail::transfer_direction::PRIVATE_TO_LOCAL, global_data);
    }
  }

  sycl::group_barrier(global_data.sg);

  if (wi_working) {
    global_data.log_message(__func__, "loading back the result from local memory for the backward phase");
    if (storage == complex_storage::INTERLEAVED_COMPLEX) {
      local_private_strided_copy<1, Idx>(loc_view, priv, {{1}, {loc_view_load_offset}}, factor_wi,
                                         detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    } else {
      local_private_strided_copy<1, Idx>(loc_view, loc_view, priv, {{1}, {loc_view_load_offset}},
                                         {{1}, {loc_view_load_offset + local_imag_offset}}, factor_wi,
                                         detail::transfer_direction::LOCAL_TO_PRIVATE, global_data);
    }
  }
  global_data.log_message(__func__, "computing backward FFT and applying user provided scale value");
  sg_cooley_tukey<SubgroupSize>(priv, priv_scratch, detail::elementwise_multiply::NOT_APPLIED,
                                detail::elementwise_multiply::APPLIED, detail::complex_conjugate::APPLIED,
                                detail::complex_conjugate::APPLIED, scale_applied, static_cast<const T*>(nullptr),
                                load_modifier, loc_twiddles, scale_factor, 0, id_of_wi_in_fft, factor_sg, factor_wi,
                                global_data);
  if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
    global_data.log_message(__func__, "Applying complex conjugate on the output");
    detail::conjugate_inplace(priv, factor_wi);
  }
}
}  // namespace portfft

#endif