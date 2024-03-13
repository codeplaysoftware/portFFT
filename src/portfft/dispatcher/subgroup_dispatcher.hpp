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

#ifndef PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP

#include "portfft/common/bluestein.hpp"
#include "portfft/common/helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/memory_views.hpp"
#include "portfft/common/subgroup.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/defines.hpp"
#include "portfft/descriptor.hpp"
#include "portfft/enums.hpp"
#include "portfft/specialization_constant.hpp"

#include <memory>

namespace portfft {
namespace detail {
/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param n_transforms number of transforms
 * @param factor_sg cross-subgroup factor of the fft size
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param num_sgs_per_wg number of subgroups in a workgroup
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
IdxGlobal get_global_size_subgroup(IdxGlobal n_transforms, Idx factor_sg, Idx subgroup_size, Idx num_sgs_per_wg,
                                   Idx n_compute_units) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  Idx maximum_n_sgs = 2 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  Idx wg_size = subgroup_size * num_sgs_per_wg;

  Idx n_ffts_per_wg = (subgroup_size / factor_sg) * num_sgs_per_wg;
  IdxGlobal n_wgs_we_can_utilize = divide_ceil(n_transforms, static_cast<IdxGlobal>(n_ffts_per_wg));
  return static_cast<IdxGlobal>(wg_size) * sycl::min(static_cast<IdxGlobal>(maximum_n_wgs), n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by a subgroup.
 *
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input pointer to global memory containing input data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param output pointer to global memory for output data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param input pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param output pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param loc pointer to local memory. Size requirement is determined by `num_scalars_in_local_mem_struct`.
 * @param loc_twiddles pointer to local memory for twiddle factors. Must have enough space for `2 * FactorWI * FactorSG`
 * values
 * @param n_transforms number of FFT transforms to do in one call
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param twiddles pointer containing twiddles
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <Idx SubgroupSize, typename T>
PORTFFT_INLINE void subgroup_impl(const T* input, T* output, const T* input_imag, T* output_imag, T* loc,
                                  T* loc_twiddles, IdxGlobal n_transforms, const T* twiddles,
                                  global_data_struct<1> global_data, sycl::kernel_handler& kh,
                                  const T* load_modifier_data = nullptr, const T* store_modifier_data = nullptr) {
  const complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  const detail::elementwise_multiply multiply_on_load =
      kh.get_specialization_constant<detail::SpecConstMultiplyOnLoad>();
  const detail::elementwise_multiply multiply_on_store =
      kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  const detail::apply_scale_factor apply_scale_factor =
      kh.get_specialization_constant<detail::SpecConstApplyScaleFactor>();
  const detail::complex_conjugate conjugate_on_load =
      kh.get_specialization_constant<detail::SpecConstConjugateOnLoad>();
  const detail::complex_conjugate conjugate_on_store =
      kh.get_specialization_constant<detail::SpecConstConjugateOnStore>();
  const T scaling_factor = kh.get_specialization_constant<detail::get_spec_constant_scale<T>()>();

  const Idx factor_wi = kh.get_specialization_constant<SubgroupFactorWISpecConst>();
  const Idx factor_sg = kh.get_specialization_constant<SubgroupFactorSGSpecConst>();
  const IdxGlobal input_stride = kh.get_specialization_constant<detail::SpecConstInputStride>();
  const IdxGlobal output_stride = kh.get_specialization_constant<detail::SpecConstOutputStride>();
  const IdxGlobal input_distance = kh.get_specialization_constant<detail::SpecConstInputDistance>();
  const IdxGlobal output_distance = kh.get_specialization_constant<detail::SpecConstOutputDistance>();
  const Idx committed_length = kh.get_specialization_constant<detail::SpecConstCommittedLength>();
  detail::fft_algorithm algorithm = kh.get_specialization_constant<detail::SpecConstFFTAlgorithm>();

  global_data.log_message_global(__func__, "entered", "FactorWI", factor_wi, "FactorSG", factor_sg, "n_transforms",
                                 n_transforms);
  const Idx n_reals_per_wi = 2 * factor_wi;

#ifdef PORTFFT_USE_SCLA
  T wi_private_scratch[detail::SpecConstWIScratchSize];
  T priv[detail::SpecConstNumRealsPerFFT];
#else
  // zero initializing these arrays avoids a bug with the AMD backend
  T wi_private_scratch[2 * wi_temps(detail::MaxComplexPerWI)]{};
  T priv[2 * MaxComplexPerWI]{};
#endif
  Idx local_size = static_cast<Idx>(global_data.it.get_local_range(0));
  Idx subgroup_local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  Idx subgroup_id = static_cast<Idx>(global_data.sg.get_group_id());
  Idx n_sgs_in_wg = static_cast<Idx>(global_data.it.get_local_range(0)) / SubgroupSize;
  Idx id_of_sg_in_kernel = subgroup_id + static_cast<Idx>(global_data.it.get_group_linear_id()) * n_sgs_in_wg;
  Idx n_sgs_in_kernel = static_cast<Idx>(global_data.it.get_group_range(0)) * n_sgs_in_wg;

  Idx n_ffts_per_sg = SubgroupSize / factor_sg;
  Idx max_wis_working = n_ffts_per_sg * factor_sg;
  Idx n_reals_per_fft = factor_sg * n_reals_per_wi;
  Idx fft_size = factor_sg * factor_wi;
  Idx n_cplx_per_sg = n_ffts_per_sg * fft_size;
  Idx n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  // id_of_fft_in_sg must be < n_ffts_per_sg
  Idx id_of_fft_in_sg = std::min(subgroup_local_id / factor_sg, n_ffts_per_sg - 1);
  Idx id_of_wi_in_fft = subgroup_local_id % factor_sg;
  Idx n_ffts_per_wg = n_ffts_per_sg * n_sgs_in_wg;

  // round up so the whole work-group enters the loop and can be used for synchronization
  IdxGlobal rounded_up_n_ffts = round_up_to_multiple(n_transforms, static_cast<IdxGlobal>(n_ffts_per_wg));

  const bool is_input_batch_interleaved = input_stride == n_transforms && input_distance == 1;
  const bool is_output_batch_interleaved = output_stride == n_transforms && output_distance == 1;
  const bool is_input_packed = input_stride == 1 && input_distance == committed_length;
  const bool is_output_packed = output_stride == 1 && output_distance == committed_length;

  IdxGlobal id_of_fft_in_kernel;
  IdxGlobal n_ffts_in_kernel;
  if (is_input_batch_interleaved) {
    id_of_fft_in_kernel = static_cast<IdxGlobal>(global_data.it.get_group(0) * global_data.it.get_local_range(0)) / 2;
    n_ffts_in_kernel = static_cast<Idx>(global_data.it.get_group_range(0)) * local_size / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  constexpr Idx BankLinesPerPad = 1;
  auto loc_view = detail::padded_view(loc, BankLinesPerPad);

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles, n_reals_per_fft);
  sycl::group_barrier(global_data.it.get_group());
  global_data.log_dump_local("twiddles in local memory:", loc_twiddles, n_reals_per_fft);

  for (IdxGlobal i = static_cast<IdxGlobal>(id_of_fft_in_kernel); i < rounded_up_n_ffts;
       i += static_cast<IdxGlobal>(n_ffts_in_kernel)) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    Idx n_ffts_worked_on_by_sg = sycl::min(static_cast<Idx>(n_transforms - i) + id_of_fft_in_sg, n_ffts_per_sg);

    if (is_input_batch_interleaved) {
      /**
       * Codepath taken if the input is transposed
       * The number of transforms that are loaded is equal to half of the workgroup size.
       * Each workitem is responsible for loading all of either the real or complex part of the transform being loaded.
       * The data in local memory is also stored in a transposed manner so that there are no bank conflicts
       * while storing the data.
       * Thus it is loaded in a transposed manner and stored in a transposed manner to prevent data overwrites.
       * Going ahead with the assumption that output will not be stored in a transposed manner(always out of place), it
       * would need to transpose the final result in local memory and store it to global.
       */
      // TODO should we make sure that: max_num_batches_local_mem >= n_ffts_per_wg ?
      Idx max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      Idx num_batches_in_local_mem = [=]() {
        if (i + static_cast<IdxGlobal>(local_size) / 2 < n_transforms) {
          return local_size / 2;
        }
        return static_cast<Idx>(n_transforms - i);
      }();
      Idx rounded_up_ffts_in_local = detail::round_up_to_multiple(num_batches_in_local_mem, n_ffts_per_sg);
      Idx local_imag_offset = factor_wi * factor_sg * max_num_batches_local_mem;

      const bool store_directly_from_private =
          SubgroupSize == factor_sg && is_output_packed && algorithm == detail::fft_algorithm::COOLEY_TUKEY;

      global_data.log_message_global(__func__, "loading transposed data from global to local memory");
      // load / store in a transposed manner
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        subgroup_impl_global2local_strided_copy<detail::level::WORKGROUP, 2, 2, 2>(
            input, loc_view, {2 * n_transforms, static_cast<IdxGlobal>(1)}, {2 * max_num_batches_local_mem, 1}, 2 * i,
            0, {committed_length, 2 * num_batches_in_local_mem}, global_data);
      } else {
        subgroup_impl_global2local_strided_copy<detail::level::WORKGROUP, 2, 2, 2>(
            input, input_imag, loc_view, {n_transforms, static_cast<IdxGlobal>(1)}, {max_num_batches_local_mem, 1}, i,
            0, local_imag_offset, {committed_length, num_batches_in_local_mem}, global_data);
      }

      sycl::group_barrier(global_data.it.get_group());
      global_data.log_dump_local("data loaded to local memory:", loc_view,
                                 n_reals_per_wi * factor_sg * max_num_batches_local_mem);

      const Idx first_fft_in_local_for_wi =
          static_cast<Idx>(global_data.sg.get_group_id()) * n_ffts_per_sg + id_of_fft_in_sg;
      for (Idx fft_idx_in_local = first_fft_in_local_for_wi; fft_idx_in_local < rounded_up_ffts_in_local;
           fft_idx_in_local += n_ffts_per_wg) {
        bool working_inner = fft_idx_in_local < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (working_inner) {
          global_data.log_message_global(__func__, "loading batch_interleaved data from local to private memory");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            const Idx fft_element = 2 * id_of_wi_in_fft * factor_wi;
            subgroup_impl_local_private_copy<1, Idx>(
                loc_view, priv,
                {{{max_num_batches_local_mem}, {fft_element * max_num_batches_local_mem + 2 * fft_idx_in_local}}},
                factor_wi, global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
          } else {
            subgroup_impl_local_private_copy<2, 1, Idx>(
                loc_view, loc_view, priv,
                {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local}}}, {{{2}, {0}}},
                {{{1, max_num_batches_local_mem}, {id_of_wi_in_fft * factor_wi, fft_idx_in_local + local_imag_offset}}},
                {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
          }
          global_data.log_dump_private("data loaded in registers:", priv, n_reals_per_wi);
        }
        IdxGlobal modifier_offset =
            static_cast<IdxGlobal>(n_reals_per_fft) * (i + static_cast<IdxGlobal>(fft_idx_in_local));
        if (algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
          sg_dft_compute<SubgroupSize>(priv, wi_private_scratch, multiply_on_load, multiply_on_store, conjugate_on_load,
                                       conjugate_on_store, apply_scale_factor, load_modifier_data, store_modifier_data,
                                       loc_twiddles, scaling_factor, modifier_offset, id_of_wi_in_fft, factor_sg,
                                       factor_wi, global_data.sg);
        } else {
          sg_bluestein_batch_interleaved<SubgroupSize>(
              priv, wi_private_scratch, loc_view, load_modifier_data, store_modifier_data, loc_twiddles,
              conjugate_on_load, conjugate_on_store, apply_scale_factor, scaling_factor, id_of_wi_in_fft, factor_sg,
              factor_wi, storage, working_inner, local_imag_offset, max_num_batches_local_mem, fft_idx_in_local,
              global_data.sg, global_data);
        }
        // Async DMA can start here for the next set of load/store modifiers.
        if (working_inner) {
          global_data.log_dump_private("data in registers after scaling:", priv, n_reals_per_wi);
        }
        if (store_directly_from_private) {
          if (working_inner) {
            global_data.log_message_global(
                __func__, "storing transposed data from private to packed global memory (SubgroupSize == FactorSG)");
            // Store directly from registers for fully coalesced accesses
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              subgroup_impl_local_private_copy<1, IdxGlobal>(
                  output, priv,
                  {{{static_cast<IdxGlobal>(factor_sg)},
                    {static_cast<IdxGlobal>(i + static_cast<IdxGlobal>(fft_idx_in_local)) *
                         static_cast<IdxGlobal>(2 * fft_size) +
                     static_cast<IdxGlobal>(2 * id_of_wi_in_fft)}}},
                  factor_wi, global_data, detail::transfer_direction::PRIVATE_TO_GLOBAL);
            } else {
              subgroup_impl_local_private_copy<1, 1, IdxGlobal>(
                  output, output_imag, priv,
                  {{{static_cast<IdxGlobal>(factor_sg)},
                    {(i + static_cast<IdxGlobal>(fft_idx_in_local)) * static_cast<IdxGlobal>(fft_size) +
                     static_cast<IdxGlobal>(id_of_wi_in_fft)}}},
                  {{{2}, {0}}},
                  {{{static_cast<IdxGlobal>(factor_sg)},
                    {(i + static_cast<IdxGlobal>(fft_idx_in_local)) * static_cast<IdxGlobal>(fft_size) +
                     static_cast<IdxGlobal>(id_of_wi_in_fft)}}},
                  {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::PRIVATE_TO_GLOBAL);
            }
          }
        } else {
          if (working_inner) {
            global_data.log_message_global(
                __func__,
                "storing transposed data from private to batch interleaved local memory (SubgroupSize != "
                "FactorSG or not packed output layout)");
            // Store back to local memory only
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              subgroup_impl_local_private_copy<2, Idx>(
                  loc_view, priv,
                  {{{factor_sg, max_num_batches_local_mem}, {2 * id_of_wi_in_fft, 2 * fft_idx_in_local}}}, factor_wi,
                  global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
            } else {
              subgroup_impl_local_private_copy<2, 1, Idx>(
                  loc_view, loc_view, priv,
                  {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local}}}, {{{2}, {0}}},
                  {{{factor_sg, max_num_batches_local_mem}, {id_of_wi_in_fft, fft_idx_in_local + local_imag_offset}}},
                  {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
            }
          }
        }
      }
      sycl::group_barrier(global_data.it.get_group());
      if (!store_directly_from_private) {
        global_data.log_dump_local("computed data in local memory:", loc_view, n_reals_per_wi * factor_sg);
        // store back all loaded batches at once.
        // data is batch interleaved in local
        if (!is_output_batch_interleaved) {
          global_data.log_message_global(__func__,
                                         "storing data from batch interleaved local memory to not batch interleaved "
                                         "global memory (SubgroupSize != FactorSG)");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            subgroup_impl_local2global_strided_copy<detail::level::WORKGROUP, 3, 3, 3>(
                output, loc_view, {output_stride * 2, output_distance * 2, 1}, {max_num_batches_local_mem * 2, 2, 1},
                i * output_distance * 2, 0, {committed_length, num_batches_in_local_mem, 2}, global_data);
          } else {
            subgroup_impl_local2global_strided_copy<detail::level::WORKGROUP, 2, 2, 2>(
                output, output_imag, loc_view, {output_stride, output_distance}, {max_num_batches_local_mem, 1},
                i * output_distance, 0, local_imag_offset, {committed_length, num_batches_in_local_mem}, global_data);
          }
        } else {
          global_data.log_message_global(
              __func__, "storing data from batch interleaved local memory to batch interleaved global memory");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            subgroup_impl_local2global_strided_copy<detail::level::WORKGROUP, 2, 2, 2>(
                output, loc_view, {2 * n_transforms, static_cast<IdxGlobal>(1)}, {2 * max_num_batches_local_mem, 1},
                2 * i, 0, {committed_length, 2 * num_batches_in_local_mem}, global_data);
          } else {
            subgroup_impl_local2global_strided_copy<detail::level::WORKGROUP, 2, 2, 2>(
                output, output_imag, loc_view, {n_transforms, static_cast<IdxGlobal>(1)},
                {max_num_batches_local_mem, 1}, i, 0, local_imag_offset, {committed_length, num_batches_in_local_mem},
                global_data);
          }
        }
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {
      // Codepath taken if input is not transposed
      Idx local_imag_offset = n_cplx_per_sg * n_sgs_in_wg;
      const IdxGlobal n_io_reals_per_fft = storage == complex_storage::INTERLEAVED_COMPLEX ? n_reals_per_fft : fft_size;
      const Idx n_io_reals_per_sg = storage == complex_storage::INTERLEAVED_COMPLEX ? n_reals_per_sg : n_cplx_per_sg;
      const Idx local_offset = subgroup_id * n_io_reals_per_sg;

      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
      if (algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
        if (is_input_packed) {
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            global2local<level::SUBGROUP, SubgroupSize>(
                global_data, input, loc_view, n_ffts_worked_on_by_sg * n_reals_per_fft,
                static_cast<IdxGlobal>(n_reals_per_fft) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)),
                subgroup_id * n_reals_per_sg);
          } else {
            global2local<level::SUBGROUP, SubgroupSize>(
                global_data, input, loc_view, n_ffts_worked_on_by_sg * fft_size,
                static_cast<IdxGlobal>(fft_size) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)),
                subgroup_id * n_cplx_per_sg);
            global2local<level::SUBGROUP, SubgroupSize>(
                global_data, input_imag, loc_view, n_ffts_worked_on_by_sg * fft_size,
                static_cast<IdxGlobal>(fft_size) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)),
                local_imag_offset + subgroup_id * n_cplx_per_sg);
          }
        } else {
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            global_data.log_message_global(__func__, "storing data from unpacked global memory to local");
            subgroup_impl_global2local_strided_copy<level::SUBGROUP, 3, 3, 3>(
                input, loc_view, {input_distance * 2, input_stride * 2, 1}, {committed_length * 2, 2, 1},
                input_distance * 2 * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)), local_offset,
                {n_ffts_worked_on_by_sg, committed_length, 2}, global_data);
          } else {
            subgroup_impl_global2local_strided_copy<level::SUBGROUP, 2, 2, 2>(
                input, input_imag, loc_view, {input_distance, input_stride}, {committed_length, 1},
                input_distance * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)), local_offset, local_imag_offset,
                {n_ffts_worked_on_by_sg, committed_length}, global_data);
          }
        }
      } else {
        if (is_input_packed) {
          auto global_ptr_offset = storage == complex_storage::INTERLEAVED_COMPLEX
                                       ? 2 * committed_length * (i - static_cast<IdxGlobal>(id_of_fft_in_sg))
                                       : committed_length * (i - static_cast<IdxGlobal>(id_of_fft_in_sg));
          auto loc_view_offset = storage == complex_storage::INTERLEAVED_COMPLEX
                                     ? 2 * factor_sg * factor_wi * subgroup_id * n_ffts_per_sg
                                     : factor_sg * factor_wi * subgroup_id * n_ffts_per_sg;
          auto loc_view_imag_offset = factor_sg * factor_wi * n_sgs_in_wg;

          subgroup_impl_bluestein_global2local_packed_copy<SubgroupSize>(
              input, input_imag, loc_view, committed_length, factor_sg * factor_wi, global_ptr_offset, loc_view_offset,
              loc_view_imag_offset, n_ffts_worked_on_by_sg, i, n_transforms, global_data.sg, storage, global_data);
        } else {
          // TODO: Bluestein Strided copy
        }
      }

      global_data.log_dump_local("data in local memory:", loc_view, n_reals_per_fft);
      sycl::group_barrier(global_data.sg);

      if (working) {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          subgroup_impl_local_private_copy<1, Idx>(
              loc_view, priv, {{{1}, {subgroup_id * n_reals_per_sg + subgroup_local_id * n_reals_per_wi}}}, factor_wi,
              global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
        } else {
          subgroup_impl_local_private_copy<1, 1, Idx>(
              loc_view, loc_view, priv, {{{1}, {subgroup_id * n_cplx_per_sg + subgroup_local_id * factor_wi}}},
              {{{2}, {0}}}, {{{1}, {subgroup_id * n_cplx_per_sg + subgroup_local_id * factor_wi + local_imag_offset}}},
              {{{2}, {1}}}, factor_wi, global_data, detail::transfer_direction::LOCAL_TO_PRIVATE);
        }
        global_data.log_dump_private("data loaded in registers:", priv, n_reals_per_wi);
      }
      sycl::group_barrier(global_data.sg);
      if (algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
        sg_dft_compute<SubgroupSize>(priv, wi_private_scratch, multiply_on_load, multiply_on_store, conjugate_on_load,
                                     conjugate_on_store, apply_scale_factor, load_modifier_data, store_modifier_data,
                                     loc_twiddles, scaling_factor,
                                     static_cast<IdxGlobal>(fft_size) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)),
                                     id_of_wi_in_fft, factor_sg, factor_wi, global_data.sg);
      } else {
        // Idx loc_view_offset = subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft;
        // subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft + 2 * id_of_wi_in_fft;
        //  subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft;
        auto loc_offset_store_view =
            storage == complex_storage::INTERLEAVED_COMPLEX
                ? subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft + 2 * id_of_wi_in_fft
                : subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft;
        auto loc_offset_load_view = storage == complex_storage::INTERLEAVED_COMPLEX
                                        ? subgroup_id * n_reals_per_sg + subgroup_local_id * n_reals_per_wi
                                        : subgroup_id * n_cplx_per_sg + subgroup_local_id * factor_wi;
        sg_bluestein<SubgroupSize>(priv, wi_private_scratch, loc_view, loc_twiddles, load_modifier_data,
                                   store_modifier_data, conjugate_on_load, conjugate_on_store, apply_scale_factor,
                                   scaling_factor, id_of_wi_in_fft, factor_sg, factor_wi, storage, working,
                                   loc_offset_store_view, loc_offset_load_view, local_imag_offset, global_data.sg,
                                   global_data);
      }
      if (working) {
        global_data.log_dump_private("data in registers after scaling:", priv, n_reals_per_wi);
      }
      if (factor_sg == SubgroupSize && is_output_packed && algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
        if (working) {
          global_data.log_message_global(__func__,
                                         "storing transposed data from private to global memory (FactorSG == "
                                         "SubgroupSize) and packed layout");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            IdxGlobal output_offset = i * static_cast<IdxGlobal>(n_reals_per_sg) +
                                      static_cast<IdxGlobal>(id_of_fft_in_sg * n_reals_per_fft) +
                                      static_cast<IdxGlobal>(id_of_wi_in_fft * 2);
            subgroup_impl_local_private_copy<1, IdxGlobal>(
                output, priv, {{{static_cast<IdxGlobal>(factor_sg)}, {output_offset}}}, factor_wi, global_data,
                detail::transfer_direction::PRIVATE_TO_GLOBAL);
          } else {
            IdxGlobal output_offset = i * static_cast<IdxGlobal>(n_cplx_per_sg) +
                                      static_cast<IdxGlobal>(id_of_fft_in_sg * fft_size) +
                                      static_cast<IdxGlobal>(id_of_wi_in_fft);
            subgroup_impl_local_private_copy<1, 1, IdxGlobal>(
                output, output_imag, priv, {{{static_cast<IdxGlobal>(factor_sg)}, {output_offset}}}, {{{2}, {0}}},
                {{{static_cast<IdxGlobal>(factor_sg)}, {output_offset}}}, {{{2}, {1}}}, factor_wi, global_data,
                detail::transfer_direction::PRIVATE_TO_GLOBAL);
          }
        }
      } else if (is_output_batch_interleaved && algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
        if (working) {
          global_data.log_message_global(__func__, "Storing data from private to Global with batch interleaved layout");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            detail::strided_view output_view{output, std::array{static_cast<IdxGlobal>(factor_sg), n_transforms},
                                             std::array{static_cast<IdxGlobal>(2 * id_of_wi_in_fft), 2 * i}};
            copy_wi<2>(global_data, priv, output_view, factor_wi);
          } else {
            detail::strided_view priv_real_view{priv, 2};
            detail::strided_view priv_imag_view{priv, 2, 1};
            detail::strided_view output_real_view{output, std::array{static_cast<IdxGlobal>(factor_sg), n_transforms},
                                                  std::array{static_cast<IdxGlobal>(id_of_wi_in_fft), i}};
            detail::strided_view output_imag_view{output_imag,
                                                  std::array{static_cast<IdxGlobal>(factor_sg), n_transforms},
                                                  std::array{static_cast<IdxGlobal>(id_of_wi_in_fft), i}};
            copy_wi(global_data, priv_real_view, output_real_view, factor_wi);
            copy_wi(global_data, priv_imag_view, output_imag_view, factor_wi);
          }
        }
      } else {
        if (working) {
          global_data.log_message_global(
              __func__, "storing transposed data from private to local memory (FactorSG != SubgroupSize)");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            Idx loc_view_offset =
                subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft + 2 * id_of_wi_in_fft;
            subgroup_impl_local_private_copy<1, Idx>(loc_view, priv, {{{factor_sg}, {loc_view_offset}}}, factor_wi,
                                                     global_data, detail::transfer_direction::PRIVATE_TO_LOCAL);
          } else {
            detail::strided_view priv_real_view{priv, 2};
            detail::strided_view priv_imag_view{priv, 2, 1};
            detail::strided_view local_real_view{
                loc_view, factor_sg, subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft};
            detail::strided_view local_imag_view{
                loc_view, factor_sg,
                subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft + local_imag_offset};
            copy_wi(global_data, priv_real_view, local_real_view, factor_wi);
            copy_wi(global_data, priv_imag_view, local_imag_view, factor_wi);
            // Idx loc_view_offset = subgroup_id * n_cplx_per_sg + id_of_fft_in_sg * fft_size + id_of_wi_in_fft;
            // subgroup_impl_local_private_copy<1, 1, Idx>(
            //     loc_view, loc_view, priv, {{{factor_sg}, {local_offset}}}, {{{2}, {0}}},
            //     {{{factor_sg}, {loc_view_offset + local_imag_offset}}}, {{{2}, {1}}}, factor_wi, global_data,
            //     detail::transfer_direction::PRIVATE_TO_LOCAL);
          }
        }
        sycl::group_barrier(global_data.sg);
        global_data.log_dump_local("computed data in local memory:", loc, n_reals_per_fft);
        global_data.log_message_global(
            __func__, "storing transposed data from local to global memory (FactorSG != SubgroupSize)");
        if (algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
          if (is_output_packed) {
            const IdxGlobal global_output_offset = n_io_reals_per_fft * (i - static_cast<IdxGlobal>(id_of_fft_in_sg));
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output,
                                                          n_ffts_worked_on_by_sg * n_reals_per_fft, local_offset,
                                                          global_output_offset);
            } else {
              local2global<level::SUBGROUP, SubgroupSize>(
                  global_data, loc_view, output, n_ffts_worked_on_by_sg * fft_size, local_offset, global_output_offset);
              local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output_imag,
                                                          n_ffts_worked_on_by_sg * fft_size,
                                                          local_offset + local_imag_offset, global_output_offset);
            }
          } else {
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              const IdxGlobal global_output_offset =
                  2 * output_distance * (i - static_cast<IdxGlobal>(id_of_fft_in_sg));
              global_data.log_message_global(__func__, "storing data from local to unpacked global memory");
              subgroup_impl_local2global_strided_copy<level::SUBGROUP, 3, 3, 3>(
                  output, loc_view, {output_distance * 2, output_stride * 2, 1}, {committed_length * 2, 2, 1},
                  global_output_offset, local_offset, {n_ffts_worked_on_by_sg, fft_size, 2}, global_data);
            } else {
              const IdxGlobal global_output_offset = output_distance * (i - static_cast<IdxGlobal>(id_of_fft_in_sg));
              subgroup_impl_local2global_strided_copy<level::SUBGROUP, 2, 2, 2>(
                  output, output_imag, loc_view, {output_distance, output_stride}, {committed_length, 1},
                  global_output_offset, local_offset, local_imag_offset, {n_ffts_worked_on_by_sg, committed_length},
                  global_data);
            }
          }
        } else {
          if (is_output_packed) {
            auto global_ptr_offset = storage == complex_storage::INTERLEAVED_COMPLEX
                                         ? 2 * committed_length * (i - static_cast<IdxGlobal>(id_of_fft_in_sg))
                                         : committed_length * (i - static_cast<IdxGlobal>(id_of_fft_in_sg));
            auto loc_view_offset = storage == complex_storage::INTERLEAVED_COMPLEX
                                       ? 2 * factor_sg * factor_wi * subgroup_id
                                       : factor_sg * factor_wi * subgroup_id;
            auto loc_view_imag_offset = factor_sg * factor_wi * n_sgs_in_wg;

            subgroup_impl_bluestein_local2global_packed_copy<SubgroupSize>(
                output, output_imag, loc_view, committed_length, factor_sg * factor_wi, global_ptr_offset,
                loc_view_offset, loc_view_imag_offset, n_ffts_worked_on_by_sg, i, n_transforms, global_data.sg, storage,
                global_data);
          } else {
            // TODO: Blustein Strided Copy
          }
        }
        sycl::group_barrier(global_data.sg);
      }
    }
  }
  global_data.log_message_global(__func__, "exited");
}

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::SUBGROUP, Dummy> {
  static Scalar* execute(committed_descriptor_impl& desc, dimension_struct& dimension_data,
                         std::vector<kernel_data_struct>& kernels) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    const auto& kernel_data = kernels.at(0);
    Idx factor_wi = kernel_data.factors[0];
    Idx factor_sg = kernel_data.factors[1];
    std::size_t twiddles_alloc_size = [&]() {
      if (dimension_data.is_prime) {
        // sg twiddles + load_modifiers + store_modifiers
        return 6 * dimension_data.length;
      }
      return 2 * dimension_data.length;
    }();
    PORTFFT_LOG_TRACE("Allocating global memory for twiddles for subgroup implementation. Allocation size",
                      kernel_data.length * 2);
    Scalar* res = sycl::aligned_alloc_device<Scalar>(
        alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>), twiddles_alloc_size, desc.queue);
    std::vector<Scalar> host_twiddles(twiddles_alloc_size);

    for (Idx i = 0; i < factor_sg; i++) {
      for (Idx j = 0; j < factor_wi; j++) {
        double theta = -2 * M_PI * static_cast<double>(i * j) / static_cast<double>(factor_wi * factor_sg);
        auto twiddle = std::complex<Scalar>(static_cast<Scalar>(std::cos(theta)), static_cast<Scalar>(std::sin(theta)));
        host_twiddles[static_cast<std::size_t>(j * factor_sg + i)] = twiddle.real();
        host_twiddles[static_cast<std::size_t>((j + factor_wi) * factor_sg + i)] = twiddle.imag();
      }
    }
    if (dimension_data.is_prime) {
      detail::populate_bluestein_input_modifiers(host_twiddles.data() + 2 * factor_sg * factor_wi,
                                                 dimension_data.committed_length, dimension_data.length);
      detail::populate_fft_chirp_signal(host_twiddles.data() + 4 * factor_sg * factor_wi,
                                        dimension_data.committed_length, dimension_data.length);
    }

    desc.queue.copy(host_twiddles.data(), res, twiddles_alloc_size).wait();
    return res;
  }
};

template <typename Scalar, domain Domain>
template <Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::run_kernel_struct<SubgroupSize, TIn,
                                                                    TOut>::inner<detail::level::SUBGROUP, Dummy> {
  static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                             TOut& out_imag, const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, dimension_struct& dimension_data,
                             direction compute_direction, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    auto& kernel_data = compute_direction == direction::FORWARD ? dimension_data.forward_kernels.at(0)
                                                                : dimension_data.backward_kernels.at(0);
    Scalar* twiddles = kernel_data.twiddles_forward.get();
    Idx factor_sg = kernel_data.factors[1];
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg,
            input_layout);
    std::size_t global_size = static_cast<std::size_t>(detail::get_global_size_subgroup<Scalar>(
        n_transforms, factor_sg, SubgroupSize, kernel_data.num_sgs_per_wg, desc.n_compute_units));
    std::size_t twiddle_elements = 2 * kernel_data.length;
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(kernel_data.exec_bundle);
      auto in_acc_or_usm = detail::get_access(in, cgh);
      auto out_acc_or_usm = detail::get_access(out, cgh);
      auto in_imag_acc_or_usm = detail::get_access(in_imag, cgh);
      auto out_imag_acc_or_usm = detail::get_access(out_imag, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
      auto fft_size = dimension_data.length;
#ifdef PORTFFT_KERNEL_LOG
      sycl::stream s{1024 * 16 * 16, 1024 * 8, cgh};
#endif
      PORTFFT_LOG_TRACE("Launching subgroup kernel with global_size", global_size, "local_size",
                        SubgroupSize * kernel_data.num_sgs_per_wg, "local memory allocation of size", local_elements,
                        "local memory allocation for twiddles of size", twiddle_elements);
      cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Mem, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * kernel_data.num_sgs_per_wg)}},
          [=
#ifdef PORTFFT_KERNEL_LOG
               ,
           global_logging_config = detail::global_logging_config
#endif
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_KERNEL_LOG
                s, global_logging_config,
#endif
                it};
            global_data.log_message_global("Running subgroup kernel");
            detail::fft_algorithm algorithm = kh.get_specialization_constant<detail::SpecConstFFTAlgorithm>();
            if (algorithm == detail::fft_algorithm::COOLEY_TUKEY) {
              detail::subgroup_impl<SubgroupSize>(&in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                                                  &in_imag_acc_or_usm[0] + input_offset,
                                                  &out_imag_acc_or_usm[0] + output_offset, &loc[0], &loc_twiddles[0],
                                                  n_transforms, twiddles, global_data, kh);
            } else {
              auto loc_ptr = &loc[0];
              for (auto idx = global_data.it.get_local_id(0); idx < local_elements;
                   idx += global_data.it.get_local_range(0)) {
                loc_ptr[idx] = 0;
              }
              sycl::group_barrier(global_data.it.get_group());
              detail::subgroup_impl<SubgroupSize>(&in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                                                  &in_imag_acc_or_usm[0] + input_offset,
                                                  &out_imag_acc_or_usm[0] + output_offset, loc_ptr, &loc_twiddles[0],
                                                  n_transforms, twiddles, global_data, kh, twiddles + 2 * fft_size,
                                                  twiddles + 4 * fft_size);
            }
            global_data.log_message_global("Exiting subgroup kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::SUBGROUP, Dummy> {
  static void execute(committed_descriptor_impl& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      Idx /*length*/, const std::vector<Idx>& factors, detail::level /*level*/, Idx /*factor_num*/,
                      Idx /*num_factors*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("SubgroupFactorWISpecConst:", factors[0]);
    in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[0]);
    PORTFFT_LOG_TRACE("SubgroupFactorSGSpecConst:", factors[1]);
    in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[1]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::SUBGROUP,
                                                                                         Dummy> {
  static std::size_t execute(committed_descriptor_impl& desc, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& factors, Idx& num_sgs_per_wg, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    Idx dft_length = static_cast<Idx>(length);
    Idx twiddle_bytes = 2 * dft_length * static_cast<Idx>(sizeof(Scalar));
    if (input_layout == detail::layout::BATCH_INTERLEAVED) {
      Idx padded_fft_bytes = detail::pad_local(2 * dft_length, Idx(1)) * static_cast<Idx>(sizeof(Scalar));
      Idx max_batches_in_local_mem = (desc.local_memory_size - twiddle_bytes) / padded_fft_bytes;
      Idx batches_per_sg = used_sg_size / 2;
      Idx num_sgs_required =
          std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_batches_in_local_mem / batches_per_sg));
      num_sgs_per_wg = num_sgs_required;
      Idx num_batches_in_local_mem = used_sg_size * num_sgs_per_wg / 2;
      return static_cast<std::size_t>(detail::pad_local(2 * dft_length * num_batches_in_local_mem, 1));
    }

    Idx factor_sg = factors[1];
    Idx n_ffts_per_sg = used_sg_size / factor_sg;
    Idx num_scalars_per_sg = detail::pad_local(2 * dft_length * n_ffts_per_sg, 1);
    Idx max_n_sgs = (desc.local_memory_size - twiddle_bytes) / static_cast<Idx>(sizeof(Scalar)) / num_scalars_per_sg;
    num_sgs_per_wg = std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_n_sgs));
    // recalculate padding since `num_scalars_per_sg` is a floored value
    Idx res = detail::pad_local(2 * dft_length * n_ffts_per_sg * num_sgs_per_wg, 1);
    return static_cast<std::size_t>(res);
  }
};

}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
