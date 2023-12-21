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

#include "portfft/common/helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/memory_views.hpp"
#include "portfft/common/subgroup.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/defines.hpp"
#include "portfft/descriptor.hpp"
#include "portfft/enums.hpp"
#include "portfft/specialization_constant.hpp"

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
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam LayoutIn Input Layout
 * @tparam LayoutOut Output Layout
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param output accessor or pointer to global memory for output data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param input accessor or pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param output accessor or pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param loc local accessor. Must have enough space for 2*FactorWI*FactorSG*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*FactorWI*FactorSG
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 * @param loc_load_modifier Pointer to load modifier data in local memory
 * @param loc_store_modifier Pointer to store modifier data in local memory
 */
template <direction Dir, Idx SubgroupSize, detail::layout LayoutIn, detail::layout LayoutOut, typename T>
PORTFFT_INLINE void subgroup_impl(const T* input, T* output, const T* input_imag, T* output_imag, T* loc,
                                  T* loc_twiddles, IdxGlobal n_transforms, const T* twiddles, T scaling_factor,
                                  global_data_struct<1> global_data, sycl::kernel_handler& kh,
                                  const T* load_modifier_data = nullptr, const T* store_modifier_data = nullptr,
                                  T* loc_load_modifier = nullptr, T* loc_store_modifier = nullptr) {
  complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  detail::elementwise_multiply multiply_on_load = kh.get_specialization_constant<detail::SpecConstMultiplyOnLoad>();
  detail::elementwise_multiply multiply_on_store = kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  detail::apply_scale_factor apply_scale_factor = kh.get_specialization_constant<detail::SpecConstApplyScaleFactor>();

  const Idx factor_wi = kh.get_specialization_constant<SubgroupFactorWISpecConst>();
  const Idx factor_sg = kh.get_specialization_constant<SubgroupFactorSGSpecConst>();
  global_data.log_message_global(__func__, "entered", "FactorWI", factor_wi, "FactorSG", factor_sg, "n_transforms",
                                 n_transforms);
  const Idx n_reals_per_wi = 2 * factor_wi;

#ifdef PORTFFT_USE_SCLA
  T wi_private_scratch[detail::SpecConstWIScratchSize];
  T priv[detail::SpecConstNumRealsPerFFT];
#else
  T wi_private_scratch[2 * wi_temps(detail::MaxComplexPerWI)];
  T priv[2 * MaxComplexPerWI];
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
  Idx id_of_fft_in_sg = subgroup_local_id / factor_sg;
  Idx id_of_wi_in_fft = subgroup_local_id % factor_sg;
  Idx n_ffts_per_wg = n_ffts_per_sg * n_sgs_in_wg;

  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  IdxGlobal rounded_up_n_ffts = round_up_to_multiple(n_transforms, static_cast<IdxGlobal>(n_ffts_per_wg)) +
                                (subgroup_local_id >= max_wis_working);

  IdxGlobal id_of_fft_in_kernel;
  IdxGlobal n_ffts_in_kernel;
  if (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
    id_of_fft_in_kernel = static_cast<IdxGlobal>(global_data.it.get_group(0) * global_data.it.get_local_range(0)) / 2;
    n_ffts_in_kernel = static_cast<Idx>(global_data.it.get_group_range(0)) * local_size / 2;
  } else {
    id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
    n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  }

  constexpr Idx BankLinesPerPad = 1;
  auto loc_view = detail::padded_view(loc, BankLinesPerPad);
  auto loc_load_modifier_view = detail::padded_view(loc_load_modifier, BankLinesPerPad);
  auto loc_store_modifier_view = detail::padded_view(loc_store_modifier, BankLinesPerPad);

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles, n_reals_per_fft);
  sycl::group_barrier(global_data.it.get_group());
  global_data.log_dump_local("twiddles in local memory:", loc_twiddles, n_reals_per_fft);

  for (IdxGlobal i = static_cast<IdxGlobal>(id_of_fft_in_kernel); i < rounded_up_n_ffts;
       i += static_cast<IdxGlobal>(n_ffts_in_kernel)) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    Idx n_ffts_worked_on_by_sg = sycl::min(static_cast<Idx>(n_transforms - i) + id_of_fft_in_sg, n_ffts_per_sg);

    if (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      /**
       * Codepath taken if the input is transposed
       * The number of batches that are loaded, is equal to half of the workgroup size.
       * Each workitem is responsible for all of either the real or complex part of the batch being loaded.
       * The data in local memory is also stored in a transposed manner, so that there are no bank conflicts
       * while storing the data.
       * Thus it is loaded in a transposed manner and stored in a transposed manner to prevent data overwrites.
       * Going ahead with the assumption that output will not be stored in a transposed manner(always out of place), it
       * would need to transpose the final result in local memory and store it to global.
       */
      Idx id_of_fft_in_sub_batch = static_cast<Idx>(global_data.sg.get_group_id()) * n_ffts_per_sg + id_of_fft_in_sg;
      // TODO should we make sure that: max_num_batches_local_mem >= n_ffts_per_wg ?
      Idx max_num_batches_local_mem = n_sgs_in_wg * SubgroupSize / 2;
      Idx num_batches_in_local_mem = [=]() {
        if (i + static_cast<IdxGlobal>(local_size) / 2 < n_transforms) {
          return local_size / 2;
        }
        return static_cast<Idx>(n_transforms - i);
      }();
      Idx rounded_up_sub_batches = detail::round_up_to_multiple(num_batches_in_local_mem, n_ffts_per_sg);
      Idx local_imag_offset = factor_wi * factor_sg * max_num_batches_local_mem;
      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        global_data.log_message_global(__func__, "loading load multipliers from global to local memory");
        global2local<detail::level::WORKGROUP, SubgroupSize>(global_data, load_modifier_data, loc_load_modifier_view,
                                                             n_reals_per_fft * num_batches_in_local_mem,
                                                             i * n_reals_per_fft);
      }
      // TODO: Replace this with Async DMA where the hardware supports it.
      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        global_data.log_message_global(__func__, "loading store multipliers from global to local memory");
        global2local<detail::level::WORKGROUP, SubgroupSize>(global_data, store_modifier_data, loc_store_modifier_view,
                                                             n_reals_per_fft * num_batches_in_local_mem,
                                                             i * n_reals_per_fft);
      }
      global_data.log_message_global(__func__, "loading transposed data from global to local memory");
      // load / store in a transposed manner
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        detail::md_view input_view{input, std::array{2 * n_transforms, static_cast<IdxGlobal>(1)}, 2 * i};
        detail::md_view local_md_view{loc_view, std::array{2 * max_num_batches_local_mem, 1}};
        copy_group<level::WORKGROUP>(global_data, input_view, local_md_view,
                                     std::array{fft_size, 2 * num_batches_in_local_mem});
      } else {
        detail::md_view input_real_view{input, std::array{n_transforms, static_cast<IdxGlobal>(1)}, i};
        detail::md_view input_imag_view{input_imag, std::array{n_transforms, static_cast<IdxGlobal>(1)}, i};
        detail::md_view local_real_view{loc_view, std::array{max_num_batches_local_mem, 1}};
        detail::md_view local_imag_view{loc_view, std::array{max_num_batches_local_mem, 1}, local_imag_offset};
        global_data.log_message_global(__func__, "params", max_num_batches_local_mem, fft_size,
                                       num_batches_in_local_mem);
        global_data.log_message_global(__func__, "loading transposed real data from global to local memory");
        copy_group<level::WORKGROUP>(global_data, input_real_view, local_real_view,
                                     std::array{fft_size, num_batches_in_local_mem});
        global_data.log_message_global(__func__, "loading transposed imag data from global to local memory");
        copy_group<level::WORKGROUP>(global_data, input_imag_view, local_imag_view,
                                     std::array{fft_size, num_batches_in_local_mem});
      }
      sycl::group_barrier(global_data.it.get_group());
      global_data.log_dump_local("data loaded to local memory:", loc_view,
                                 n_reals_per_wi * factor_sg * max_num_batches_local_mem);
      for (Idx sub_batch = id_of_fft_in_sub_batch; sub_batch < rounded_up_sub_batches;
           sub_batch += n_sgs_in_wg * n_ffts_per_sg) {
        bool working_inner = sub_batch < num_batches_in_local_mem && subgroup_local_id < max_wis_working;
        if (working_inner) {
          global_data.log_message_global(__func__, "loading transposed data from local to private memory");
          // load from local memory in a transposed manner
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            detail::strided_view strided_local_view{loc_view, std::array{1, max_num_batches_local_mem},
                                                    std::array{2 * id_of_wi_in_fft * factor_wi, 2 * sub_batch}};
            copy_wi<2>(global_data, strided_local_view, priv, factor_wi);
          } else {
            detail::strided_view local_real_view{loc_view, std::array{1, max_num_batches_local_mem},
                                                 std::array{id_of_wi_in_fft * factor_wi, sub_batch}};
            detail::strided_view local_imag_view{
                loc_view, std::array{1, max_num_batches_local_mem},
                std::array{id_of_wi_in_fft * factor_wi, sub_batch + local_imag_offset}};
            detail::strided_view priv_real_view{priv, 2};
            detail::strided_view priv_imag_view{priv, 2, 1};
            copy_wi(global_data, local_real_view, priv_real_view, factor_wi);
            copy_wi(global_data, local_imag_view, priv_imag_view, factor_wi);
          }
          global_data.log_dump_private("data loaded in registers:", priv, n_reals_per_wi);
        }
        if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
          // Note: if using load modifier, this data need to be stored in the transposed fashion per batch to ensure
          // low latency reads from shared memory, as this will result in much lesser bank conflicts.
          // Tensor shape for load modifier in local memory = num_batches_in_local_mem x  FactorWI x FactorSG
          // TODO: change the above mentioned layout to the following tenshor shape: num_batches_in_local_mem x
          // n_ffts_in_sg x FactorWI x FactorSG
          global_data.log_message_global(__func__, "multiplying load modifier data");
          if (working_inner) {
            PORTFFT_UNROLL
            for (Idx j = 0; j < factor_wi; j++) {
              Idx base_offset = sub_batch * n_reals_per_fft + 2 * j * factor_sg + 2 * id_of_wi_in_fft;
              multiply_complex(priv[2 * j], priv[2 * j + 1], loc_load_modifier_view[base_offset],
                               loc_load_modifier_view[base_offset + 1], priv[2 * j], priv[2 * j + 1]);
            }
          }
        }
        sg_dft<Dir, SubgroupSize>(priv, global_data.sg, factor_wi, factor_sg, loc_twiddles, wi_private_scratch);
        if (working_inner) {
          global_data.log_dump_private("data in registers after computation:", priv, n_reals_per_wi);
        }
        if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
          // No need to store the store modifier data in a transposed fashion as data after sg_dft is already transposed
          // Tensor Shape for store modifier is num_batches_in_local_memory x FactorSG x FactorWI
          global_data.log_message_global(__func__, "multiplying store modifier data");
          if (working_inner) {
            PORTFFT_UNROLL
            for (Idx j = 0; j < factor_wi; j++) {
              sycl::vec<T, 2> modifier_priv;
              Idx base_offset = sub_batch * n_reals_per_fft + 2 * j * factor_sg + 2 * id_of_wi_in_fft;
              // TODO: this leads to compilation error on AMD. Revert back to this once it is resolved
              // modifier_priv.load(0, detail::get_local_multi_ptr(&loc_store_modifier_view[base_offset]));
              modifier_priv[0] = loc_store_modifier_view[base_offset];
              modifier_priv[1] = loc_store_modifier_view[base_offset + 1];
              if (Dir == direction::BACKWARD) {
                modifier_priv[1] *= -1;
              }
              multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_priv[0], modifier_priv[1], priv[2 * j],
                               priv[2 * j + 1]);
            }
          }
        }
        if (apply_scale_factor == detail::apply_scale_factor::APPLIED) {
          PORTFFT_UNROLL
          for (Idx idx = 0; idx < factor_wi; idx++) {
            priv[2 * idx] *= scaling_factor;
            priv[2 * idx + 1] *= scaling_factor;
          }
        }
        // Async DMA can start here for the next set of load/store modifiers.
        if (working_inner) {
          global_data.log_dump_private("data in registers after scaling:", priv, n_reals_per_wi);
        }
        if (SubgroupSize == factor_sg && LayoutOut == detail::layout::PACKED) {
          if (working_inner) {
            global_data.log_message_global(
                __func__, "storing transposed data from private to global memory (SubgroupSize == FactorSG)");
            // Store directly from registers for fully coalesced accesses
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              detail::strided_view output_view{
                  output, static_cast<IdxGlobal>(factor_sg),
                  (i + static_cast<IdxGlobal>(sub_batch)) * static_cast<IdxGlobal>(n_reals_per_fft) +
                      static_cast<IdxGlobal>(2 * id_of_wi_in_fft)};
              copy_wi<2>(global_data, priv, output_view, factor_wi);
            } else {
              detail::strided_view output_real_view{
                  output, static_cast<IdxGlobal>(factor_sg),
                  (i + static_cast<IdxGlobal>(sub_batch)) * static_cast<IdxGlobal>(fft_size) +
                      static_cast<IdxGlobal>(id_of_wi_in_fft)};
              detail::strided_view output_imag_view{
                  output_imag, static_cast<IdxGlobal>(factor_sg),
                  (i + static_cast<IdxGlobal>(sub_batch)) * static_cast<IdxGlobal>(fft_size) +
                      static_cast<IdxGlobal>(id_of_wi_in_fft)};
              detail::strided_view priv_real_view{priv, 2};
              detail::strided_view priv_imag_view{priv, 2, 1};
              copy_wi(global_data, priv_real_view, output_real_view, factor_wi);
              copy_wi(global_data, priv_imag_view, output_imag_view, factor_wi);
            }
          }
        } else {
          if (working_inner) {
            global_data.log_message_global(__func__,
                                           "storing transposed data from private to local memory (SubgroupSize != "
                                           "FactorSG or LayoutOut == detail::layout::BATCH_INTERLEAVED)");
            // Store back to local memory only
            if (storage == complex_storage::INTERLEAVED_COMPLEX) {
              detail::strided_view strided_local_view{loc_view, std::array{factor_sg, max_num_batches_local_mem},
                                                      std::array{2 * id_of_wi_in_fft, 2 * sub_batch}};
              copy_wi<2>(global_data, priv, strided_local_view, factor_wi);
            } else {
              detail::strided_view local_real_view{loc_view, std::array{factor_sg, max_num_batches_local_mem},
                                                   std::array{id_of_wi_in_fft, sub_batch}};
              detail::strided_view local_imag_view{loc_view, std::array{factor_sg, max_num_batches_local_mem},
                                                   std::array{id_of_wi_in_fft, sub_batch + local_imag_offset}};
              detail::strided_view priv_real_view{priv, 2};
              detail::strided_view priv_imag_view{priv, 2, 1};
              copy_wi(global_data, priv_real_view, local_real_view, factor_wi);
              copy_wi(global_data, priv_imag_view, local_imag_view, factor_wi);
            }
          }
        }
      }
      sycl::group_barrier(global_data.it.get_group());
      if (SubgroupSize != factor_sg || LayoutOut == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_dump_local("computed data in local memory:", loc_view, n_reals_per_wi * factor_sg);
        // store back all loaded batches at once.
        if (LayoutOut == detail::layout::PACKED) {
          global_data.log_message_global(__func__,
                                         "storing transposed data from local to global memory (SubgroupSize != "
                                         "FactorSG) with LayoutOut = detail::layout::PACKED");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            detail::md_view local_md_view2{loc_view, std::array{2 * max_num_batches_local_mem, 1, 2}};
            detail::md_view output_view{output, std::array{2, 1, 2 * fft_size}, i * n_reals_per_fft};
            copy_group<level::WORKGROUP>(global_data, local_md_view2, output_view,
                                         std::array{fft_size, 2, num_batches_in_local_mem});
          } else {
            detail::md_view local_real_view{loc_view, std::array{max_num_batches_local_mem, 1}};
            detail::md_view local_imag_view{loc_view, std::array{max_num_batches_local_mem, 1}, local_imag_offset};
            detail::md_view output_real_view{output, std::array{1, fft_size}, i * fft_size};
            detail::md_view output_imag_view{output_imag, std::array{1, fft_size}, i * fft_size};
            copy_group<level::WORKGROUP>(global_data, local_real_view, output_real_view,
                                         std::array{fft_size, num_batches_in_local_mem});
            copy_group<level::WORKGROUP>(global_data, local_imag_view, output_imag_view,
                                         std::array{fft_size, num_batches_in_local_mem});
          }
        } else {
          global_data.log_message_global(__func__,
                                         "storing transposed data from local memory to global memory with LayoutOut == "
                                         "detail::layout::BATCH_INTERLEAVED");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            detail::md_view local_md_view2{loc_view, std::array{2 * max_num_batches_local_mem, 1}};
            detail::md_view output_view{output, std::array{2 * n_transforms, static_cast<IdxGlobal>(1)}, 2 * i};
            copy_group<level::WORKGROUP>(global_data, local_md_view2, output_view,
                                         std::array{factor_wi * factor_sg, 2 * num_batches_in_local_mem});
          } else {
            detail::md_view local_real_view{loc_view, std::array{max_num_batches_local_mem, 1}};
            detail::md_view local_imag_view{loc_view, std::array{max_num_batches_local_mem, 1}, local_imag_offset};
            detail::md_view output_real_view{output, std::array{n_transforms, static_cast<IdxGlobal>(1)}, i};
            detail::md_view output_imag_view{output_imag, std::array{n_transforms, static_cast<IdxGlobal>(1)}, i};
            copy_group<level::WORKGROUP>(global_data, local_real_view, output_real_view,
                                         std::array{factor_wi * factor_sg, num_batches_in_local_mem});
            copy_group<level::WORKGROUP>(global_data, local_imag_view, output_imag_view,
                                         std::array{factor_wi * factor_sg, num_batches_in_local_mem});
          }
        }
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {
      // Codepath taken if input is not transposed
      Idx local_imag_offset = n_cplx_per_sg * n_sgs_in_wg;

      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
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
      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        global_data.log_message_global(__func__, "loading load modifier data");
        global2local<detail::level::SUBGROUP, SubgroupSize>(
            global_data, load_modifier_data, loc_load_modifier_view, n_ffts_worked_on_by_sg * n_reals_per_fft,
            n_reals_per_fft * (i - id_of_fft_in_sg), subgroup_id * n_reals_per_sg);
      }
      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        global_data.log_message_global(__func__, "loading store modifier data");
        global2local<detail::level::SUBGROUP, SubgroupSize>(
            global_data, store_modifier_data, loc_store_modifier_view, n_ffts_worked_on_by_sg * n_reals_per_fft,
            n_reals_per_fft * (i - id_of_fft_in_sg), subgroup_id * n_reals_per_sg);
      }
      sycl::group_barrier(global_data.sg);
      global_data.log_dump_local("data in local memory:", loc_view, n_reals_per_fft);
      if (working) {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::offset_view offset_local_view{loc_view,
                                                subgroup_id * n_reals_per_sg + subgroup_local_id * n_reals_per_wi};
          copy_wi(global_data, offset_local_view, priv, n_reals_per_wi);
        } else {
          detail::offset_view local_real_view{loc_view, subgroup_id * n_cplx_per_sg + subgroup_local_id * factor_wi};
          detail::offset_view local_imag_view{
              loc_view, subgroup_id * n_cplx_per_sg + subgroup_local_id * factor_wi + local_imag_offset};
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          copy_wi(global_data, local_real_view, priv_real_view, factor_wi);
          copy_wi(global_data, local_imag_view, priv_imag_view, factor_wi);
        }
        global_data.log_dump_private("data loaded in registers:", priv, n_reals_per_wi);
      }
      sycl::group_barrier(global_data.sg);
      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        if (working) {
          global_data.log_message_global(__func__, "Multiplying load modifier before sg_dft");
          PORTFFT_UNROLL
          for (Idx j = 0; j < factor_wi; j++) {
            Idx base_offset = static_cast<Idx>(global_data.sg.get_group_id()) * n_ffts_per_sg +
                              id_of_fft_in_sg * n_reals_per_fft + 2 * j * factor_sg + 2 * id_of_wi_in_fft;
            multiply_complex(priv[2 * j], priv[2 * j + 1], loc_load_modifier_view[base_offset],
                             loc_load_modifier_view[base_offset + 1], priv[2 * j], priv[2 * j + 1]);
          }
        }
      }
      sg_dft<Dir, SubgroupSize>(priv, global_data.sg, factor_wi, factor_sg, loc_twiddles, wi_private_scratch);
      if (working) {
        global_data.log_dump_private("data in registers after computation:", priv, n_reals_per_wi);
      }
      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        if (working) {
          global_data.log_message_global(__func__, "Multiplying store modifier before sg_dft");
          PORTFFT_UNROLL
          for (Idx j = 0; j < factor_wi; j++) {
            sycl::vec<T, 2> modifier_priv;
            Idx base_offset = static_cast<Idx>(global_data.it.get_sub_group().get_group_id()) * n_ffts_per_sg +
                              id_of_fft_in_sg * n_reals_per_fft + 2 * j * factor_sg + 2 * id_of_wi_in_fft;
            // modifier_priv.load(0, detail::get_local_multi_ptr(&loc_store_modifier_view[base_offset]));
            modifier_priv[0] = loc_store_modifier_view[base_offset];
            modifier_priv[1] = loc_store_modifier_view[base_offset + 1];
            if (Dir == direction::BACKWARD) {
              modifier_priv[1] *= -1;
            }
            multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_priv[0], modifier_priv[1], priv[2 * j],
                             priv[2 * j + 1]);
          }
        }
      }
      if (apply_scale_factor == detail::apply_scale_factor::APPLIED) {
        PORTFFT_UNROLL
        for (Idx j = 0; j < factor_wi; j++) {
          priv[2 * j] *= scaling_factor;
          priv[2 * j + 1] *= scaling_factor;
        }
      }
      if (working) {
        global_data.log_dump_private("data in registers after scaling:", priv, n_reals_per_wi);
      }
      if (factor_sg == SubgroupSize && LayoutOut == detail::layout::PACKED) {
        // in this case we get fully coalesced memory access even without going through local memory
        // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
        if (working) {
          global_data.log_message_global(__func__,
                                         "storing transposed data from private to global memory (FactorSG == "
                                         "SubgroupSize) and LayoutOut == detail::level::PACKED");
          if (storage == complex_storage::INTERLEAVED_COMPLEX) {
            detail::strided_view output_view{output, static_cast<IdxGlobal>(factor_sg),
                                             i * static_cast<IdxGlobal>(n_reals_per_sg) +
                                                 static_cast<IdxGlobal>(id_of_fft_in_sg * n_reals_per_fft) +
                                                 static_cast<IdxGlobal>(id_of_wi_in_fft * 2)};
            copy_wi<2>(global_data, priv, output_view, factor_wi);
          } else {
            detail::strided_view priv_real_view{priv, 2};
            detail::strided_view priv_imag_view{priv, 2, 1};
            detail::strided_view output_real_view{output, static_cast<IdxGlobal>(factor_sg),
                                                  i * static_cast<IdxGlobal>(n_cplx_per_sg) +
                                                      static_cast<IdxGlobal>(id_of_fft_in_sg * fft_size) +
                                                      static_cast<IdxGlobal>(id_of_wi_in_fft)};
            detail::strided_view output_imag_view{output_imag, static_cast<IdxGlobal>(factor_sg),
                                                  i * static_cast<IdxGlobal>(n_cplx_per_sg) +
                                                      static_cast<IdxGlobal>(id_of_fft_in_sg * fft_size) +
                                                      static_cast<IdxGlobal>(id_of_wi_in_fft)};
            copy_wi(global_data, priv_real_view, output_real_view, factor_wi);
            copy_wi(global_data, priv_imag_view, output_imag_view, factor_wi);
          }
        }
      } else if (LayoutOut == detail::layout::BATCH_INTERLEAVED) {
        if (working) {
          global_data.log_message_global(
              __func__, "Storing data from private to Global with LayoutOut == detail::level::BATCH_INTERLEAVED");
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
            detail::strided_view strided_local_view{
                loc_view, factor_sg,
                subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft + 2 * id_of_wi_in_fft};
            copy_wi<2>(global_data, priv, strided_local_view, factor_wi);
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
          }
        }
        sycl::group_barrier(global_data.sg);
        global_data.log_dump_local("computed data in local memory:", loc, n_reals_per_fft);
        global_data.log_message_global(
            __func__, "storing transposed data from local to global memory (FactorSG != SubgroupSize)");
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          local2global<level::SUBGROUP, SubgroupSize>(
              global_data, loc_view, output, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_id * n_reals_per_sg,
              static_cast<IdxGlobal>(n_reals_per_fft) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)));
        } else {
          local2global<level::SUBGROUP, SubgroupSize>(
              global_data, loc_view, output, n_ffts_worked_on_by_sg * fft_size, subgroup_id * n_cplx_per_sg,
              static_cast<IdxGlobal>(fft_size) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)));
          local2global<level::SUBGROUP, SubgroupSize>(
              global_data, loc_view, output_imag, n_ffts_worked_on_by_sg * fft_size,
              subgroup_id * n_cplx_per_sg + local_imag_offset,
              static_cast<IdxGlobal>(fft_size) * (i - static_cast<IdxGlobal>(id_of_fft_in_sg)));
        }
        sycl::group_barrier(global_data.sg);
      }
    }
  }
  global_data.log_message_global(__func__, "exited");
}
}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::SUBGROUP, Dummy> {
  static Scalar* execute(committed_descriptor& desc, dimension_struct& dimension_data) {
    const auto& kernel_data = dimension_data.kernels.at(0);
    Idx factor_wi = kernel_data.factors[0];
    Idx factor_sg = kernel_data.factors[1];
    Scalar* res = sycl::aligned_alloc_device<Scalar>(
        alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>), kernel_data.length * 2, desc.queue);
    sycl::range<2> kernel_range({static_cast<std::size_t>(factor_sg), static_cast<std::size_t>(factor_wi)});
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(kernel_range, [=](sycl::item<2> it) {
        Idx n = static_cast<Idx>(it.get_id(0));
        Idx k = static_cast<Idx>(it.get_id(1));
        sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
      });
    });
    desc.queue.wait();  // waiting once here can be better than depending on the event
                        // for all future calls to compute
    return res;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
          typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::SUBGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                             const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, Scalar scale_factor,
                             dimension_struct& dimension_data) {
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    auto& kernel_data = dimension_data.kernels.at(0);
    Scalar* twiddles = kernel_data.twiddles_forward.get();
    Idx factor_sg = kernel_data.factors[1];
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, LayoutIn, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg);
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
#ifdef PORTFFT_LOG
      sycl::stream s{1024 * 16 * 16, 1024 * 8, cgh};
#endif
      cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Dir, Mem, LayoutIn, LayoutOut, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * kernel_data.num_sgs_per_wg)}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
                s,
#endif
                it};
            global_data.log_message_global("Running subgroup kernel");
            detail::subgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut>(
                &in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                &in_imag_acc_or_usm[0] + input_offset, &out_imag_acc_or_usm[0] + output_offset, &loc[0],
                &loc_twiddles[0], n_transforms, twiddles, scale_factor, global_data, kh);
            global_data.log_message_global("Exiting subgroup kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::SUBGROUP, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t /*length*/, const std::vector<Idx>& factors, detail::level /*level*/,
                      Idx /*factor_num*/, Idx /*num_factors*/) {
    in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[0]);
    in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[1]);
  }
};

template <typename Scalar, domain Domain>
template <detail::layout LayoutIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::SUBGROUP, LayoutIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& factors, Idx& num_sgs_per_wg) {
    Idx dft_length = static_cast<Idx>(length);
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      Idx twiddle_bytes = 2 * dft_length * static_cast<Idx>(sizeof(Scalar));
      Idx padded_fft_bytes = detail::pad_local(2 * dft_length, Idx(1)) * static_cast<Idx>(sizeof(Scalar));
      Idx max_batches_in_local_mem = (desc.local_memory_size - twiddle_bytes) / padded_fft_bytes;
      Idx batches_per_sg = used_sg_size / 2;
      Idx num_sgs_required =
          std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_batches_in_local_mem / batches_per_sg));
      num_sgs_per_wg = num_sgs_required;
      Idx num_batches_in_local_mem = used_sg_size * num_sgs_per_wg / 2;
      return static_cast<std::size_t>(detail::pad_local(2 * dft_length * num_batches_in_local_mem, 1));
    } else {
      Idx factor_sg = factors[1];
      Idx n_ffts_per_sg = used_sg_size / factor_sg;
      Idx num_scalars_per_sg = detail::pad_local(2 * dft_length * n_ffts_per_sg, 1);
      Idx max_n_sgs = desc.local_memory_size / static_cast<Idx>(sizeof(Scalar)) / num_scalars_per_sg;
      num_sgs_per_wg = std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_n_sgs));
      Idx res = num_scalars_per_sg * num_sgs_per_wg;
      return static_cast<std::size_t>(res);
    }
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_SUBGROUP_DISPATCHER_HPP
