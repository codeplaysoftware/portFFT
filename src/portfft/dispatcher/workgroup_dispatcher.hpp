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

#ifndef PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP

#include "portfft/common/helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/memory_views.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/common/workgroup.hpp"
#include "portfft/defines.hpp"
#include "portfft/descriptor.hpp"
#include "portfft/enums.hpp"
#include "portfft/specialization_constant.hpp"

namespace portfft {
namespace detail {
/**
 * Calculates the number of batches that will be loaded into local memory at any one time for the work-group
 * implementation.
 *
 * @param is_batch_interleaved is the input data layout batch interleaved
 * @param workgroup_size The size of the work-group. Must be divisible by 2.
 */
PORTFFT_INLINE constexpr Idx get_num_batches_in_local_mem_workgroup(bool is_batch_interleaved,
                                                                    Idx workgroup_size) noexcept {
  return is_batch_interleaved ? workgroup_size / 2 : 1;
}

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param num_sgs_per_wg number of subgroups in a workgroup
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
IdxGlobal get_global_size_workgroup(IdxGlobal n_transforms, Idx subgroup_size, Idx num_sgs_per_wg, Idx n_compute_units,
                                    layout input_layout) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  Idx maximum_n_sgs = 8 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  Idx wg_size = subgroup_size * num_sgs_per_wg;
  Idx dfts_per_wg = get_num_batches_in_local_mem_workgroup(input_layout == layout::BATCH_INTERLEAVED, wg_size);

  return static_cast<IdxGlobal>(wg_size) * sycl::min(static_cast<IdxGlobal>(maximum_n_wgs),
                                                     divide_ceil(n_transforms, static_cast<IdxGlobal>(dfts_per_wg)));
}

/**
 * Implementation of FFT for sizes that can be done by a workgroup.
 *
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 *
 * @param input pointer to global memory containing input data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param output pointer to global memory for output data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param input_imag pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param output_imag pointer to global memory containing imaginary part of the input data if complex
 * storage (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param loc Pointer to local memory. Size requirement is determined by `num_scalars_in_local_mem_struct`.
 * @param loc_twiddles pointer to local allocation for subgroup level twiddles
 * @param n_transforms number of fft batches
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param twiddles Pointer to twiddles in the global memory
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <Idx SubgroupSize, typename T>
PORTFFT_INLINE void workgroup_impl(const T* input, T* output, const T* input_imag, T* output_imag, T* loc,
                                   T* loc_twiddles, IdxGlobal n_transforms, const T* twiddles,
                                   global_data_struct<1> global_data, sycl::kernel_handler& kh,
                                   const T* load_modifier_data = nullptr, const T* store_modifier_data = nullptr) {
  complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  detail::elementwise_multiply multiply_on_load = kh.get_specialization_constant<detail::SpecConstMultiplyOnLoad>();
  detail::elementwise_multiply multiply_on_store = kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  detail::apply_scale_factor apply_scale_factor = kh.get_specialization_constant<detail::SpecConstApplyScaleFactor>();
  detail::complex_conjugate conjugate_on_load = kh.get_specialization_constant<detail::SpecConstConjugateOnLoad>();
  detail::complex_conjugate conjugate_on_store = kh.get_specialization_constant<detail::SpecConstConjugateOnStore>();
  T scaling_factor = kh.get_specialization_constant<detail::get_spec_constant_scale<T>()>();

  const Idx fft_size = kh.get_specialization_constant<detail::SpecConstFftSize>();
  const IdxGlobal input_stride = kh.get_specialization_constant<detail::SpecConstInputStride>();
  const IdxGlobal output_stride = kh.get_specialization_constant<detail::SpecConstOutputStride>();
  const IdxGlobal input_distance = kh.get_specialization_constant<detail::SpecConstInputDistance>();
  const IdxGlobal output_distance = kh.get_specialization_constant<detail::SpecConstOutputDistance>();

  const bool is_input_batch_interleaved = input_stride == n_transforms && input_distance == 1;
  const bool is_input_packed = input_stride == 1 && input_distance == fft_size;

  global_data.log_message_global(__func__, "entered", "fft_size", fft_size, "n_transforms", n_transforms);
  Idx num_workgroups = static_cast<Idx>(global_data.it.get_group_range(0));
  Idx wg_id = static_cast<Idx>(global_data.it.get_group(0));

  Idx factor_n = detail::factorize(fft_size);
  Idx factor_m = fft_size / factor_n;
  const Idx vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
  const T* wg_twiddles = twiddles + 2 * (factor_m + factor_n);
  const Idx bank_lines_per_pad = bank_lines_per_pad_wg(2 * static_cast<Idx>(sizeof(T)) * factor_m);
  auto loc_view = padded_view(loc, bank_lines_per_pad);

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles, 2 * (factor_m + factor_n));
  global_data.log_dump_local("twiddles loaded to local memory:", loc_twiddles, 2 * (factor_m + factor_n));

  Idx max_num_batches_in_local_mem = get_num_batches_in_local_mem_workgroup(
      is_input_batch_interleaved, static_cast<Idx>(global_data.it.get_local_range(0)));

  IdxGlobal first_batch_start = static_cast<IdxGlobal>(wg_id) * static_cast<IdxGlobal>(max_num_batches_in_local_mem);
  IdxGlobal num_batches_in_kernel =
      static_cast<IdxGlobal>(num_workgroups) * static_cast<IdxGlobal>(max_num_batches_in_local_mem);
  Idx local_imag_offset = fft_size * max_num_batches_in_local_mem;

  for (IdxGlobal batch_start_idx = first_batch_start; batch_start_idx < n_transforms;
       batch_start_idx += num_batches_in_kernel) {
    IdxGlobal input_global_offset = static_cast<IdxGlobal>(vec_size * input_distance) * batch_start_idx;
    IdxGlobal output_global_offset = static_cast<IdxGlobal>(vec_size * output_distance) * batch_start_idx;
    if (is_input_batch_interleaved) {
      /**
       * In the transposed case, the data is laid out in the local memory column-wise, viewing it as a FFT_Size x
       * WG_SIZE / 2 matrix, Each column contains either the real or the complex component of the batch.  Loads WG_SIZE
       * / 2 consecutive batches into the local memory
       */
      const Idx num_batches_in_local_mem =
          std::min(max_num_batches_in_local_mem, static_cast<Idx>(n_transforms - batch_start_idx));
      global_data.log_message_global(__func__, "loading transposed data from global to local memory");
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        detail::md_view input_view{input, std::array{2 * n_transforms, static_cast<IdxGlobal>(1)}, 2 * batch_start_idx};
        detail::md_view loc_md_view{loc_view, std::array{2 * max_num_batches_in_local_mem, 1}};
        copy_group<level::WORKGROUP>(global_data, input_view, loc_md_view,
                                     std::array{fft_size, 2 * num_batches_in_local_mem});
      } else {  // storage == complex_storage::SPLIT_COMPLEX
        detail::md_view input_real_view{input, std::array{n_transforms, static_cast<IdxGlobal>(1)}, batch_start_idx};
        detail::md_view input_imag_view{input_imag, std::array{n_transforms, static_cast<IdxGlobal>(1)},
                                        batch_start_idx};
        detail::md_view loc_real_view{loc_view, std::array{max_num_batches_in_local_mem, 1}};
        detail::md_view loc_imag_view{loc_view, std::array{max_num_batches_in_local_mem, 1}, local_imag_offset};
        copy_group<level::WORKGROUP>(global_data, input_real_view, loc_real_view,
                                     std::array{fft_size, num_batches_in_local_mem});
        copy_group<level::WORKGROUP>(global_data, input_imag_view, loc_imag_view,
                                     std::array{fft_size, num_batches_in_local_mem});
      }
      sycl::group_barrier(global_data.it.get_group());

      for (Idx sub_batch = 0; sub_batch < num_batches_in_local_mem; sub_batch++) {
        wg_dft<SubgroupSize>(loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem,
                             sub_batch, batch_start_idx, load_modifier_data, store_modifier_data, fft_size, factor_n,
                             factor_m, storage, layout::BATCH_INTERLEAVED, multiply_on_load, multiply_on_store,
                             apply_scale_factor, conjugate_on_load, conjugate_on_store, global_data);
        sycl::group_barrier(global_data.it.get_group());
      }

      global_data.log_message_global(__func__, "storing data from local to global memory (with 2 transposes)");
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        std::array<IdxGlobal, 4> global_strides{2 * output_distance, 2 * factor_n * output_stride, 2 * output_stride,
                                                1};
        std::array<Idx, 4> local_strides{2, 2 * max_num_batches_in_local_mem,
                                         2 * factor_m * max_num_batches_in_local_mem, 1};
        std::array<Idx, 4> copy_lengths{num_batches_in_local_mem, factor_m, factor_n, 2};

        detail::md_view global_output_view{output, global_strides, output_global_offset};
        detail::md_view local_output_view{loc_view, local_strides};

        copy_group<level::WORKGROUP>(global_data, local_output_view, global_output_view, copy_lengths);
      } else {  // storage == complex_storage::SPLIT_COMPLEX
        std::array<IdxGlobal, 3> global_strides{output_distance, factor_n * output_stride, output_stride};
        std::array<Idx, 3> local_strides{1, max_num_batches_in_local_mem, factor_m * max_num_batches_in_local_mem};
        std::array<Idx, 3> copy_lengths{num_batches_in_local_mem, factor_m, factor_n};

        detail::md_view global_output_real_view{output, global_strides, output_global_offset};
        detail::md_view global_output_imag_view{output_imag, global_strides, output_global_offset};
        detail::md_view local_output_real_view{loc_view, local_strides};
        detail::md_view local_output_imag_view{loc_view, local_strides, local_imag_offset};

        copy_group<level::WORKGROUP>(global_data, local_output_real_view, global_output_real_view, copy_lengths);
        copy_group<level::WORKGROUP>(global_data, local_output_imag_view, global_output_imag_view, copy_lengths);
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {  // not batch interleaved input layout
      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
      if (is_input_packed) {
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          global2local<level::WORKGROUP, SubgroupSize>(global_data, input, loc_view, 2 * fft_size, input_global_offset);
        } else {
          global2local<level::WORKGROUP, SubgroupSize>(global_data, input, loc_view, fft_size, input_global_offset);
          global2local<level::WORKGROUP, SubgroupSize>(global_data, input_imag, loc_view, fft_size, input_global_offset,
                                                       local_imag_offset);
        }
      } else {
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          std::array<IdxGlobal, 2> global_strides{input_stride * 2, 1};
          std::array<Idx, 2> local_strides{2, 1};
          std::array<Idx, 2> copy_lengths{fft_size, 2};
          detail::md_view global_input_view{input, global_strides, input_global_offset};
          detail::md_view local_input_view{loc_view, local_strides};

          global_data.log_message_global(__func__, "storing data from unpacked global memory to local");
          copy_group<level::WORKGROUP>(global_data, global_input_view, local_input_view, copy_lengths);
        } else {
          detail::strided_view global_input_real_view{input, input_stride, input_global_offset};
          detail::strided_view global_input_imag_view{input_imag, input_stride, input_global_offset};
          detail::offset_view local_input_imag_view{loc_view, local_imag_offset};

          global_data.log_message_global(__func__, "storing real data from unpacked global memory to local");
          copy_group<level::WORKGROUP>(global_data, global_input_real_view, loc_view, fft_size);
          global_data.log_message_global(__func__, "storing imaginary data from unpacked global memory to local");
          copy_group<level::WORKGROUP>(global_data, global_input_imag_view, local_input_imag_view, fft_size);
        }
      }
      sycl::group_barrier(global_data.it.get_group());
      wg_dft<SubgroupSize>(loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, 0,
                           batch_start_idx, load_modifier_data, store_modifier_data, fft_size, factor_n, factor_m,
                           storage, layout::PACKED, multiply_on_load, multiply_on_store, apply_scale_factor,
                           conjugate_on_load, conjugate_on_store, global_data);
      sycl::group_barrier(global_data.it.get_group());
      global_data.log_message_global(__func__, "storing non-transposed data from local to global memory");
      // transposition for WG CT
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        std::array<IdxGlobal, 3> global_strides{2 * factor_n * output_stride, 2 * output_stride, 1};
        std::array<Idx, 3> local_strides{2, 2 * factor_m, 1};
        std::array<Idx, 3> copy_lengths{factor_m, factor_n, 2};

        detail::md_view global_output_view{output, global_strides, output_global_offset};
        detail::md_view local_output_view{loc_view, local_strides};

        copy_group<level::WORKGROUP>(global_data, local_output_view, global_output_view, copy_lengths);
      } else {
        std::array<IdxGlobal, 2> global_strides{factor_n * output_stride, output_stride};
        std::array<Idx, 2> local_strides{1, factor_m};
        std::array<Idx, 2> copy_lengths{factor_m, factor_n};

        detail::md_view global_output_real_view{output, global_strides, output_global_offset};
        detail::md_view global_output_imag_view{output_imag, global_strides, output_global_offset};
        detail::md_view loc_output_real_view{loc_view, local_strides};
        detail::md_view loc_output_imag_view{loc_view, local_strides, local_imag_offset};

        copy_group<level::WORKGROUP>(global_data, loc_output_real_view, global_output_real_view, copy_lengths);
        copy_group<level::WORKGROUP>(global_data, loc_output_imag_view, global_output_imag_view, copy_lengths);
      }
      sycl::group_barrier(global_data.it.get_group());
    }
  }
  global_data.log_message_global(__func__, "exited");
}

template <typename Scalar, domain Domain>
template <Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::run_kernel_struct<SubgroupSize, TIn,
                                                                    TOut>::inner<detail::level::WORKGROUP, Dummy> {
  static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                             TOut& out_imag, const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, dimension_struct& dimension_data,
                             direction compute_direction, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    auto& kernel_data = compute_direction == direction::FORWARD ? dimension_data.forward_kernels.at(0)
                                                                : dimension_data.backward_kernels.at(0);
    Idx num_batches_in_local_mem =
        input_layout == layout::BATCH_INTERLEAVED ? kernel_data.used_sg_size * PORTFFT_SGS_IN_WG / 2 : 1;
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    Scalar* twiddles = kernel_data.twiddles_forward.get();
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg,
            input_layout);
    std::size_t global_size = static_cast<std::size_t>(detail::get_global_size_workgroup<Scalar>(
        n_transforms, SubgroupSize, kernel_data.num_sgs_per_wg, desc.n_compute_units, input_layout));
    const Idx bank_lines_per_pad =
        bank_lines_per_pad_wg(2 * static_cast<Idx>(sizeof(Scalar)) * kernel_data.factors[2] * kernel_data.factors[3]);
    std::size_t sg_twiddles_offset = static_cast<std::size_t>(
        detail::pad_local(2 * static_cast<Idx>(kernel_data.length) * num_batches_in_local_mem, bank_lines_per_pad));
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(kernel_data.exec_bundle);
      auto in_acc_or_usm = detail::get_access(in, cgh);
      auto out_acc_or_usm = detail::get_access(out, cgh);
      auto in_imag_acc_or_usm = detail::get_access(in_imag, cgh);
      auto out_imag_acc_or_usm = detail::get_access(out_imag, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
#ifdef PORTFFT_KERNEL_LOG
      sycl::stream s{1024 * 16 * 8 * 2, 1024, cgh};
#endif
      PORTFFT_LOG_TRACE("Launching workgroup kernel with global_size", global_size, "local_size",
                        SubgroupSize * kernel_data.num_sgs_per_wg, "local memory allocation of size", local_elements);
      cgh.parallel_for<detail::workgroup_kernel<Scalar, Domain, Mem, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * PORTFFT_SGS_IN_WG)}},
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
            global_data.log_message_global("Running workgroup kernel");
            detail::workgroup_impl<SubgroupSize>(&in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                                                 &in_imag_acc_or_usm[0] + input_offset,
                                                 &out_imag_acc_or_usm[0] + output_offset, &loc[0],
                                                 &loc[0] + sg_twiddles_offset, n_transforms, twiddles, global_data, kh);
            global_data.log_message_global("Exiting workgroup kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKGROUP, Dummy> {
  static void execute(committed_descriptor_impl& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      Idx length, const std::vector<Idx>& /*factors*/, detail::level /*level*/, Idx /*factor_num*/,
                      Idx /*num_factors*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("SpecConstFftSize:", length);
    in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKGROUP,
                                                                                         Dummy> {
  static std::size_t execute(committed_descriptor_impl& /*desc*/, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& factors, Idx& /*num_sgs_per_wg*/, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    std::size_t n = static_cast<std::size_t>(factors[0]) * static_cast<std::size_t>(factors[1]);
    std::size_t m = static_cast<std::size_t>(factors[2]) * static_cast<std::size_t>(factors[3]);
    // working memory + twiddles for subgroup impl for the two sizes
    Idx num_batches_in_local_mem = detail::get_num_batches_in_local_mem_workgroup(
        input_layout == layout::BATCH_INTERLEAVED, used_sg_size * PORTFFT_SGS_IN_WG);
    return detail::pad_local(static_cast<std::size_t>(2 * num_batches_in_local_mem) * length,
                             bank_lines_per_pad_wg(2 * static_cast<std::size_t>(sizeof(Scalar)) * m)) +
           2 * (m + n);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKGROUP, Dummy> {
  static Scalar* execute(committed_descriptor_impl& desc, dimension_struct& /*dimension_data*/,
                         std::vector<kernel_data_struct>& kernels) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    const auto& kernel_data = kernels.at(0);
    Idx factor_wi_n = kernel_data.factors[0];
    Idx factor_sg_n = kernel_data.factors[1];
    Idx factor_wi_m = kernel_data.factors[2];
    Idx factor_sg_m = kernel_data.factors[3];
    Idx fft_size = static_cast<Idx>(kernel_data.length);
    Idx n = factor_wi_n * factor_sg_n;
    Idx m = factor_wi_m * factor_sg_m;
    Idx res_size = 2 * (m + n + fft_size);
    PORTFFT_LOG_TRACE("Allocating global memory for twiddles for workgroup implementation. Allocation size", res_size);
    Scalar* res =
        sycl::aligned_alloc_device<Scalar>(alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>),
                                           static_cast<std::size_t>(res_size), desc.queue);
    desc.queue.submit([&](sycl::handler& cgh) {
      PORTFFT_LOG_TRACE(
          "Launching twiddle calculation kernel for factor 1 of workgroup implementation with global size", factor_sg_n,
          factor_wi_n);
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_n), static_cast<std::size_t>(factor_wi_n)}),
                       [=](sycl::item<2> it) {
                         Idx n = static_cast<Idx>(it.get_id(0));
                         Idx k = static_cast<Idx>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_n, factor_wi_n, n, k, res + (2 * m));
                       });
    });
    desc.queue.submit([&](sycl::handler& cgh) {
      PORTFFT_LOG_TRACE(
          "Launching twiddle calculation kernel for factor 2 of workgroup implementation with global size", factor_sg_m,
          factor_wi_m);
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_m), static_cast<std::size_t>(factor_wi_m)}),
                       [=](sycl::item<2> it) {
                         Idx n = static_cast<Idx>(it.get_id(0));
                         Idx k = static_cast<Idx>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_m, factor_wi_m, n, k, res);
                       });
    });
    desc.queue.submit([&](sycl::handler& cgh) {
      PORTFFT_LOG_TRACE("Launching twiddle calculation kernel for workgroup implementation with global size", n,
                        factor_wi_m, factor_sg_m);
      cgh.parallel_for(sycl::range<3>({static_cast<std::size_t>(n), static_cast<std::size_t>(factor_wi_m),
                                       static_cast<std::size_t>(factor_sg_m)}),
                       [=](sycl::item<3> it) {
                         Idx i = static_cast<Idx>(it.get_id(0));
                         Idx j_wi = static_cast<Idx>(it.get_id(1));
                         Idx j_sg = static_cast<Idx>(it.get_id(2));
                         Idx j = j_wi + j_sg * factor_wi_m;
                         Idx j_loc = j_wi * factor_sg_m + j_sg;
                         std::complex<Scalar> twiddle = detail::calculate_twiddle<Scalar>(i * j, fft_size);
                         Idx index = 2 * (n + m + i * m + j_loc);
                         res[index] = twiddle.real();
                         res[index + 1] = twiddle.imag();
                       });
    });
    desc.queue.wait();
    return res;
  }
};

}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP
