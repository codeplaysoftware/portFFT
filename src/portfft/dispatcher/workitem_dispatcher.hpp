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

#ifndef PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP

#include "portfft/common/helpers.hpp"
#include "portfft/common/logging.hpp"
#include "portfft/common/memory_views.hpp"
#include "portfft/common/transfers.hpp"
#include "portfft/common/workitem.hpp"
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
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param num_sgs_per_wg number of subgroups in a workgroup
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
IdxGlobal get_global_size_workitem(IdxGlobal n_transforms, Idx subgroup_size, Idx num_sgs_per_wg, Idx n_compute_units) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  Idx maximum_n_sgs = 8 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  Idx wg_size = subgroup_size * num_sgs_per_wg;

  IdxGlobal n_wgs_we_can_utilize = divide_ceil(n_transforms, static_cast<IdxGlobal>(wg_size));
  return static_cast<IdxGlobal>(wg_size) * sycl::min(static_cast<IdxGlobal>(maximum_n_wgs), n_wgs_we_can_utilize);
}

/**
 * Utility function for applying load/store modifiers in workitem impl
 *
 * @tparam PrivT Private view type
 * @tparam T Type of pointer for load/store modifier global array
 * @param num_elements Num complex values per workitem
 * @param priv private memory array
 * @param modifier_data global modifier data pointer
 * @param offset offset for the global modifier data pointer
 */
template <typename PrivT, typename T>
PORTFFT_INLINE void apply_modifier(Idx num_elements, PrivT priv, const T* modifier_data, IdxGlobal offset) {
  PORTFFT_UNROLL
  for (Idx j = 0; j < num_elements; j++) {
    sycl::vec<T, 2> modifier_vec;
    modifier_vec.load(0, detail::get_global_multi_ptr(&modifier_data[offset + 2 * j]));
    multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j], priv[2 * j + 1]);
  }
}

/**
 * Implementation of FFT for sizes that can be done by independent work items.
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
 * @param loc local memory pointer. Size requirement is determined by `num_scalars_in_local_mem_struct`.
 * @param n_transforms number of FT transforms to do in one call
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param load_modifier_data Pointer to the load modifier data in global memory
 * @param store_modifier_data Pointer to the store modifier data in global memory
 */
template <Idx SubgroupSize, typename T>
PORTFFT_INLINE void workitem_impl(const T* input, T* output, const T* input_imag, T* output_imag, T* loc,
                                  IdxGlobal n_transforms, global_data_struct<1> global_data, sycl::kernel_handler& kh,
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

  const bool is_packed_input = input_stride == 1 && input_distance == fft_size;
  const bool interleaved_transforms_input = input_distance < input_stride;
  const bool is_packed_output = output_stride == 1 && output_distance == fft_size;
  const bool interleaved_transforms_output = output_distance < output_stride;

  global_data.log_message_global(__func__, "entered", "fft_size", fft_size, "n_transforms", n_transforms);

  bool interleaved_storage = storage == complex_storage::INTERLEAVED_COMPLEX;
  const Idx n_reals = 2 * fft_size;
  const Idx n_io_reals = interleaved_storage ? n_reals : fft_size;
  const IdxGlobal input_distance_in_reals = interleaved_storage ? 2 * input_distance : input_distance;
  const IdxGlobal output_distance_in_reals = interleaved_storage ? 2 * output_distance : output_distance;

#ifdef PORTFFT_USE_SCLA
  T wi_private_scratch[detail::SpecConstWIScratchSize];
  T priv_scla[detail::SpecConstNumRealsPerFFT];
  // Decay the scla to T* to avoid assert when it is decayed to const T*
  T* priv = priv_scla;
#else
  T wi_private_scratch[2 * wi_temps(detail::MaxComplexPerWI)];
  T priv[2 * MaxComplexPerWI];
#endif
  Idx subgroup_local_id = static_cast<Idx>(global_data.sg.get_local_linear_id());
  Idx subgroup_id = static_cast<Idx>(global_data.sg.get_group_id());
  Idx local_offset = n_reals * SubgroupSize * subgroup_id;
  Idx local_imag_offset = fft_size * SubgroupSize;
  constexpr Idx BankLinesPerPad = 1;
  auto loc_view = detail::padded_view(loc, BankLinesPerPad);

  const IdxGlobal transform_idx_begin = static_cast<IdxGlobal>(global_data.it.get_global_id(0));
  const IdxGlobal transform_idx_step = static_cast<IdxGlobal>(global_data.it.get_global_range(0));
  const IdxGlobal transform_idx_end = round_up_to_multiple(n_transforms, static_cast<IdxGlobal>(SubgroupSize));
  for (IdxGlobal i = transform_idx_begin; i < transform_idx_end; i += transform_idx_step) {
    const bool working = i < n_transforms;
    IdxGlobal leader_i = i - static_cast<IdxGlobal>(subgroup_local_id);

    Idx n_working = sycl::min(SubgroupSize, static_cast<Idx>(n_transforms - leader_i));
    IdxGlobal global_offset = static_cast<IdxGlobal>(n_io_reals) * leader_i;
    IdxGlobal global_input_offset = static_cast<IdxGlobal>(input_distance_in_reals) * leader_i;
    IdxGlobal global_output_offset = static_cast<IdxGlobal>(output_distance_in_reals) * leader_i;

    if (is_packed_input) {
      // copy into local memory cooperatively as a subgroup, allowing coalesced memory access for when elements of a
      // single FFT are sequential. When distance < stride, skip this step and load straight from global to registers
      // since the sequential work-items already access sequential elements.
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        global_data.log_message_global(__func__, "loading packed data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input, loc_view, n_reals * n_working, global_offset,
                                                    local_offset);
      } else {
        global_data.log_message_global(__func__, "loading packed real data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input, loc_view, fft_size * n_working, global_offset,
                                                    local_offset);
        global_data.log_message_global(__func__, "loading packed imaginary data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input_imag, loc_view, fft_size * n_working,
                                                    global_offset, local_offset + local_imag_offset);
      }
    } else if (!interleaved_transforms_input) {
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        std::array<IdxGlobal, 3> global_strides{input_distance * 2, input_stride * 2, 1};
        std::array<Idx, 3> local_strides{fft_size * 2, 2, 1};
        std::array<Idx, 3> copy_indices{n_working, fft_size, 2};

        detail::md_view global_input_view{input, global_strides, global_input_offset};
        detail::md_view local_input_view{loc_view, local_strides, local_offset};

        global_data.log_message_global(__func__, "loading unpacked data from global to local memory");
        copy_group<level::SUBGROUP>(global_data, global_input_view, local_input_view, copy_indices);
      } else {
        std::array<IdxGlobal, 2> global_strides{input_distance, input_stride};
        std::array<Idx, 2> local_strides{fft_size, 1};
        std::array<Idx, 2> copy_indices{n_working, fft_size};

        detail::md_view global_input_real_view{input, global_strides, global_input_offset};
        detail::md_view local_input_real_view{loc_view, local_strides, local_offset};
        detail::md_view global_input_imag_view{input_imag, global_strides, global_input_offset};
        detail::md_view local_input_imag_view{loc_view, local_strides, local_offset + local_imag_offset};

        global_data.log_message_global(__func__, "loading unpacked real data from global to local memory");
        copy_group<level::SUBGROUP>(global_data, global_input_real_view, local_input_real_view, copy_indices);
        global_data.log_message_global(__func__, "loading unpacked imaginary data from global to local memory");
        copy_group<level::SUBGROUP>(global_data, global_input_imag_view, local_input_imag_view, copy_indices);
      }
    }
    if (is_packed_input || !interleaved_transforms_input) {
#ifdef PORTFFT_LOG_DUMPS
      sycl::group_barrier(global_data.sg);
#endif
      global_data.log_dump_local("input data loaded in local memory:", loc, n_reals * n_working);
    }

    sycl::group_barrier(global_data.sg);

    if (working) {
      if (interleaved_transforms_input) {
        global_data.log_message_global(__func__, "loading transposed data from global to private memory");
        // Load directly into registers from global memory so work-items read from nearby memory addresses.
        // No need of going through local memory either as it is an unnecessary extra write step.
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::strided_view input_view{input, input_stride, input_distance * i * 2};
          copy_wi<2>(global_data, input_view, priv, fft_size);
        } else {
          detail::strided_view input_real_view{input, input_stride, input_distance * i};
          detail::strided_view input_imag_view{input_imag, input_stride, input_distance * i};
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          copy_wi(global_data, input_real_view, priv_real_view, fft_size);
          copy_wi(global_data, input_imag_view, priv_imag_view, fft_size);
        }
      } else {
        global_data.log_message_global(__func__, "loading non-transposed data from local to private memory");
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::offset_view offset_local_view{loc_view, local_offset + subgroup_local_id * n_reals};
          copy_wi(global_data, offset_local_view, priv, n_reals);
        } else {
          detail::offset_view local_real_view{loc_view, local_offset + subgroup_local_id * fft_size};
          detail::offset_view local_imag_view{loc_view,
                                              local_offset + subgroup_local_id * fft_size + local_imag_offset};
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          copy_wi(global_data, local_real_view, priv_real_view, fft_size);
          copy_wi(global_data, local_imag_view, priv_imag_view, fft_size);
        }
      }
      if (conjugate_on_load == detail::complex_conjugate::APPLIED) {
        conjugate_inplace(priv, fft_size);
      }
      global_data.log_dump_private("data loaded in registers:", priv, n_reals);

      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        // Assumes load modifier data is stored in a transposed fashion (fft_size x  num_batches_local_mem)
        // to ensure much lesser bank conflicts
        global_data.log_message_global(__func__, "applying load modifier");
        detail::apply_modifier(fft_size, priv, load_modifier_data, i * n_reals);
      }
      wi_dft<0>(priv, priv, fft_size, 1, 1, wi_private_scratch);
      global_data.log_dump_private("data in registers after computation:", priv, n_reals);

      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        // Assumes store modifier data is stored in a transposed fashion (fft_size x  num_batches_local_mem)
        // to ensure much lesser bank conflicts
        global_data.log_message_global(__func__, "applying store modifier");
        detail::apply_modifier(fft_size, priv, store_modifier_data, i * n_reals);
      }
      if (conjugate_on_store == detail::complex_conjugate::APPLIED) {
        conjugate_inplace(priv, fft_size);
      }
      if (apply_scale_factor == detail::apply_scale_factor::APPLIED) {
        PORTFFT_UNROLL
        for (Idx idx = 0; idx < n_reals; idx += 2) {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        }
      }
      global_data.log_dump_private("data in registers after scaling:", priv, n_reals);

      if (interleaved_transforms_output) {
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::strided_view output_view{output, output_stride, output_distance * i * 2};
          copy_wi<2>(global_data, priv, output_view, fft_size);
        } else {
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          detail::strided_view output_real_view{output, output_stride, output_distance * i};
          detail::strided_view output_imag_view{output_imag, output_stride, output_distance * i};
          copy_wi(global_data, priv_real_view, output_real_view, fft_size);
          copy_wi(global_data, priv_imag_view, output_imag_view, fft_size);
        }
      } else {
        global_data.log_message_global(__func__, "loading data from private to local memory");
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::offset_view offset_local_view{loc_view, local_offset + subgroup_local_id * n_reals};
          copy_wi(global_data, priv, offset_local_view, n_reals);
        } else {
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          detail::offset_view local_real_view{loc_view, local_offset + subgroup_local_id * fft_size};
          detail::offset_view local_imag_view{loc_view,
                                              local_offset + subgroup_local_id * fft_size + local_imag_offset};
          copy_wi(global_data, priv_real_view, local_real_view, fft_size);
          copy_wi(global_data, priv_imag_view, local_imag_view, fft_size);
        }
      }
    }
    if (is_packed_output) {
      sycl::group_barrier(global_data.sg);
      global_data.log_dump_local("computed data local memory:", loc, n_reals * n_working);
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        global_data.log_message_global(__func__, "storing data from local to packed global memory");
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output, n_reals * n_working, local_offset,
                                                    global_offset);
      } else {
        global_data.log_message_global(__func__, "storing real data from local to packed global memory");
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output, fft_size * n_working, local_offset,
                                                    global_offset);
        global_data.log_message_global(__func__, "storing imaginary data from local to packed global memory");
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output_imag, fft_size * n_working,
                                                    local_offset + local_imag_offset, global_output_offset);
      }
    } else if (!interleaved_transforms_output) {
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        std::array<IdxGlobal, 3> global_strides{output_distance * 2, output_stride * 2, 1};
        std::array<Idx, 3> local_strides{fft_size * 2, 2, 1};
        std::array<Idx, 3> copy_indices{n_working, fft_size, 2};

        detail::md_view global_output_view{output, global_strides, global_output_offset};
        detail::md_view local_output_view{loc_view, local_strides, local_offset};
        global_data.log_message_global(__func__, "storing data from local to unpacked global memory");
        copy_group<level::SUBGROUP>(global_data, local_output_view, global_output_view, copy_indices);
      } else {
        std::array<IdxGlobal, 2> global_strides{output_distance, output_stride};
        std::array<Idx, 2> local_strides{fft_size, 1};
        std::array<Idx, 2> copy_indices{n_working, fft_size};

        detail::md_view global_output_real_view{output, global_strides, global_output_offset};
        detail::md_view local_output_real_view{loc_view, local_strides, local_offset};
        detail::md_view global_output_imag_view{output_imag, global_strides, global_output_offset};
        detail::md_view local_output_imag_view{loc_view, local_strides, local_offset + local_imag_offset};
        global_data.log_message_global(__func__, "storing real data from local to unpacked global memory");
        copy_group<level::SUBGROUP>(global_data, local_output_real_view, global_output_real_view, copy_indices);
        global_data.log_message_global(__func__, "storing imaginary data from local to unpacked global memory");
        copy_group<level::SUBGROUP>(global_data, local_output_imag_view, global_output_imag_view, copy_indices);
      }
    }
    if (is_packed_output || !interleaved_transforms_output) {
      sycl::group_barrier(global_data.sg);
    }
  }
  global_data.log_message_global(__func__, "exited");
}

template <typename Scalar, domain Domain>
template <Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::run_kernel_struct<SubgroupSize, TIn,
                                                                    TOut>::inner<detail::level::WORKITEM, Dummy> {
  static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                             TOut& out_imag, const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, dimension_struct& dimension_data,
                             direction compute_direction, layout input_layout) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    auto& kernel_data = compute_direction == direction::FORWARD ? dimension_data.forward_kernels.at(0)
                                                                : dimension_data.backward_kernels.at(0);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg,
            input_layout);
    std::size_t global_size = static_cast<std::size_t>(detail::get_global_size_workitem<Scalar>(
        n_transforms, SubgroupSize, kernel_data.num_sgs_per_wg, desc.n_compute_units));

    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(kernel_data.exec_bundle);
      auto in_acc_or_usm = detail::get_access(in, cgh);
      auto out_acc_or_usm = detail::get_access(out, cgh);
      auto in_imag_acc_or_usm = detail::get_access(in_imag, cgh);
      auto out_imag_acc_or_usm = detail::get_access(out_imag, cgh);
      sycl::local_accessor<Scalar, 1> loc(static_cast<std::size_t>(local_elements), cgh);
#ifdef PORTFFT_KERNEL_LOG
      sycl::stream s{1024 * 16 * 8, 1024, cgh};
#endif
      PORTFFT_LOG_TRACE("Launching workitem kernel with global_size", global_size, "local_size",
                        SubgroupSize * kernel_data.num_sgs_per_wg, "local memory allocation of size", local_elements);
      cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Mem, SubgroupSize>>(
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
            global_data.log_message_global("Running workitem kernel");
            detail::workitem_impl<SubgroupSize>(&in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                                                &in_imag_acc_or_usm[0] + input_offset,
                                                &out_imag_acc_or_usm[0] + output_offset, &loc[0], n_transforms,
                                                global_data, kh);
            global_data.log_message_global("Exiting workitem kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy> {
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
struct committed_descriptor_impl<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKITEM,
                                                                                         Dummy> {
  static std::size_t execute(committed_descriptor_impl& desc, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& /*factors*/, Idx& num_sgs_per_wg, layout /*input_layout*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    Idx num_scalars_per_sg = detail::pad_local(2 * static_cast<Idx>(length) * used_sg_size, 1);
    Idx max_n_sgs = desc.local_memory_size / static_cast<Idx>(sizeof(Scalar)) / num_scalars_per_sg;
    num_sgs_per_wg = std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_n_sgs));
    Idx res = num_scalars_per_sg * num_sgs_per_wg;
    return static_cast<std::size_t>(res);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKITEM, Dummy> {
  static Scalar* execute(committed_descriptor_impl& /*desc*/, dimension_struct& /*dimension_data*/,
                         std::vector<kernel_data_struct>& /*kernels*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return nullptr;
  }
};

}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
