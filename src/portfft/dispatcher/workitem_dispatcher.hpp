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
  Idx maximum_n_sgs = 8 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  Idx wg_size = subgroup_size * num_sgs_per_wg;

  IdxGlobal n_wgs_we_can_utilize = divide_ceil(n_transforms, static_cast<IdxGlobal>(wg_size));
  return static_cast<IdxGlobal>(wg_size) * sycl::min(static_cast<IdxGlobal>(maximum_n_wgs), n_wgs_we_can_utilize);
}

/**
 * Utility function for applying load/store modifiers in workitem impl
 *
 * @tparam Dir Direction of the FFT
 * @tparam PrivT Private view type
 * @tparam T Type of pointer for load/store modifier global array
 * @param num_elements Num complex values per workitem
 * @param priv private memory array
 * @param modifier_data global modifier data pointer
 * @param offset offset for the global modifier data pointer
 */
template <direction Dir, typename PrivT, typename T>
PORTFFT_INLINE void apply_modifier(Idx num_elements, PrivT priv, const T* modifier_data, IdxGlobal offset) {
  PORTFFT_UNROLL
  for (Idx j = 0; j < num_elements; j++) {
    sycl::vec<T, 2> modifier_vec;
    modifier_vec.load(0, detail::get_global_multi_ptr(&modifier_data[offset + 2 * j]));
    if (Dir == direction::BACKWARD) {
      modifier_vec[1] *= -1;
    }
    multiply_complex(priv[2 * j], priv[2 * j + 1], modifier_vec[0], modifier_vec[1], priv[2 * j], priv[2 * j + 1]);
  }
}

/**
 * Implementation of FFT for sizes that can be done by independent work items.
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
 * @param loc local memory pointer. Must have enough space for 2*fft_size*SubgroupSize
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 * @param load_modifier_data Pointer to the load modifier data in global memory
 * @param store_modifier_data Pointer to the store modifier data in global memory
 * @param loc_load_modifier Pointer to load modifier data in local memory
 * @param loc_store_modifier Pointer to store modifier data in local memory
 */
template <direction Dir, Idx SubgroupSize, detail::layout LayoutIn, detail::layout LayoutOut, typename T>
PORTFFT_INLINE void workitem_impl(const T* input, T* output, const T* input_imag, T* output_imag, T* loc,
                                  IdxGlobal n_transforms, T scaling_factor, global_data_struct<1> global_data,
                                  sycl::kernel_handler& kh, const T* load_modifier_data = nullptr,
                                  const T* store_modifier_data = nullptr, T* loc_load_modifier = nullptr,
                                  T* loc_store_modifier = nullptr) {
  complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  detail::elementwise_multiply multiply_on_load = kh.get_specialization_constant<detail::SpecConstMultiplyOnLoad>();
  detail::elementwise_multiply multiply_on_store = kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  detail::apply_scale_factor apply_scale_factor = kh.get_specialization_constant<detail::SpecConstApplyScaleFactor>();
  const Idx fft_size = kh.get_specialization_constant<detail::SpecConstFftSize>();

  global_data.log_message_global(__func__, "entered", "fft_size", fft_size, "n_transforms", n_transforms);

  bool interleaved_storage = storage == complex_storage::INTERLEAVED_COMPLEX;
  const Idx n_reals = 2 * fft_size;
  const Idx n_io_reals = interleaved_storage ? n_reals : fft_size;

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
  IdxGlobal global_id = static_cast<IdxGlobal>(global_data.it.get_global_id(0));
  IdxGlobal global_size = static_cast<IdxGlobal>(global_data.it.get_global_range(0));
  Idx subgroup_id = static_cast<Idx>(global_data.sg.get_group_id());
  Idx local_offset = n_reals * SubgroupSize * subgroup_id;
  Idx local_imag_offset = fft_size * SubgroupSize;
  constexpr Idx BankLinesPerPad = 1;
  auto loc_view = detail::padded_view(loc, BankLinesPerPad);
  auto loc_load_modifier_view = detail::padded_view(loc_load_modifier, BankLinesPerPad);
  auto loc_store_modifier_view = detail::padded_view(loc_store_modifier, BankLinesPerPad);

  for (IdxGlobal i = global_id; i < round_up_to_multiple(n_transforms, static_cast<IdxGlobal>(SubgroupSize));
       i += global_size) {
    bool working = i < n_transforms;
    Idx n_working = sycl::min(SubgroupSize, static_cast<Idx>(n_transforms - i) + subgroup_local_id);

    IdxGlobal global_offset = static_cast<IdxGlobal>(n_io_reals) * (i - static_cast<IdxGlobal>(subgroup_local_id));
    if (LayoutIn == detail::layout::PACKED) {
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input, loc_view, n_reals * n_working, global_offset,
                                                    local_offset);
      } else {
        global_data.log_message_global(__func__, "loading non-transposed real data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input, loc_view, fft_size * n_working, global_offset,
                                                    local_offset);
        global_data.log_message_global(__func__, "loading non-transposed imaginary data from global to local memory");
        global2local<level::SUBGROUP, SubgroupSize>(global_data, input_imag, loc_view, fft_size * n_working,
                                                    global_offset, local_offset + local_imag_offset);
      }
#ifdef PORTFFT_LOG_DUMPS
      sycl::group_barrier(global_data.sg);
#endif
      global_data.log_dump_local("input data loaded in local memory:", loc, n_reals * n_working);
    }

    sycl::group_barrier(global_data.sg);

    if (working) {
      if (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        global_data.log_message_global(__func__, "loading transposed data from global to private memory");
        // Load directly into registers from global memory as all loads will be fully coalesced.
        // No need of going through local memory either as it is an unnecessary extra write step.
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::strided_view input_view{input, n_transforms, i * 2};
          copy_wi<2>(global_data, input_view, priv, fft_size);
        } else {
          detail::strided_view input_real_view{input, n_transforms, i};
          detail::strided_view input_imag_view{input_imag, n_transforms, i};
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
      global_data.log_dump_private("data loaded in registers:", priv, n_reals);
      if (multiply_on_load == detail::elementwise_multiply::APPLIED) {
        // Assumes load modifier data is stored in a transposed fashion (fft_size x  num_batches_local_mem)
        // to ensure much lesser bank conflicts
        global_data.log_message_global(__func__, "applying load modifier");
        detail::apply_modifier<Dir>(fft_size, priv, load_modifier_data, i * n_reals);
      }
      wi_dft<Dir, 0>(priv, priv, fft_size, 1, 1, wi_private_scratch);
      global_data.log_dump_private("data in registers after computation:", priv, n_reals);
      if (multiply_on_store == detail::elementwise_multiply::APPLIED) {
        // Assumes store modifier data is stored in a transposed fashion (fft_size x  num_batches_local_mem)
        // to ensure much lesser bank conflicts
        global_data.log_message_global(__func__, "applying store modifier");
        detail::apply_modifier<Dir>(fft_size, priv, store_modifier_data, i * n_reals);
      }
      if (apply_scale_factor == detail::apply_scale_factor::APPLIED) {
        PORTFFT_UNROLL
        for (Idx idx = 0; idx < n_reals; idx += 2) {
          priv[idx] *= scaling_factor;
          priv[idx + 1] *= scaling_factor;
        }
      }
      global_data.log_dump_private("data in registers after scaling:", priv, n_reals);
      global_data.log_message_global(__func__, "loading data from private to local memory");
      if (LayoutOut == detail::layout::PACKED) {
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
      } else {
        if (storage == complex_storage::INTERLEAVED_COMPLEX) {
          detail::strided_view output_view{output, n_transforms, i * 2};
          copy_wi<2>(global_data, priv, output_view, fft_size);
        } else {
          detail::strided_view priv_real_view{priv, 2};
          detail::strided_view priv_imag_view{priv, 2, 1};
          detail::strided_view output_real_view{output, n_transforms, i};
          detail::strided_view output_imag_view{output_imag, n_transforms, i};
          copy_wi(global_data, priv_real_view, output_real_view, fft_size);
          copy_wi(global_data, priv_imag_view, output_imag_view, fft_size);
        }
      }
    }
    if (LayoutOut == detail::layout::PACKED) {
      sycl::group_barrier(global_data.sg);
      global_data.log_dump_local("computed data local memory:", loc, n_reals * n_working);
      global_data.log_message_global(__func__, "storing data from local to global memory");
      if (storage == complex_storage::INTERLEAVED_COMPLEX) {
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output, n_reals * n_working, local_offset,
                                                    global_offset);
      } else {
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output, fft_size * n_working, local_offset,
                                                    global_offset);
        local2global<level::SUBGROUP, SubgroupSize>(global_data, loc_view, output_imag, fft_size * n_working,
                                                    local_offset + local_imag_offset, global_offset);
      }
      sycl::group_barrier(global_data.sg);
    }
  }
  global_data.log_message_global(__func__, "exited");
}
}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
          typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::WORKITEM, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                             const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, Scalar scale_factor,
                             dimension_struct& dimension_data) {
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    auto& kernel_data = dimension_data.kernels.at(0);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, LayoutIn, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg);
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
#ifdef PORTFFT_LOG
      sycl::stream s{1024 * 16 * 8, 1024, cgh};
#endif
      cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, Mem, LayoutIn, LayoutOut, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * kernel_data.num_sgs_per_wg)}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
                s,
#endif
                it};
            global_data.log_message_global("Running workitem kernel");
            detail::workitem_impl<Dir, SubgroupSize, LayoutIn, LayoutOut>(
                &in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                &in_imag_acc_or_usm[0] + input_offset, &out_imag_acc_or_usm[0] + output_offset, &loc[0], n_transforms,
                scale_factor, global_data, kh);
            global_data.log_message_global("Exiting workitem kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t length, const std::vector<Idx>& /*factors*/, detail::level /*level*/,
                      Idx /*factor_num*/, Idx /*num_factors*/) {
    const Idx length_idx = static_cast<Idx>(length);
    in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length_idx);
  }
};

template <typename Scalar, domain Domain>
template <detail::layout LayoutIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKITEM, LayoutIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& /*factors*/, Idx& num_sgs_per_wg) {
    Idx num_scalars_per_sg = detail::pad_local(2 * static_cast<Idx>(length) * used_sg_size, 1);
    Idx max_n_sgs = desc.local_memory_size / static_cast<Idx>(sizeof(Scalar)) / num_scalars_per_sg;
    num_sgs_per_wg = std::min(Idx(PORTFFT_SGS_IN_WG), std::max(Idx(1), max_n_sgs));
    Idx res = num_scalars_per_sg * num_sgs_per_wg;
    return static_cast<std::size_t>(res);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKITEM, Dummy> {
  static Scalar* execute(committed_descriptor& /*desc*/, dimension_struct& /*dimension_data*/) { return nullptr; }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
