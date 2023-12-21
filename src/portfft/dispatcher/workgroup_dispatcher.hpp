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
 * @tparam LayoutIn The input data layout
 * @param workgroup_size The size of the work-group. Must be divisible by 2.
 */
template <detail::layout LayoutIn>
PORTFFT_INLINE constexpr Idx get_num_batches_in_local_mem_workgroup(Idx workgroup_size) noexcept {
  if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
    return workgroup_size / 2;
  } else {
    return 1;
  }
}

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @tparam LayoutIn The input data layout
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param num_sgs_per_wg number of subgroups in a workgroup
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T, detail::layout LayoutIn>
IdxGlobal get_global_size_workgroup(IdxGlobal n_transforms, Idx subgroup_size, Idx num_sgs_per_wg,
                                    Idx n_compute_units) {
  Idx maximum_n_sgs = 8 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  Idx wg_size = subgroup_size * num_sgs_per_wg;
  Idx dfts_per_wg = get_num_batches_in_local_mem_workgroup<LayoutIn>(wg_size);

  return static_cast<IdxGlobal>(wg_size) * sycl::min(static_cast<IdxGlobal>(maximum_n_wgs),
                                                     divide_ceil(n_transforms, static_cast<IdxGlobal>(dfts_per_wg)));
}

/**
 * Implementation of FFT for sizes that can be done by a workgroup.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input Layout
 * @tparam LayoutOut Output Layout
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 *
 * @param input accessor or pointer to global memory containing input data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param output accessor or pointer to global memory for output data. If complex storage (from
 * `SpecConstComplexStorage`) is split, this is just the real part of data.
 * @param input accessor or pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param output accessor or pointer to global memory containing imaginary part of the input data if complex storage
 * (from `SpecConstComplexStorage`) is split. Otherwise unused.
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to local allocation for subgroup level twiddles
 * @param n_transforms number of fft batches
 * @param global_data global data for the kernel
 * @param kh kernel handler associated with the kernel launch
 * @param twiddles Pointer to twiddles in the global memory
 * @param scaling_factor scaling factor applied to the result
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <direction Dir, Idx SubgroupSize, detail::layout LayoutIn, detail::layout LayoutOut, typename T>
PORTFFT_INLINE void workgroup_impl(const T* input, T* output, const T* /*input_imag*/, T* /*output_imag*/, T* loc,
                                   T* loc_twiddles, IdxGlobal n_transforms, const T* twiddles, T scaling_factor,
                                   global_data_struct<1> global_data, sycl::kernel_handler& kh,
                                   const T* load_modifier_data = nullptr, const T* store_modifier_data = nullptr) {
  // complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  detail::elementwise_multiply multiply_on_load = kh.get_specialization_constant<detail::SpecConstMultiplyOnLoad>();
  detail::elementwise_multiply multiply_on_store = kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  detail::apply_scale_factor apply_scale_factor = kh.get_specialization_constant<detail::SpecConstApplyScaleFactor>();

  const Idx fft_size = kh.get_specialization_constant<detail::SpecConstFftSize>();

  global_data.log_message_global(__func__, "entered", "fft_size", fft_size, "n_transforms", n_transforms);
  Idx num_workgroups = static_cast<Idx>(global_data.it.get_group_range(0));
  Idx wg_id = static_cast<Idx>(global_data.it.get_group(0));
  IdxGlobal max_global_offset = 2 * (n_transforms - 1) * fft_size;

  Idx factor_n = detail::factorize(fft_size);
  Idx factor_m = fft_size / factor_n;
  const T* wg_twiddles = twiddles + 2 * (factor_m + factor_n);
  const Idx bank_lines_per_pad = bank_lines_per_pad_wg(2 * static_cast<Idx>(sizeof(T)) * factor_m);
  auto loc_view = padded_view(loc, bank_lines_per_pad);

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles, 2 * (factor_m + factor_n));
  global_data.log_dump_local("twiddles loaded to local memory:", loc_twiddles, 2 * (factor_m + factor_n));

  Idx max_num_batches_in_local_mem =
      get_num_batches_in_local_mem_workgroup<LayoutIn>(static_cast<Idx>(global_data.it.get_local_range(0)));
  Idx max_reals_in_local_memory = 2 * fft_size * max_num_batches_in_local_mem;
  IdxGlobal global_offset = static_cast<IdxGlobal>(wg_id) * static_cast<IdxGlobal>(max_reals_in_local_memory);
  IdxGlobal offset_increment =
      static_cast<IdxGlobal>(num_workgroups) * static_cast<IdxGlobal>(max_reals_in_local_memory);
  for (IdxGlobal offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    if (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      /**
       * In the transposed case, the data is laid out in the local memory column-wise, veiwing it as a FFT_Size x
       * WG_SIZE / 2 matrix, Each column contains either the real or the complex component of the batch.  Loads WG_SIZE
       * / 2 consecutive batches into the local memory
       */
      const IdxGlobal batch_start_idx = offset / static_cast<IdxGlobal>(2 * fft_size);
      const Idx num_batches_in_local_mem =
          std::min(max_num_batches_in_local_mem, static_cast<Idx>(n_transforms - batch_start_idx));
      global_data.log_message_global(__func__, "loading transposed data from global to local memory");
      detail::md_view input_view{input, std::array{2 * n_transforms, static_cast<IdxGlobal>(1)}, offset / fft_size};
      detail::md_view loc_md_view{loc_view, std::array{2 * max_num_batches_in_local_mem, 1}};
      copy_group<level::WORKGROUP>(global_data, input_view, loc_md_view,
                                   std::array{fft_size, 2 * num_batches_in_local_mem});
      sycl::group_barrier(global_data.it.get_group());
      for (Idx sub_batch = 0; sub_batch < num_batches_in_local_mem; sub_batch++) {
        wg_dft<Dir, SubgroupSize>(loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem,
                                  sub_batch, offset / (2 * fft_size), load_modifier_data, store_modifier_data, fft_size,
                                  factor_n, factor_m, LayoutIn, multiply_on_load, multiply_on_store, apply_scale_factor,
                                  global_data);
        sycl::group_barrier(global_data.it.get_group());
      }
      if constexpr (LayoutOut == detail::layout::PACKED) {
        global_data.log_message_global(__func__, "storing data from local to global memory (with 2 transposes)");
        detail::md_view loc_md_view2{
            loc_view, std::array{2, 1, 2 * max_num_batches_in_local_mem, 2 * max_num_batches_in_local_mem * factor_m}};
        detail::md_view output_view{output, std::array{2 * fft_size, 1, 2 * factor_n, 2}, offset};
        copy_group<level::WORKGROUP>(global_data, loc_md_view2, output_view,
                                     std::array{num_batches_in_local_mem, 2, factor_m, factor_n});
      } else {
        detail::md_view loc_md_view2{
            loc_view, std::array{2 * max_num_batches_in_local_mem, 2 * max_num_batches_in_local_mem * factor_m, 1}};
        detail::md_view output_view{
            output, std::array{2 * n_transforms * factor_n, 2 * n_transforms, static_cast<IdxGlobal>(1)},
            2 * batch_start_idx};
        copy_group<level::WORKGROUP>(global_data, loc_md_view2, output_view,
                                     std::array{factor_m, factor_n, 2 * num_batches_in_local_mem});
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {
      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
      global2local<level::WORKGROUP, SubgroupSize>(global_data, input, loc_view, 2 * fft_size, offset);
      sycl::group_barrier(global_data.it.get_group());
      wg_dft<Dir, SubgroupSize>(loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, 0,
                                offset / static_cast<IdxGlobal>(2 * fft_size), load_modifier_data, store_modifier_data,
                                fft_size, factor_n, factor_m, LayoutIn, multiply_on_load, multiply_on_store,
                                apply_scale_factor, global_data);
      sycl::group_barrier(global_data.it.get_group());
      global_data.log_message_global(__func__, "storing non-transposed data from local to global memory");
      // transposition for WG CT
      if (LayoutOut == detail::layout::PACKED) {
        detail::md_view local_md_view2{loc_view, std::array{1, 2, 2 * factor_m}};
        detail::md_view output_view{output, std::array{1, 2 * factor_n, 2}, offset};
        copy_group<level::WORKGROUP>(global_data, local_md_view2, output_view, std::array{2, factor_m, factor_n});
      } else {
        IdxGlobal current_batch = offset / static_cast<IdxGlobal>(2 * fft_size);
        detail::md_view local_md_view2{loc_view, std::array{2, 1, 2 * factor_m}};
        detail::md_view output_view{
            output, std::array{2 * factor_n * n_transforms, static_cast<IdxGlobal>(1), 2 * n_transforms},
            2 * current_batch};
        copy_group<level::WORKGROUP>(global_data, local_md_view2, output_view, std::array{factor_m, 2, factor_n});
      }
      sycl::group_barrier(global_data.it.get_group());
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
                                                               TOut>::inner<detail::level::WORKGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                             const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, Scalar scale_factor,
                             dimension_struct& dimension_data) {
    auto& kernel_data = dimension_data.kernels.at(0);
    Idx num_batches_in_local_mem = [=]() {
      if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        return kernel_data.used_sg_size * PORTFFT_SGS_IN_WG / 2;
      } else {
        return 1;
      }
    }();
    constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
    Scalar* twiddles = kernel_data.twiddles_forward.get();
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP, LayoutIn, Dummy>::execute(
            desc, kernel_data.length, kernel_data.used_sg_size, kernel_data.factors, kernel_data.num_sgs_per_wg);
    std::size_t global_size = static_cast<std::size_t>(detail::get_global_size_workgroup<Scalar, LayoutIn>(
        n_transforms, SubgroupSize, kernel_data.num_sgs_per_wg, desc.n_compute_units));
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
#ifdef PORTFFT_LOG
      sycl::stream s{1024 * 16, 1024, cgh};
#endif
      cgh.parallel_for<detail::workgroup_kernel<Scalar, Domain, Dir, Mem, LayoutIn, LayoutOut, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * PORTFFT_SGS_IN_WG)}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
                s,
#endif
                it};
            global_data.log_message_global("Running workgroup kernel");
            detail::workgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut>(
                &in_acc_or_usm[0] + input_offset, &out_acc_or_usm[0] + output_offset,
                &in_imag_acc_or_usm[0] + input_offset, &out_imag_acc_or_usm[0] + output_offset, &loc[0],
                &loc[0] + sg_twiddles_offset, n_transforms, twiddles, scale_factor, global_data, kh);
            global_data.log_message_global("Exiting workgroup kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKGROUP, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t length, const std::vector<Idx>& /*factors*/, detail::level /*level*/,
                      Idx /*factor_num*/, Idx /*num_factors*/) {
    const Idx length_idx = static_cast<Idx>(length);
    in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length_idx);
  }
};

template <typename Scalar, domain Domain>
template <typename detail::layout LayoutIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKGROUP, LayoutIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& /*desc*/, std::size_t length, Idx used_sg_size,
                             const std::vector<Idx>& factors, Idx& /*num_sgs_per_wg*/) {
    std::size_t n = static_cast<std::size_t>(factors[0]) * static_cast<std::size_t>(factors[1]);
    std::size_t m = static_cast<std::size_t>(factors[2]) * static_cast<std::size_t>(factors[3]);
    // working memory + twiddles for subgroup impl for the two sizes
    Idx num_batches_in_local_mem =
        detail::get_num_batches_in_local_mem_workgroup<LayoutIn>(used_sg_size * PORTFFT_SGS_IN_WG);
    return detail::pad_local(static_cast<std::size_t>(2 * num_batches_in_local_mem) * length,
                             bank_lines_per_pad_wg(2 * static_cast<std::size_t>(sizeof(Scalar)) * m)) +
           2 * (m + n);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKGROUP, Dummy> {
  static Scalar* execute(committed_descriptor& desc, dimension_struct& dimension_data) {
    const auto& kernel_data = dimension_data.kernels.at(0);
    Idx factor_wi_n = kernel_data.factors[0];
    Idx factor_sg_n = kernel_data.factors[1];
    Idx factor_wi_m = kernel_data.factors[2];
    Idx factor_sg_m = kernel_data.factors[3];
    Idx fft_size = static_cast<Idx>(kernel_data.length);
    Idx n = factor_wi_n * factor_sg_n;
    Idx m = factor_wi_m * factor_sg_m;
    Idx res_size = 2 * (m + n + fft_size);
    Scalar* res =
        sycl::aligned_alloc_device<Scalar>(alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>),
                                           static_cast<std::size_t>(res_size), desc.queue);
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_n), static_cast<std::size_t>(factor_wi_n)}),
                       [=](sycl::item<2> it) {
                         Idx n = static_cast<Idx>(it.get_id(0));
                         Idx k = static_cast<Idx>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_n, factor_wi_n, n, k, res + (2 * m));
                       });
    });
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_m), static_cast<std::size_t>(factor_wi_m)}),
                       [=](sycl::item<2> it) {
                         Idx n = static_cast<Idx>(it.get_id(0));
                         Idx k = static_cast<Idx>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_m, factor_wi_m, n, k, res);
                       });
    });
    desc.queue.submit([&](sycl::handler& cgh) {
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

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP
