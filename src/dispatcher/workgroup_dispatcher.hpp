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

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/helpers.hpp>
#include <common/logging.hpp>
#include <common/memory_views.hpp>
#include <common/transfers.hpp>
#include <common/workgroup.hpp>
#include <defines.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<Idx> WorkgroupSpecConstFftSize{};

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
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T, detail::layout LayoutIn>
IdxGlobal get_global_size_workgroup(IdxGlobal n_transforms, Idx subgroup_size, Idx n_compute_units) {
  Idx maximum_n_sgs = 8 * n_compute_units * 64;
  Idx maximum_n_wgs = maximum_n_sgs / PORTFFT_SGS_IN_WG;
  Idx wg_size = subgroup_size * PORTFFT_SGS_IN_WG;
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
 * @tparam MultiplyOnLoad Whether the input data is multiplied with some data array before fft computation.
 * @tparam MultiplyOnStore Whether the input data is multiplied with some data array after fft computation.
 * @tparam ApplyScaleFactor Whether or not the scale factor is applied
 * @tparam FFTSize Problem size
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 *
 * @param input global input pointer
 * @param output global output pointer
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to local allocation for subgroup level twiddles
 * @param n_transforms number of fft batches
 * @param global_data global data for the kernel
 * @param twiddles Pointer to twiddles in the global memory
 * @param scaling_factor scaling factor applied to the result
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, detail::elementwise_multiply MultiplyOnLoad,
          detail::elementwise_multiply MultiplyOnStore, detail::apply_scale_factor ApplyScaleFactor, Idx FFTSize,
          Idx SubgroupSize, typename T>
PORTFFT_INLINE void workgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles, IdxGlobal n_transforms,
                                   global_data_struct global_data, const T* twiddles, T scaling_factor,
                                   const T* load_modifier_data, const T* store_modifier_data) {
  global_data.log_message_global(__func__, "entered", "FFTSize", FFTSize, "n_transforms", n_transforms);
  Idx num_workgroups = static_cast<Idx>(global_data.it.get_group_range(0));
  Idx wg_id = static_cast<Idx>(global_data.it.get_group(0));
  IdxGlobal max_global_offset = 2 * (n_transforms - 1) * FFTSize;

  constexpr Idx N = detail::factorize(FFTSize);
  constexpr Idx M = FFTSize / N;
  const T* wg_twiddles = twiddles + 2 * (M + N);
  constexpr Idx BankLinesPerPad = bank_lines_per_pad_wg(2 * static_cast<Idx>(sizeof(T)) * M);
  auto loc_view = make_padded_view<BankLinesPerPad>(loc);

  global_data.log_message_global(__func__, "loading sg twiddles from global to local memory");
  global2local<level::WORKGROUP, SubgroupSize>(global_data, twiddles, loc_twiddles, 2 * (M + N));
  global_data.log_dump_local("twiddles loaded to local memory:", loc_twiddles, 2 * (M + N));

  Idx max_num_batches_in_local_mem =
      get_num_batches_in_local_mem_workgroup<LayoutIn>(static_cast<Idx>(global_data.it.get_local_range(0)));
  Idx max_reals_in_local_memory = 2 * FFTSize * max_num_batches_in_local_mem;
  IdxGlobal global_offset = static_cast<IdxGlobal>(wg_id) * static_cast<IdxGlobal>(max_reals_in_local_memory);
  IdxGlobal offset_increment =
      static_cast<IdxGlobal>(num_workgroups) * static_cast<IdxGlobal>(max_reals_in_local_memory);
  for (IdxGlobal offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
      /**
       * In the transposed case, the data is laid out in the local memory column-wise, veiwing it as a FFT_Size x
       * WG_SIZE / 2 matrix, Each column contains either the real or the complex component of the batch.  Loads WG_SIZE
       * / 2 consecutive batches into the local memory
       */
      const IdxGlobal batch_start_idx = offset / static_cast<IdxGlobal>(2 * FFTSize);
      const Idx num_batches_in_local_mem =
          std::min(max_num_batches_in_local_mem, static_cast<Idx>(n_transforms - batch_start_idx));
      global_data.log_message_global(__func__, "loading transposed data from global to local memory");
      global_batchinter_2_local_batchinter<level::WORKGROUP>(global_data, input, loc_view, offset / FFTSize,
                                                             2 * num_batches_in_local_mem, FFTSize, 2 * n_transforms,
                                                             2 * max_num_batches_in_local_mem);
      sycl::group_barrier(global_data.it.get_group());
      for (Idx sub_batch = 0; sub_batch < num_batches_in_local_mem; sub_batch++) {
        wg_dft<Dir, LayoutIn, MultiplyOnLoad, MultiplyOnStore, ApplyScaleFactor, FFTSize, N, M, SubgroupSize>(
            loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, sub_batch,
            offset / (2 * FFTSize), load_modifier_data, store_modifier_data, global_data);
        sycl::group_barrier(global_data.it.get_group());
      }
      if constexpr (LayoutOut == detail::layout::PACKED) {
        global_data.log_message_global(__func__, "storing data from local to global memory (with 2 transposes)");
        // local2global_transposed cannot be used over here. This is because the data in the local memory is also
        // stored in a strided fashion.
        local_batchinter_batchinter_2_global_packed<SubgroupSize>(
            global_data, loc_view, output, offset, 2 * max_num_batches_in_local_mem, N, M, num_batches_in_local_mem);
      } else {
        local_batchinter_batchinter_2_global_batchinter(global_data, output, loc_view, 2 * n_transforms,
                                                        2 * batch_start_idx, 2 * max_num_batches_in_local_mem,
                                                        num_batches_in_local_mem, N, M);
      }
      sycl::group_barrier(global_data.it.get_group());
    } else {
      global_data.log_message_global(__func__, "loading non-transposed data from global to local memory");
      global2local<level::WORKGROUP, SubgroupSize>(global_data, input, loc_view, 2 * FFTSize, offset);
      sycl::group_barrier(global_data.it.get_group());
      wg_dft<Dir, LayoutIn, MultiplyOnLoad, MultiplyOnStore, ApplyScaleFactor, FFTSize, N, M, SubgroupSize>(
          loc_view, loc_twiddles, wg_twiddles, scaling_factor, max_num_batches_in_local_mem, 0,
          offset / static_cast<IdxGlobal>(2 * FFTSize), load_modifier_data, store_modifier_data, global_data);
      sycl::group_barrier(global_data.it.get_group());
      global_data.log_message_global(__func__, "storing non-transposed data from local to global memory");
      // transposition for WG CT
      if constexpr (LayoutOut == detail::layout::PACKED) {
        local2global_transposed(global_data, N, M, M, loc_view, output, offset);
      } else {
        IdxGlobal current_batch = offset / static_cast<IdxGlobal>(2 * FFTSize);
        localstrided_2global_strided(global_data, output, loc_view, 2 * n_transforms, 2 * current_batch, FFTSize, N, M);
      }
      sycl::group_barrier(global_data.it.get_group());
    }
  }
  global_data.log_message_global(__func__, "exited");
}

/**
 * Launch specialized subgroup DFT size matching fft_size if one is available.
 *
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input Layout
 * @tparam LayoutOut Output Layout
 * @tparam MultiplyOnLoad Whether the input data is multiplied with some data array before fft computation.
 * @tparam MultiplyOnStore Whether the input data is multiplied with some data array after fft computation.
 * @tparam ApplyScaleFactor Whether or not the scale factor is applied
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 * @tparam SizeList The list of sizes that will be specialized.
 * @param input global input pointer
 * @param output global output pointer
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to twiddles residing in the local memory
 * @param n_transforms number of fft batches
 * @param global_data global data for the kernel
 * @param twiddles Pointer to twiddles residing in the global memory
 * @param scaling_factor scaling factor applied to the result
 * @tparam fft_size Problem size
 * @param load_modifier_data Pointer to the load modifier data in global Memory
 * @param store_modifier_data Pointer to the store modifier data in global Memory
 */
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, detail::elementwise_multiply MultiplyOnLoad,
          detail::elementwise_multiply MultiplyOnStore, detail::apply_scale_factor ApplyScaleFactor, Idx SubgroupSize,
          typename T, typename SizeList>
PORTFFT_INLINE void workgroup_dispatch_impl(const T* input, T* output, T* loc, T* loc_twiddles, IdxGlobal n_transforms,
                                            global_data_struct global_data, const T* twiddles, T scaling_factor,
                                            Idx fft_size, const T* load_modifier_data = nullptr,
                                            const T* store_modifier_data = nullptr) {
  if constexpr (!SizeList::ListEnd) {
    constexpr Idx ThisSize = SizeList::Size;
    if (fft_size == ThisSize) {
      if constexpr (!fits_in_sg<T>(ThisSize, SubgroupSize)) {
        workgroup_impl<Dir, LayoutIn, LayoutOut, MultiplyOnLoad, MultiplyOnStore, ApplyScaleFactor, ThisSize,
                       SubgroupSize>(input, output, loc, loc_twiddles, n_transforms, global_data, twiddles,
                                     scaling_factor, load_modifier_data, store_modifier_data);
      }
    } else {
      workgroup_dispatch_impl<Dir, LayoutIn, LayoutOut, MultiplyOnLoad, MultiplyOnStore, ApplyScaleFactor, SubgroupSize,
                              T, typename SizeList::child_t>(input, output, loc, loc_twiddles, n_transforms,
                                                             global_data, twiddles, scaling_factor, fft_size,
                                                             load_modifier_data, store_modifier_data);
    }
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
          typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::WORKGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies,
                             std::vector<kernel_data_struct>& kernel_data) {
    Idx num_batches_in_local_mem = [=]() {
      if constexpr (LayoutIn == detail::layout::BATCH_INTERLEAVED) {
        return kernel_data[0].used_sg_size * PORTFFT_SGS_IN_WG / 2;
      } else {
        return 1;
      }
    }();
    constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    Scalar* twiddles = kernel_data[0].twiddles_forward.get();
    IdxGlobal n_transforms = static_cast<IdxGlobal>(desc.params.number_of_transforms);
    std::size_t global_size = static_cast<std::size_t>(
        detail::get_global_size_workgroup<Scalar, LayoutIn>(n_transforms, SubgroupSize, desc.n_compute_units));
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP, LayoutIn, Dummy>::execute(
            desc, kernel_data[0].length, kernel_data[0].used_sg_size, kernel_data[0].factors,
            kernel_data[0].num_sgs_per_wg);
    const Idx bank_lines_per_pad = bank_lines_per_pad_wg(2 * static_cast<Idx>(sizeof(Scalar)) *
                                                         kernel_data[0].factors[2] * kernel_data[0].factors[3]);
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(kernel_data[0].exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
#ifdef PORTFFT_LOG
      sycl::stream s{1024 * 16, 1024, cgh};
#endif
      cgh.parallel_for<detail::workgroup_kernel<
          Scalar, Domain, Dir, Mem, LayoutIn, LayoutOut, detail::elementwise_multiply::NOT_APPLIED,
          detail::elementwise_multiply::NOT_APPLIED, detail::apply_scale_factor::APPLIED, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {static_cast<std::size_t>(SubgroupSize * PORTFFT_SGS_IN_WG)}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            Idx fft_size = kh.get_specialization_constant<detail::WorkgroupSpecConstFftSize>();
            detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
                s,
#endif
                it};
            global_data.log_message_global("Running workgroup kernel");
            detail::workgroup_dispatch_impl<Dir, LayoutIn, LayoutOut, detail::elementwise_multiply::NOT_APPLIED,
                                            detail::elementwise_multiply::NOT_APPLIED,
                                            detail::apply_scale_factor::APPLIED, SubgroupSize, Scalar,
                                            detail::cooley_tukey_size_list_t>(
                &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0],
                &loc[static_cast<std::size_t>(detail::pad_local<detail::pad::DO_PAD>(
                    2 * fft_size * num_batches_in_local_mem, bank_lines_per_pad))],
                n_transforms, global_data, twiddles, scale_factor, fft_size);
            global_data.log_message_global("Exiting workgroup kernel");
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKGROUP, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t length, const std::vector<Idx>& /*factors*/) {
    in_bundle.template set_specialization_constant<detail::WorkgroupSpecConstFftSize>(static_cast<Idx>(length));
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
  static Scalar* execute(committed_descriptor& desc, kernel_data_struct& kernel_data) {
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
