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
#include <common/transfers.hpp>
#include <common/workgroup.hpp>
#include <defines.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> WorkgroupSpecConstFftSize{};

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param n_compute_units number of compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t get_global_size_workgroup(std::size_t n_transforms, std::size_t subgroup_size,
                                      std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / PORTFFT_SGS_IN_WG;
  std::size_t wg_size = subgroup_size * PORTFFT_SGS_IN_WG;

  return wg_size * sycl::min(maximum_n_wgs, n_transforms);
}

/**
 * Implementation of FFT for sizes that can be done by a workgroup.
 *
 * @tparam Dir Direction of the FFT
 * @tparam TransposeIn Whether or not the input is transposed
 * @tparam FFTSize Problem size
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 *
 * @param input global input pointer
 * @param output global output pointer
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to local allocation for subgroup level twiddles
 * @param n_transforms number of fft batches
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles Pointer to twiddles in the global memory
 * @param scaling_factor scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, bool ApplyLoadModifier,
          bool ApplyStoreModifier, bool ApplyScaleFactor, std::size_t FFTSize, int SubgroupSize, typename T>
PORTFFT_ALWAYS_INLINE void workgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles, std::size_t n_transforms,
                                          sycl::nd_item<1> it, const T* twiddles, T scaling_factor,
                                          const T* load_modifier_data, const T* store_modifier_data) {
  std::size_t num_workgroups = it.get_group_range(0);
  std::size_t wg_id = it.get_group(0);
  std::size_t max_global_offset = 2 * (n_transforms - 1) * FFTSize;
  std::size_t global_offset = 2 * FFTSize * wg_id;
  constexpr std::size_t N = detail::factorize(FFTSize);
  constexpr std::size_t M = FFTSize / N;
  const T* wg_twiddles = twiddles + 2 * (M + N);
  constexpr std::size_t BankLinesPerPad = bank_lines_per_pad_wg(2 * sizeof(T) * M);

  std::size_t max_num_batches_in_local_mem = [=]() {
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      return it.get_local_range(0) / 2;
    } else {
      return 1;
    }
  }();
  std::size_t offset_increment = 2 * FFTSize * num_workgroups * max_num_batches_in_local_mem;
  global2local<level::WORKGROUP, SubgroupSize, pad::DONT_PAD, 0>(it, twiddles, loc_twiddles, 2 * (M + N));
  for (std::size_t offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      /**
       * In the transposed case, the data is laid out in the local memory column-wise, veiwing it as a FFT_Size x
       * WG_SIZE / 2 matrix, Each column contains either the real or the complex component of the batch.  Loads WG_SIZE
       * / 2 consecutive batches into the local memory
       */
      const std::size_t num_batches_in_local_mem = [=]() {
        if ((offset / (2 * FFTSize)) + it.get_local_range(0) / 2 < n_transforms) {
          return it.get_local_range(0) / 2;
        }
        return n_transforms - offset / (2 * FFTSize);
      }();
      // Load in a transposed manner, similar to subgroup impl.
      if (it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        // transposition requested by the caller
        global2local_transposed<level::WORKGROUP, pad::DO_PAD, BankLinesPerPad>(
            it, input, loc, offset / FFTSize, FFTSize, n_transforms, max_num_batches_in_local_mem);
      }
      sycl::group_barrier(it.get_group());
      for (std::size_t sub_batch = 0; sub_batch < num_batches_in_local_mem; sub_batch++) {
        wg_dft<Dir, TransposeIn, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor, FFTSize, N, M, SubgroupSize,
               BankLinesPerPad>(loc, loc_twiddles, wg_twiddles, it, scaling_factor, max_num_batches_in_local_mem,
                                sub_batch, load_modifier_data, store_modifier_data);
        sycl::group_barrier(it.get_group());
      }
      if (it.get_local_linear_id() / 2 < num_batches_in_local_mem) {
        // local2global_transposed cannot be used over here. This is because the data in the local memory is also stored
        // in a strided fashion.
        local_strided_2_global_strided_transposed<detail::pad::DO_PAD>(
            loc, output, offset, 2 * max_num_batches_in_local_mem, N, M, FFTSize, BankLinesPerPad, it);
      }
      sycl::group_barrier(it.get_group());
    } else {
      global2local<level::WORKGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(it, input, loc, 2 * FFTSize, offset);
      sycl::group_barrier(it.get_group());
      wg_dft<Dir, TransposeIn, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor, FFTSize, N, M, SubgroupSize,
             BankLinesPerPad>(loc, loc_twiddles, wg_twiddles, it, scaling_factor, max_num_batches_in_local_mem, 0,
                              load_modifier_data, store_modifier_data);
      sycl::group_barrier(it.get_group());
      // transposition for WG CT
      local2global_transposed<detail::pad::DO_PAD, BankLinesPerPad>(it, N, M, M, loc, output, offset);
      sycl::group_barrier(it.get_group());
    }
  }
}

/**
 * Launch specialized subgroup DFT size matching fft_size if one is available.
 *
 * @tparam Dir Direction of the FFT
 * @tparam SubgroupSize size of the subgroup
 * @tparam T Scalar type
 * @tparam SizeList The list of sizes that will be specialized.
 * @param input global input pointer
 * @param output global output pointer
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to twiddles residing in the local memory
 * @param n_transforms number of fft batches
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles Pointer to twiddles residing in the global memory
 * @param scaling_factor scaling factor applied to the result
 * @tparam fft_size Problem size
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, bool ApplyLoadModifier,
          bool ApplyStoreModifier, bool ApplyScaleFactor, int SubgroupSize, typename T, typename SizeList>
PORTFFT_INLINE void workgroup_dispatch_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                            std::size_t n_transforms, sycl::nd_item<1> it, const T* twiddles,
                                            T scaling_factor, std::size_t fft_size,
                                            const T* load_modifier_data = nullptr,
                                            const T* store_modifier_data = nullptr) {
  if constexpr (!SizeList::ListEnd) {
    constexpr size_t ThisSize = SizeList::Size;
    if (fft_size == ThisSize) {
      if constexpr (!fits_in_sg<T>(ThisSize, SubgroupSize)) {
        workgroup_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor,
                       ThisSize, SubgroupSize>(input, output, loc, loc_twiddles, n_transforms, it, twiddles,
                                               scaling_factor, load_modifier_data, store_modifier_data);
      }
    } else {
      workgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor,
                              SubgroupSize, T, typename SizeList::child_t>(
          input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor, fft_size, load_modifier_data,
          store_modifier_data);
    }
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::WORKGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    std::size_t num_batches_in_local_mem = [=]() {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        return static_cast<std::size_t>(desc.used_sg_size * PORTFFT_SGS_IN_WG / 2);
      } else {
        return static_cast<std::size_t>(1);
      }
    }();
    constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t n_transforms = desc.params.number_of_transforms;
    Scalar* twiddles = desc.twiddles_forward.get();
    std::size_t global_size =
        detail::get_global_size_workgroup<Scalar>(n_transforms, SubgroupSize, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP, TransposeIn, Dummy>::execute(desc);
    const std::size_t bank_lines_per_pad =
        bank_lines_per_pad_wg(2 * sizeof(Scalar) * static_cast<std::size_t>(desc.factors[2] * desc.factors[3]));
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle[0]);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      cgh.parallel_for<detail::workgroup_kernel<Scalar, Domain, Dir, Mem, TransposeIn,
                                                detail::transpose::NOT_TRANSPOSED, false, false, true, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * PORTFFT_SGS_IN_WG}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::WorkgroupSpecConstFftSize>();
            detail::workgroup_dispatch_impl<Dir, TransposeIn, detail::transpose::NOT_TRANSPOSED, false, false, true,
                                            SubgroupSize, Scalar, detail::cooley_tukey_size_list_t>(
                &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0],
                &loc[detail::pad_local<detail::pad::DO_PAD>(2 * fft_size * num_batches_in_local_mem,
                                                            bank_lines_per_pad)],
                n_transforms, it, twiddles, scale_factor, fft_size);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKGROUP, Dummy> {
  static void execute(committed_descriptor& desc,
                      std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>& in_bundles) {
    for (auto& in_bundle : in_bundles) {
      in_bundle.template set_specialization_constant<detail::WorkgroupSpecConstFftSize>(desc.params.lengths[0]);
    }
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::WORKGROUP,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    std::size_t n = static_cast<std::size_t>(desc.factors[0] * desc.factors[1]);
    std::size_t m = static_cast<std::size_t>(desc.factors[2] * desc.factors[3]);
    // working memory + twiddles for subgroup impl for the two sizes
    if (TransposeIn == detail::transpose::TRANSPOSED) {
      std::size_t num_batches_in_local_mem = static_cast<std::size_t>(desc.used_sg_size) * PORTFFT_SGS_IN_WG / 2;
      return detail::pad_local(2 * fft_size * num_batches_in_local_mem) + 2 * (m + n);
    }
    return detail::pad_local(2 * fft_size) + 2 * (m + n);
  }
};

template <typename Scalar, domain Domain>
template <typename detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKGROUP,
                                                                                    TransposeIn, Dummy, std::size_t> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::WORKGROUP, TransposeIn, Dummy>::execute(
        desc, fft_size);
  }
};

template <typename Scalar, domain Domain>
template <typename detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKGROUP,
                                                                                    TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::WORKGROUP, TransposeIn, Dummy>::execute(
        desc, desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKGROUP, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    int factor_wi_n = static_cast<int>(desc.factors[0]);
    int factor_sg_n = static_cast<int>(desc.factors[1]);
    int factor_wi_m = static_cast<int>(desc.factors[2]);
    int factor_sg_m = static_cast<int>(desc.factors[3]);
    std::size_t fft_size = desc.params.lengths[0];
    std::size_t n = static_cast<std::size_t>(factor_wi_n) * static_cast<std::size_t>(factor_sg_n);
    std::size_t m = static_cast<std::size_t>(factor_wi_m) * static_cast<std::size_t>(factor_sg_m);
    Scalar* res = sycl::aligned_alloc_device<Scalar>(
        alignof(sycl::vec<Scalar, PORTFFT_VEC_LOAD_BYTES / sizeof(Scalar)>), 2 * (m + n + fft_size), desc.queue);
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_n), static_cast<std::size_t>(factor_wi_n)}),
                       [=](sycl::item<2> it) {
                         int n = static_cast<int>(it.get_id(0));
                         int k = static_cast<int>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_n, factor_wi_n, n, k, res + (2 * m));
                       });
    });
    desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_m), static_cast<std::size_t>(factor_wi_m)}),
                       [=](sycl::item<2> it) {
                         int n = static_cast<int>(it.get_id(0));
                         int k = static_cast<int>(it.get_id(1));
                         sg_calc_twiddles(factor_sg_m, factor_wi_m, n, k, res);
                       });
    });
    Scalar* global_pointer = res + 2 * (n + m);
    // Copying from pinned memory to device might be faster than from regular allocation
    Scalar* temp_host = sycl::malloc_host<Scalar>(2 * fft_size, desc.queue);

    for (std::size_t i = 0; i < n; i++) {
      for (std::size_t j_wi = 0; j_wi < static_cast<std::size_t>(factor_wi_m); j_wi++) {
        for (std::size_t j_sg = 0; j_sg < static_cast<std::size_t>(factor_sg_m); j_sg++) {
          std::size_t j = j_wi + j_sg * static_cast<std::size_t>(factor_wi_m);
          std::size_t j_loc = j_wi * static_cast<std::size_t>(factor_sg_m) + j_sg;
          std::size_t index = 2 * (i * m + j_loc);
          auto tw = detail::calculate_twiddle<Scalar>(i * j, fft_size);
          temp_host[index] = tw.real();
          temp_host[index + 1] = tw.imag();
        }
      }
    }
    desc.queue.copy(temp_host, global_pointer, 2 * fft_size);
    desc.queue.wait();
    sycl::free(temp_host, desc.queue);
    return res;
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP
