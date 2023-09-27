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

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <common/workitem.hpp>
#include <defines.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

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
std::size_t get_global_size_workitem(std::size_t n_transforms, std::size_t subgroup_size, std::size_t num_sgs_per_wg,
                                     std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / num_sgs_per_wg;
  std::size_t wg_size = subgroup_size * num_sgs_per_wg;

  std::size_t n_wgs_we_can_utilize = divide_ceil(n_transforms, wg_size);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by independent work items.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
 * @tparam N size of each transform
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local memory pointer. Must have enough space for 2*N*SubgroupSize
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut,
          detail::apply_load_modifier ApplyLoadModifier, detail::apply_store_modifier ApplyStoreModifier,
          detail::apply_scale_factor ApplyScaleFactor, int N, std::size_t SubgroupSize, typename T>
PORTFFT_INLINE void workitem_impl(const T* input, T* output, T* loc, std::size_t n_transforms, sycl::nd_item<1> it,
                                  T scaling_factor, const T* load_modifier_data, const T* store_modifier_data,
                                  T* loc_load_modifier, T* loc_store_modifier) {
  constexpr std::size_t NReals = 2 * N;

  T priv[NReals];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t global_id = it.get_global_id(0);
  std::size_t global_size = it.get_global_range(0);
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t local_offset = NReals * SubgroupSize * subgroup_id;
  constexpr std::size_t BankLinesPerPad = 1;

  for (std::size_t i = global_id; i < round_up_to_multiple(n_transforms, SubgroupSize); i += global_size) {
    bool working = i < n_transforms;
    std::size_t n_working = sycl::min(SubgroupSize, n_transforms - i + subgroup_local_id);
    // TODO: There are possibilities where only one subgroup level barrier is required, relying on compliler to replace
    // and remove barriers as of now.
    if constexpr (ApplyLoadModifier == detail::apply_load_modifier::APPLIED) {
      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
          it, load_modifier_data, loc_load_modifier, NReals * n_working, NReals * (i - subgroup_local_id),
          local_offset);
      sycl::group_barrier(sg);
    }
    if constexpr (ApplyStoreModifier == detail::apply_store_modifier::APPLIED) {
      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
          it, store_modifier_data, loc_store_modifier, NReals * n_working, NReals * (i - subgroup_local_id),
          local_offset);
      sycl::group_barrier(sg);
    }
    if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
          it, input, loc, NReals * n_working, NReals * (i - subgroup_local_id), local_offset);
      sycl::group_barrier(sg);
    }
    if (working) {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        // Load directly into registers from global memory as all loads will be fully coalesced.
        // No need of going through local memory either as it is an unnecessary extra write step.
        unrolled_loop<0, NReals, 2>([&](const std::size_t j) PORTFFT_ALWAYS_INLINE {
          using T_vec = sycl::vec<T, 2>;
          reinterpret_cast<T_vec*>(&priv[j])->load(0, detail::get_global_multi_ptr(&input[i * 2 + j * n_transforms]));
        });
      } else {
        local2private<NReals, pad::DO_PAD, BankLinesPerPad>(loc, priv, subgroup_local_id, NReals, local_offset);
      }
      if constexpr (ApplyLoadModifier == detail::apply_load_modifier::APPLIED) {
        detail::unrolled_loop<0, N, 1>([&](const std::size_t j) PORTFFT_ALWAYS_INLINE {
          std::size_t base_offset = local_offset + subgroup_local_id * NReals + 2 * j;
          T modifier_real = loc_load_modifier[detail::pad_local(base_offset, BankLinesPerPad)];
          T modifier_complex = loc_load_modifier[detail::pad_local(base_offset + 1, BankLinesPerPad)];
          detail::multiply_complex(static_cast<const T>(priv[2 * j]), static_cast<const T>(priv[2 * j + 1]),
                                   static_cast<const T>(modifier_real), static_cast<const T>(modifier_complex),
                                   priv[2 * j], priv[2 * j + 1]);
        });
      }
      wi_dft<Dir, N, 1, 1>(priv, priv);
      // Relying on compiler optimizations to fuse store_modifier and scale factor.
      if constexpr (ApplyStoreModifier == detail::apply_store_modifier::APPLIED) {
        detail::unrolled_loop<0, N, 1>([&](const std::size_t j) PORTFFT_ALWAYS_INLINE {
          std::size_t base_offset =
              detail::pad_local(local_offset + subgroup_local_id * NReals + 2 * j, BankLinesPerPad);
          T modifier_real = loc_store_modifier[base_offset];
          T modifier_complex = loc_store_modifier[detail::pad_local(base_offset + 1, BankLinesPerPad)];
          if constexpr (Dir == direction::BACKWARD) {
            modifier_complex = -modifier_complex;
          }
          detail::multiply_complex(static_cast<const T>(priv[2 * j]), static_cast<const T>(priv[2 * j + 1]),
                                   static_cast<const T>(modifier_real), static_cast<const T>(modifier_complex),
                                   priv[2 * j], priv[2 * j + 1]);
        });
      }
      if constexpr (ApplyScaleFactor == detail::apply_scale_factor::APPLIED) {
        unrolled_loop<0, NReals, 2>([&](int i) PORTFFT_ALWAYS_INLINE {
          priv[i] *= scaling_factor;
          priv[i + 1] *= scaling_factor;
        });
      }
      if (TransposeOut == detail::transpose::NOT_TRANSPOSED) {
        private2local<NReals, pad::DO_PAD, BankLinesPerPad>(priv, loc, subgroup_local_id, NReals, local_offset);
      } else {
        unrolled_loop<0, NReals, 2>([&](const std::size_t j) PORTFFT_ALWAYS_INLINE {
          using T_vec = sycl::vec<T, 2>;
          reinterpret_cast<T_vec*>(&priv[j])->store(0, detail::get_global_multi_ptr(&output[i * 2 + j * n_transforms]));
        });
      }
    }
    if (TransposeOut == detail::transpose::NOT_TRANSPOSED) {
      sycl::group_barrier(sg);
      local2global<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
          it, loc, output, NReals * n_working, local_offset, NReals * (i - subgroup_local_id));
      sycl::group_barrier(sg);
    }
  }
}

/**
 * Launch specialized WI DFT size matching fft_size if one is available.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam TransposeIn whether input is transposed (interpreting it as a matrix of batch size times FFT size)
 * @tparam SubgroupSize size of the subgroup
 * @tparam SizeList The list of sizes that will be specialized.
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local memory pointer. Must have enough space for 2*N*SubgroupSize values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 * @param fft_size The size of the FFT.
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut,
          detail::apply_load_modifier ApplyLoadModifier, detail::apply_store_modifier ApplyStoreModifier,
          detail::apply_scale_factor ApplyScaleFactor, std::size_t SubgroupSize, typename SizeList, typename T>
PORTFFT_INLINE void workitem_dispatch_impl(const T* input, T* output, T* loc, std::size_t n_transforms,
                                           sycl::nd_item<1> it, T scaling_factor, std::size_t fft_size,
                                           const T* load_modifier_data = nullptr,
                                           const T* store_modifier_data = nullptr, T* loc_load_modifier = nullptr,
                                           T* loc_store_modifier = nullptr) {
  if constexpr (!SizeList::ListEnd) {
    constexpr int ThisSize = SizeList::Size;
    if (fft_size == ThisSize) {
      if constexpr (detail::fits_in_wi<T>(ThisSize)) {
        workitem_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor, ThisSize,
                      SubgroupSize, T>(input, output, loc, n_transforms, it, scaling_factor, load_modifier_data,
                                       store_modifier_data, loc_load_modifier, loc_store_modifier);
      }
    } else {
      workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor,
                             SubgroupSize, typename SizeList::child_t, T>(
          input, output, loc, n_transforms, it, scaling_factor, fft_size, load_modifier_data, store_modifier_data,
          loc_load_modifier, loc_store_modifier);
    }
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::WORKITEM, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t n_transforms = desc.params.number_of_transforms;
    std::size_t global_size =
        detail::get_global_size_workitem<Scalar>(n_transforms, SubgroupSize, desc.num_sgs_per_wg, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, TransposeIn, Dummy>::execute(desc);
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle[0]);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      cgh.parallel_for<
          detail::workitem_kernel<Scalar, Domain, Dir, Mem, TransposeIn, detail::transpose::NOT_TRANSPOSED,
                                  detail::apply_load_modifier::NOT_APPLIED, detail::apply_store_modifier::NOT_APPLIED,
                                  detail::apply_scale_factor::APPLIED, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::WorkitemSpecConstFftSize>();
            detail::workitem_dispatch_impl<
                Dir, TransposeIn, detail::transpose::NOT_TRANSPOSED, detail::apply_load_modifier::NOT_APPLIED,
                detail::apply_store_modifier::NOT_APPLIED, detail::apply_scale_factor::APPLIED, SubgroupSize,
                detail::cooley_tukey_size_list_t, Scalar>(&in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], n_transforms,
                                                          it, scale_factor, fft_size);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy> {
  static void execute(committed_descriptor& desc,
                      std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>& in_bundles) {
    for (auto& in_bundle : in_bundles) {
      in_bundle.template set_specialization_constant<detail::WorkitemSpecConstFftSize>(desc.params.lengths[0]);
    }
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::WORKITEM,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    std::size_t num_scalars_per_sg = detail::pad_local(2 * fft_size * static_cast<std::size_t>(desc.used_sg_size));
    std::size_t max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
    desc.num_sgs_per_wg = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG), std::max(1UL, max_n_sgs));
    return num_scalars_per_sg * desc.num_sgs_per_wg;
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKITEM,
                                                                                    TransposeIn, Dummy, std::size_t> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::WORKITEM, TransposeIn, Dummy>::execute(
        desc, fft_size);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKITEM,
                                                                                    TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::WORKITEM, TransposeIn, Dummy>::execute(
        desc, desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKITEM, Dummy> {
  static Scalar* execute(committed_descriptor& /*desc*/) { return nullptr; }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
