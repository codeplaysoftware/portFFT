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
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> WorkitemSpecConstFftSize{};

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
 * @param dft_size the dft size. Expects to be < detail::MaxFftSizeWi.
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local memory pointer. Must have enough space for 2*N*SubgroupSize
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, std::size_t SubgroupSize, typename T>
__attribute__((always_inline)) inline void workitem_impl(int dft_size, const T* input, T* output, T* loc, std::size_t n_transforms,
                                                         sycl::nd_item<1> it, T scaling_factor) {
  T priv[2 * MaxFftSizeWi];
  T temp[2 * wi_temps(MaxFftSizeWi)];
  std::size_t n_reals = 2 * static_cast<std::size_t>(dft_size);

  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t global_id = it.get_global_id(0);
  std::size_t global_size = it.get_global_range(0);
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t local_offset = static_cast<std::size_t>(n_reals) * SubgroupSize * subgroup_id;
  constexpr std::size_t BankLinesPerPad = 1;

#pragma clang loop unroll(full)
  for (std::size_t i = global_id; i < round_up_to_multiple(n_transforms, SubgroupSize); i += global_size) {
    bool working = i < n_transforms;
    std::size_t n_working = sycl::min(SubgroupSize, n_transforms - i + subgroup_local_id);

    if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
      global2local<level::SUBGROUP, SubgroupSize, pad::DO_PAD>(it, input, loc, n_reals * n_working, BankLinesPerPad,
                                                               n_reals * (i - subgroup_local_id), local_offset);
      sycl::group_barrier(sg);
    }
    if (working) {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        // Load directly into registers from global memory as all loads will be fully coalesced.
        // No need of going through local memory either as it is an unnecessary extra write step.
        for (std::size_t j{0}; j < n_reals; j += 2) {
          using T_vec = sycl::vec<T, 2>;
          reinterpret_cast<T_vec*>(&priv[j])->load(0, detail::get_global_multi_ptr(&input[i * 2 + j * n_transforms]));
        }
      } else {
        local2private<pad::DO_PAD>(n_reals, loc, priv, subgroup_local_id, n_reals, BankLinesPerPad, local_offset);
      }
      wi_dft<Dir, 0>(dft_size, priv, 1, priv, 1, temp);
#pragma clang loop unroll(full)
      for (std::size_t j{0}; j < n_reals; j += 2) {
        priv[j] *= scaling_factor;
        priv[j + 1] *= scaling_factor;
      }
      private2local<pad::DO_PAD, BankLinesPerPad>(n_reals, priv, loc, subgroup_local_id, n_reals, local_offset);
    }
    sycl::group_barrier(sg);
    // Store back to global in the same manner irrespective of input data layout, as
    //  the transposed case is assumed to be used only in OOP scenario.
    local2global<level::SUBGROUP, SubgroupSize, pad::DO_PAD, BankLinesPerPad>(
        it, loc, output, n_reals * n_working, local_offset, n_reals * (i - subgroup_local_id));
    sycl::group_barrier(sg);
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
      cgh.use_kernel_bundle(desc.exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, Mem, TransposeIn, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::WorkitemSpecConstFftSize>();
            detail::workitem_impl<Dir, TransposeIn, SubgroupSize>(static_cast<int>(fft_size), &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], n_transforms, it, scale_factor);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::WorkitemSpecConstFftSize>(desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::WORKITEM,
                                                                                    TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    std::size_t num_scalars_per_sg =
        detail::pad_local(2 * desc.params.lengths[0] * static_cast<std::size_t>(desc.used_sg_size), 1);
    std::size_t max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
    desc.num_sgs_per_wg = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG), std::max(1UL, max_n_sgs));
    return num_scalars_per_sg * desc.num_sgs_per_wg;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKITEM, Dummy> {
  static Scalar* execute(committed_descriptor& /*desc*/) { return nullptr; }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
