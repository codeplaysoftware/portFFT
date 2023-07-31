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
#include <common/global.hpp>
#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <common/workitem.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> workitem_spec_const_fft_size{};

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

  std::size_t n_wgs_we_can_utilize = divideCeil(n_transforms, wg_size);
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
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, int N, std::size_t SubgroupSize,
          bool ApplyLoadCallback = false, bool ApplyStoreCallback = true, typename T>
__attribute__((always_inline)) inline void workitem_impl(const T* input, T* output, T* loc, std::size_t n_transforms,
                                                         sycl::nd_item<1> it, T scaling_factor,
                                                         const T* callback_data_array = nullptr) {
  constexpr std::size_t N_reals = 2 * N;
  using T_vec = sycl::vec<T, 2>;
  T priv[N_reals];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t global_id = it.get_global_id(0);
  std::size_t global_size = it.get_global_range(0);
  std::size_t subgroup_id = sg.get_group_id();
  std::size_t local_offset = N_reals * SubgroupSize * subgroup_id;

  for (std::size_t i = global_id; i < roundUpToMultiple(n_transforms, SubgroupSize); i += global_size) {
    bool working = i < n_transforms;
    std::size_t n_working = sycl::min(SubgroupSize, n_transforms - i + subgroup_local_id);

    if constexpr (TransposeIn == detail::transpose::NOT_TRANSPOSED) {
      global2local<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(it, input, loc, N_reals * n_working,
                                                               N_reals * (i - subgroup_local_id), local_offset);
      sycl::group_barrier(sg);
    }
    if (working) {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        // Load directly into registers from global memory as all loads will be fully coalesced.
        // No need of going through local memory either as it is an unnecessary extra write step.
        unrolled_loop<0, N_reals, 2>([&](const std::size_t j) __attribute__((always_inline)) {
          
          reinterpret_cast<T_vec*>(&priv[j])->load(0, detail::get_global_multi_ptr(&input[i * 2 + j * n_transforms]));
        });
      } else {
        local2private<N_reals, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals, local_offset);
      }
      wi_dft<Dir, N, 1, 1>(priv, priv);
      if constexpr (ApplyStoreCallback) {
        std::size_t twiddle_base_offset = 2 * i * N_reals;
        detail::unrolled_loop<0, N_reals, 2>([&](const std::size_t j) __attribute__((always_inline)) {
          detail::pointwise_multiply(priv, callback_data_array, j, twiddle_base_offset + j);
        });
      }
      // hacky way so as to not use another template parameter.
      if constexpr (!ApplyStoreCallback) {
        unrolled_loop<0, N_reals, 2>([&](int j) __attribute__((always_inline)) {
          priv[j] *= scaling_factor;
          priv[j + 1] *= scaling_factor;
        });
      }
      private2local<N_reals, pad::DO_PAD>(priv, loc, subgroup_local_id, N_reals, local_offset);
    }
    sycl::group_barrier(sg);
    // Store back to global in the same manner irrespective of input data layout, as
    //  the transposed case is assumed to be used only in OOP scenario.
    if constexpr (TransposeOut == detail::transpose::NOT_TRANSPOSED) {
      local2global<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(it, loc, output, N_reals * n_working, local_offset,
                                                               N_reals * (i - subgroup_local_id));
    } else {
      detail::unrolled_loop<0, N_reals, 2>([&](const std::size_t j) {
        reinterpret_cast<T_vec*>(output)[i * 2 + j * n_transforms] = reinterpret_cast<T_vec*>(priv)[j];
      });
    }
    sycl::group_barrier(sg);
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
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, std::size_t SubgroupSize,
          typename SizeList, bool ApplyLoadCallback = false, bool ApplyStoreCallback = false, typename T>
__attribute__((always_inline)) void workitem_dispatch_impl(const T* input, T* output, T* loc, std::size_t n_transforms,
                                                           sycl::nd_item<1> it, T scaling_factor, std::size_t fft_size,
                                                           const T* callback_data_array = nullptr) {
  if constexpr (!SizeList::list_end) {
    constexpr int this_size = SizeList::size;
    if (fft_size == this_size) {
      if constexpr (detail::fits_in_wi<T>(this_size)) {
        workitem_impl<Dir, TransposeIn, TransposeOut, this_size, SubgroupSize, ApplyLoadCallback, ApplyStoreCallback>(
            input, output, loc, n_transforms, it, scaling_factor, callback_data_array);
      }
    } else {
      workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, typename SizeList::child_t,
                             ApplyLoadCallback, ApplyStoreCallback, T>(input, output, loc, n_transforms, it,
                                                                       scaling_factor, fft_size, callback_data_array);
    }
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, int SubgroupSize,
          bool ApplyLoadCallback, bool ApplyStoreCallback, typename T_in, typename T_out>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, TransposeOut, SubgroupSize,
                                                               ApplyLoadCallback, ApplyStoreCallback, T_in,
                                                               T_out>::inner<detail::level::WORKITEM, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
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
      cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                                               ApplyStoreCallback, SubgroupSize, 0>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * desc.num_sgs_per_wg}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::workitem_spec_const_fft_size>();
            detail::workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize,
                                           detail::cooley_tukey_size_list_t, ApplyLoadCallback, ApplyStoreCallback,
                                           Scalar>(&in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], n_transforms, it,
                                                   scale_factor, fft_size);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::workitem_spec_const_fft_size>(desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::WORKITEM,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    std::size_t num_scalars_per_sg = detail::pad_local(2 * fft_size * static_cast<std::size_t>(desc.used_sg_size));
    std::size_t max_n_sgs = desc.local_memory_size / sizeof(Scalar) / num_scalars_per_sg;
    desc.num_sgs_per_wg = std::min(static_cast<std::size_t>(PORTFFT_SGS_IN_WG), std::max(1ul, max_n_sgs));
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
