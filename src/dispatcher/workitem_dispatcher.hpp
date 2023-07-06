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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
#define SYCL_FFT_DISPATCHER_WORKITEM_DISPATCHER_HPP

#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <common/workitem.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace sycl_fft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> workitem_spec_const_fft_size{};

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
std::size_t get_global_size_workitem(std::size_t n_transforms, std::size_t subgroup_size, std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / SYCLFFT_SGS_IN_WG;
  std::size_t wg_size = subgroup_size * SYCLFFT_SGS_IN_WG;

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
template <direction Dir, detail::transpose TransposeIn, int N, std::size_t SubgroupSize, typename T>
__attribute__((always_inline)) inline void workitem_impl(const T* input, T* output, T* loc, std::size_t n_transforms,
                                                         sycl::nd_item<1> it, T scaling_factor) {
  constexpr std::size_t N_reals = 2 * N;

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
        unrolled_loop<0, N_reals, 2>([&](const std::size_t j) __attribute__((always_inline)) {
          using T_vec = sycl::vec<T, 2>;
          reinterpret_cast<T_vec*>(&priv[j])->load(0, detail::get_global_multi_ptr(&input[i * 2 + j * n_transforms]));
        });
      } else {
        local2private<N_reals, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals, local_offset);
      }
      wi_dft<Dir, N, 1, 1>(priv, priv);
      unrolled_loop<0, N_reals, 2>([&](int i) __attribute__((always_inline)) {
        priv[i] *= scaling_factor;
        priv[i + 1] *= scaling_factor;
      });
      private2local<N_reals, pad::DO_PAD>(priv, loc, subgroup_local_id, N_reals, local_offset);
    }
    sycl::group_barrier(sg);
    local2global<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(it, loc, output, N_reals * n_working, local_offset,
                                                             N_reals * (i - subgroup_local_id));
    sycl::group_barrier(sg);
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename T_in, typename T_out>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, T_in, T_out>::inner<detail::level::WORKITEM, Dummy>{
  static sycl::event execute(
      committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
      const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t n_transforms = desc.params.number_of_transforms;
    std::size_t global_size = detail::get_global_size_workitem<Scalar>(n_transforms, SubgroupSize, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, TransposeIn, Dummy>::execute(desc);
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, mem, TransposeIn, SubgroupSize>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * SYCLFFT_SGS_IN_WG}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::workitem_spec_const_fft_size>();
            switch (fft_size) {
  #define SYCL_FFT_WI_DISPATCHER_IMPL(N)                                                                         \
    case N:                                                                                                      \
      if constexpr (detail::fits_in_wi<Scalar>(N)) {                                                             \
        detail::workitem_impl<Dir, TransposeIn, N, SubgroupSize>(&in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], \
                                                                n_transforms, it, scale_factor);                \
      }                                                                                                          \
      break;
              SYCL_FFT_WI_DISPATCHER_IMPL(1)
              SYCL_FFT_WI_DISPATCHER_IMPL(2)
              SYCL_FFT_WI_DISPATCHER_IMPL(4)
              SYCL_FFT_WI_DISPATCHER_IMPL(8)
              SYCL_FFT_WI_DISPATCHER_IMPL(16)
              SYCL_FFT_WI_DISPATCHER_IMPL(32)
          // We compile a limited set of configurations to limit the compilation time
  #undef SYCL_FFT_WI_DISPATCHER_IMPL
            }
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKITEM, Dummy>{
  static void execute(
      committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::workitem_spec_const_fft_size>(desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<
    detail::level::WORKITEM, detail::transpose::TRANSPOSED, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return detail::pad_local(2 * desc.params.lengths[0] * static_cast<std::size_t>(desc.used_sg_size)) *
           SYCLFFT_SGS_IN_WG;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<
    detail::level::WORKITEM, detail::transpose::NOT_TRANSPOSED, Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return detail::pad_local(2 * desc.params.lengths[0] * static_cast<std::size_t>(desc.used_sg_size)) *
           SYCLFFT_SGS_IN_WG;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::WORKITEM, Dummy>{
  static Scalar* execute(committed_descriptor& /*desc*/) {
    return nullptr;
  }
};

}  // namespace sycl_fft

#endif // SYCL_FFT_DISPATCHER_WORKITEM_DISPATCHER_HPP
