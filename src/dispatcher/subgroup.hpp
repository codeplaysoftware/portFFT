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

#ifndef SYCL_FFT_DISPATCHER_SUBGROUP_HPP
#define SYCL_FFT_DISPATCHER_SUBGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace sycl_fft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<int> factor_wi_spec_const{};
constexpr static sycl::specialization_id<int> factor_sg_spec_const{};

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
std::size_t get_global_size_subgroup(std::size_t n_transforms, std::size_t factor_sg, std::size_t subgroup_size,
                                     std::size_t n_compute_units) {
  std::size_t maximum_n_sgs = 2 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / SYCLFFT_SGS_IN_WG;
  std::size_t wg_size = subgroup_size * SYCLFFT_SGS_IN_WG;

  std::size_t n_ffts_per_wg = (subgroup_size / factor_sg) * SYCLFFT_SGS_IN_WG;
  std::size_t n_wgs_we_can_utilize = divideCeil(n_transforms, n_ffts_per_wg);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by a subgroup.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam FactorWI factor of the FFT size. How many elements per FFT are processed by one workitem
 * @tparam FactorSG factor of the FFT size. How many workitems in a subgroup work on the same FFT
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*FactorWI*FactorSG*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*FactorWI*FactorSG
 * values
 * @param n_transforms number of FT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, int FactorWI, int FactorSG, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void subgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                         std::size_t n_transforms, sycl::nd_item<1> it,
                                                         const T* twiddles, T scaling_factor) {
  constexpr int N_reals_per_wi = 2 * FactorWI;

  T priv[N_reals_per_wi];
  sycl::sub_group sg = it.get_sub_group();
  std::size_t subgroup_local_id = sg.get_local_linear_id();
  std::size_t subgroup_id = sg.get_group_id();
  constexpr std::size_t n_sgs_in_wg = SYCLFFT_SGS_IN_WG;
  std::size_t id_of_sg_in_kernel = subgroup_id + it.get_group_linear_id() * n_sgs_in_wg;
  std::size_t n_sgs_in_kernel = it.get_group_range(0) * n_sgs_in_wg;

  std::size_t n_ffts_per_sg = SubgroupSize / FactorSG;
  std::size_t max_wis_working = n_ffts_per_sg * FactorSG;
  std::size_t n_reals_per_fft = FactorSG * N_reals_per_wi;
  std::size_t n_reals_per_sg = n_ffts_per_sg * n_reals_per_fft;
  std::size_t id_of_fft_in_sg = subgroup_local_id / FactorSG;
  std::size_t id_of_fft_in_kernel = id_of_sg_in_kernel * n_ffts_per_sg + id_of_fft_in_sg;
  std::size_t n_ffts_in_kernel = n_sgs_in_kernel * n_ffts_per_sg;
  std::size_t id_of_wi_in_fft = subgroup_local_id % FactorSG;
  // the +1 is needed for workitems not working on useful data so they also
  // contribute to subgroup algorithms and data transfers in last iteration
  std::size_t rounded_up_n_ffts =
      roundUpToMultiple(n_transforms, n_ffts_per_sg) + (subgroup_local_id >= max_wis_working);

  global2local<pad::DONT_PAD, level::WORKGROUP, SubgroupSize>(it, twiddles, loc_twiddles, N_reals_per_wi * FactorSG);
  sycl::group_barrier(it.get_group());

  for (std::size_t i = id_of_fft_in_kernel; i < rounded_up_n_ffts; i += n_ffts_in_kernel) {
    bool working = subgroup_local_id < max_wis_working && i < n_transforms;
    std::size_t n_ffts_worked_on_by_sg = sycl::min(n_transforms - (i - id_of_fft_in_sg), n_ffts_per_sg);

    global2local<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(it, input, loc, n_ffts_worked_on_by_sg * n_reals_per_fft,
                                                             n_reals_per_fft * (i - id_of_fft_in_sg),
                                                             subgroup_id * n_reals_per_sg);

    sycl::group_barrier(sg);
    if (working) {
      local2private<N_reals_per_wi, pad::DO_PAD>(loc, priv, subgroup_local_id, N_reals_per_wi,
                                                 subgroup_id * n_reals_per_sg);
    }
    sg_dft<Dir, FactorWI, FactorSG>(priv, sg, loc_twiddles);
    unrolled_loop<0, N_reals_per_wi, 2>([&](int i) __attribute__((always_inline)) {
      priv[i] *= scaling_factor;
      priv[i + 1] *= scaling_factor;
    });
    if constexpr (FactorSG == SubgroupSize) {
      // in this case we get fully coalesced memory access even without going through local memory
      // TODO we may want to tune maximal `FactorSG` for which we use direct stores.
      if (working) {
        store_transposed<N_reals_per_wi, pad::DONT_PAD>(priv, output, id_of_wi_in_fft, FactorSG,
                                                        i * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
      }
    } else {
      if (working) {
        store_transposed<N_reals_per_wi, pad::DO_PAD>(priv, loc, id_of_wi_in_fft, FactorSG,
                                                      subgroup_id * n_reals_per_sg + id_of_fft_in_sg * n_reals_per_fft);
      }
      sycl::group_barrier(sg);
      local2global<pad::DO_PAD, level::SUBGROUP, SubgroupSize>(
          it, loc, output, n_ffts_worked_on_by_sg * n_reals_per_fft, subgroup_id * n_reals_per_sg,
          n_reals_per_fft * (i - id_of_fft_in_sg));
      sycl::group_barrier(sg);
    }
  }
}

/**
 * Selects appropriate template instantiation of subgroup implementation for
 * given FactorSG.
 *
 * @tparam Dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
 * @tparam FactorWI factor of the FFT size. How many elements per FFT are processed by one workitem
 * @tparam SubgroupSize size of the subgroup
 * @tparam T type of the scalar used for computations
 * @param input accessor or pointer to global memory containing input data
 * @param output accessor or pointer to global memory for output data
 * @param loc local accessor. Must have enough space for 2*FactorWI*factor_sg*SubgroupSize
 * values
 * @param loc_twiddles local accessor for twiddle factors. Must have enough space for 2*FactorWI*factor_sg
 * values
 * @param n_transforms number of FFT transforms to do in one call
 * @param it sycl::nd_item<1> for the kernel launch
 * @param twiddles pointer containing twiddles
 * @param scaling_factor Scaling factor applied to the result
 */
template <direction Dir, int FactorWI, int SubgroupSize, typename T>
__attribute__((always_inline)) void cross_sg_dispatcher(int factor_sg, const T* input, T* output, T* loc,
                                                        T* loc_twiddles, std::size_t n_transforms, sycl::nd_item<1> it,
                                                        const T* twiddles, T scaling_factor) {
  switch (factor_sg) {
#define SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(M)                                                                      \
  case M:                                                                                                         \
    if constexpr (M <= SubgroupSize && !fits_in_wi<T>(M * FactorWI)) {                                            \
      subgroup_impl<Dir, FactorWI, M, SubgroupSize>(input, output, loc, loc_twiddles, n_transforms, it, twiddles, \
                                                    scaling_factor);                                              \
    }                                                                                                             \
    break;
    // cross-sg size 1 cases are supported by workitem implementation
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(2)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(4)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(8)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(16)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(32)
    SYCL_FFT_CROSS_SG_DISPATCHER_IMPL(64)
    // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_CROSS_SG_DISPATCHER_IMPL
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename T_in, typename T_out>
sycl::event committed_descriptor<Scalar, Domain>::subgroup_impl::run_kernel(
    committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
    const std::vector<sycl::event>& dependencies) {
  constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
  std::size_t fft_size = desc.params.lengths[0];
  std::size_t n_transforms = desc.params.number_of_transforms;
  Scalar* twiddles = desc.twiddles_forward;
  int factor_sg = desc.factors[1];
  std::size_t global_size = detail::get_global_size_subgroup<Scalar>(n_transforms, static_cast<std::size_t>(factor_sg),
                                                                     SubgroupSize, desc.n_compute_units);
  std::size_t local_elements = num_scalars_in_local_mem(desc);
  std::size_t twiddle_elements = 2 * fft_size;
  return desc.queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.use_kernel_bundle(desc.exec_bundle);
    auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
    auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
    sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
    sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
    cgh.parallel_for<detail::subgroup_kernel<Scalar, Domain, Dir, mem, TransposeIn, SubgroupSize>>(
        sycl::nd_range<1>{{global_size}, {SubgroupSize * SYCLFFT_SGS_IN_WG}}, [=
    ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
          int factor_wi = kh.get_specialization_constant<detail::factor_wi_spec_const>();
          int factor_sg = kh.get_specialization_constant<detail::factor_sg_spec_const>();
          switch (factor_wi) {
#define SYCL_FFT_SG_WI_DISPATCHER_IMPL(N)                                                                            \
  case N:                                                                                                            \
    if constexpr (detail::fits_in_wi<Scalar>(N)) {                                                                   \
      detail::cross_sg_dispatcher<Dir, N, SubgroupSize>(factor_sg, &in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0],   \
                                                        &loc_twiddles[0], n_transforms, it, twiddles, scale_factor); \
    }                                                                                                                \
    break;
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(1)
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(2)
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(4)
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(8)
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(16)
            SYCL_FFT_SG_WI_DISPATCHER_IMPL(32)
        // We compile a limited set of configurations to limit the compilation time
#undef SYCL_FFT_SG_WI_DISPATCHER_IMPL
          }
        });
  });
}

template <typename Scalar, domain Domain>
void committed_descriptor<Scalar, Domain>::subgroup_impl::set_spec_constants(
    committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
  in_bundle.template set_specialization_constant<detail::factor_wi_spec_const>(desc.factors[0]);
  in_bundle.template set_specialization_constant<detail::factor_sg_spec_const>(desc.factors[1]);
}

template <typename Scalar, domain Domain>
std::size_t committed_descriptor<Scalar, Domain>::subgroup_impl::num_scalars_in_local_mem(committed_descriptor& desc) {
  int factor_sg = desc.factors[1];
  std::size_t n_ffts_per_sg = static_cast<std::size_t>(desc.used_sg_size / factor_sg);
  return detail::pad_local(2 * desc.params.lengths[0] * n_ffts_per_sg) * SYCLFFT_SGS_IN_WG;
}

template <typename Scalar, domain Domain>
Scalar* committed_descriptor<Scalar, Domain>::subgroup_impl::calculate_twiddles(committed_descriptor& desc) {
  int factor_wi = desc.factors[0];
  int factor_sg = desc.factors[1];
  Scalar* res = sycl::malloc_device<Scalar>(desc.params.lengths[0] * 2, desc.queue);
  sycl::range<2> kernel_range({static_cast<std::size_t>(factor_sg), static_cast<std::size_t>(factor_wi)});
  desc.queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(kernel_range, [=](sycl::item<2> it) {
      int n = static_cast<int>(it.get_id(0));
      int k = static_cast<int>(it.get_id(1));
      sg_calc_twiddles(factor_sg, factor_wi, n, k, res);
    });
  });
  desc.queue.wait();  // waiting once here can be better than depending on the event
                      // for all future calls to compute
  return res;
}

}  // namespace sycl_fft

#endif