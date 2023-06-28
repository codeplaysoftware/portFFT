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

#ifndef SYCL_FFT_DISPATCHER_WORKGROUP_HPP
#define SYCL_FFT_DISPATCHER_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <common/workgroup.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <enums.hpp>

namespace sycl_fft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> workgroup_spec_const_fft_size{};

/**
 * Calculates the global size needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param n_transforms number of transforms
 * @param subgroup_size size of subgroup used by the compute kernel
 * @param n_compute_units number fo compute units on target device
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t get_global_size_workgroup(std::size_t n_transforms, std::size_t subgroup_size, std::size_t n_compute_units) {
  //TODO should this really be just a copy of workitem's?
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / SYCLFFT_SGS_IN_WG;
  std::size_t wg_size = subgroup_size * SYCLFFT_SGS_IN_WG;

  std::size_t n_wgs_we_can_utilize = divideCeil(n_transforms, wg_size);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
}

/**
 * Implementation of FFT for sizes that can be done by a workgroup.
 *
 * @tparam dir Direction of the FFT
 * @tparam fft_size Problem size
 * @tparam T Scalar type
 *
 * @param input global input pointer
 * @param output global output pointer
 * @param fft_size given problem size
 * @param loc Pointer to local memory
 * @param loc_twiddles pointer to twiddles residing in the local memory
 * @param n_transforms number of fft batch size
 * @param it Associated Iterator
 * @param twiddles Pointer to twiddles residing in the global memory
 * @param scaling_factor scaling factor applied to the result
 */
template <direction dir, std::size_t fft_size, int Subgroup_size, typename T>
__attribute__((always_inline)) inline void workgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                          std::size_t n_transforms, sycl::nd_item<1> it,
                                                          const T* twiddles, T scaling_factor) {
  std::size_t num_workgroups = it.get_group_range(0);
  std::size_t wg_id = it.get_group(0);
  std::size_t max_global_offset = 2 * (n_transforms - 1) * fft_size;
  std::size_t global_offset = 2 * fft_size * wg_id;
  std::size_t offset_increment = 2 * fft_size * num_workgroups;
  constexpr std::size_t N = detail::factorize(fft_size);
  constexpr std::size_t M = fft_size / N;
  const T* wg_twiddles = twiddles + 2 * (M + N);

  global2local<pad::DONT_PAD, level::WORKGROUP, Subgroup_size>(it, twiddles, loc_twiddles, 2 * (M + N));

  for (std::size_t offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    global2local<pad::DO_PAD, level::WORKGROUP, Subgroup_size>(it, input, loc, 2 * fft_size, offset);
    sycl::group_barrier(it.get_group());
    wg_dft<dir, fft_size, N, M, Subgroup_size>(loc, loc_twiddles, wg_twiddles, it, scaling_factor);
    local2global_transposed<N, M, SYCLFFT_SGS_IN_WG, Subgroup_size, detail::pad::DO_PAD>(it, loc, output, offset);
  }
}

}

template <typename Scalar, domain Domain>
template <direction dir, detail::transpose transpose_in, int Subgroup_size, typename T_in, typename T_out>
sycl::event committed_descriptor<Scalar, Domain>::workgroup_impl::run_kernel(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
  constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
  std::size_t n_transforms = desc.params.number_of_transforms;
  Scalar* twiddles = desc.twiddles_forward;
  std::size_t global_size = detail::get_global_size_workgroup<Scalar>(n_transforms, Subgroup_size, desc.n_compute_units);
  std::size_t local_elements = num_scalars_in_local_mem(desc);
  return desc.queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.use_kernel_bundle(desc.exec_bundle);
    auto in_acc_or_usm = detail::get_access<const Scalar>(in,cgh);
    auto out_acc_or_usm = detail::get_access<Scalar>(out,cgh);
    sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
    cgh.parallel_for<detail::workgroup_kernel<Scalar, Domain, dir, mem, transpose_in, Subgroup_size>>(
        sycl::nd_range<1>{{global_size}, {Subgroup_size * SYCLFFT_SGS_IN_WG}}, [=
    ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(Subgroup_size)]] {
      std::size_t fft_size = kh.get_specialization_constant<detail::workgroup_spec_const_fft_size>();
      switch (fft_size) {
    #define SYCL_FFT_WG_DISPATCHER_IMPL(N)                                                                    \
      case N:                                                                                                 \
        detail::workgroup_impl<dir, N, Subgroup_size>(&in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0], &loc[detail::pad_local(2 * fft_size)], n_transforms, it, twiddles, scale_factor); \
        break;
        SYCL_FFT_WG_DISPATCHER_IMPL(256)
        SYCL_FFT_WG_DISPATCHER_IMPL(512)
        SYCL_FFT_WG_DISPATCHER_IMPL(1024)
        SYCL_FFT_WG_DISPATCHER_IMPL(2048)
        SYCL_FFT_WG_DISPATCHER_IMPL(4096)
        SYCL_FFT_WG_DISPATCHER_IMPL(8192)
        // We compile a limited set of configurations to limit the compilation time
    #undef SYCL_FFT_WG_DISPATCHER_IMPL
      }
    });
  });
}


template <typename Scalar, domain Domain>
void committed_descriptor<Scalar, Domain>::workgroup_impl::set_spec_constants(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle){
  in_bundle.template set_specialization_constant<detail::workgroup_spec_const_fft_size>(desc.params.lengths[0]);
}

template <typename Scalar, domain Domain>
std::size_t committed_descriptor<Scalar, Domain>::workgroup_impl::num_scalars_in_local_mem(committed_descriptor& desc){
  std::size_t fft_size = desc.params.lengths[0];
  std::size_t N = static_cast<std::size_t>(desc.factors[0] * desc.factors[1]);
  std::size_t M = static_cast<std::size_t>(desc.factors[2] * desc.factors[3]);
  // working memory + twiddles for subgroup impl for the two sizes
  return detail::pad_local(2 * fft_size) + 2 * (M + N);
}

template <typename Scalar, domain Domain>
Scalar* committed_descriptor<Scalar, Domain>::workgroup_impl::calculate_twiddles(committed_descriptor& desc){
  int factor_wi_N = desc.factors[0];
  int factor_sg_N = desc.factors[1];
  int factor_wi_M = desc.factors[2];
  int factor_sg_M = desc.factors[3];
  std::size_t fft_size = desc.params.lengths[0];
  std::size_t N = static_cast<std::size_t>(factor_wi_N * factor_sg_N);
  std::size_t M = static_cast<std::size_t>(factor_wi_M * factor_sg_M);
  Scalar* res = sycl::malloc_device<Scalar>(2 * (M + N + fft_size), desc.queue);
  desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_N), static_cast<std::size_t>(factor_wi_N)}),
                      [=](sycl::item<2> it) {
                          int n = static_cast<int>(it.get_id(0));
                          int k = static_cast<int>(it.get_id(1));
                          sg_calc_twiddles(factor_sg_N, factor_wi_N, n, k, res + (2 * M));
                      });
  });
  desc.queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<2>({static_cast<std::size_t>(factor_sg_M), static_cast<std::size_t>(factor_wi_M)}),
                      [=](sycl::item<2> it) {
                          int n = static_cast<int>(it.get_id(0));
                          int k = static_cast<int>(it.get_id(1));
                          sg_calc_twiddles(factor_sg_M, factor_wi_M, n, k, res);
                      });
  });
  Scalar* global_pointer = res + 2 * (N + M);
  // Copying from pinned memory to device might be faster than from regular allocation
  Scalar* temp_host = sycl::malloc_host<Scalar>(2 * fft_size, desc.queue);

  for (std::size_t i = 0; i < N; i++) {
    for (std::size_t j = 0; j < M; j++) {
      std::size_t index = 2 * (i * M + j);
      temp_host[index] =
          static_cast<Scalar>(std::cos((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size)));
      temp_host[index + 1] =
          static_cast<Scalar>(std::sin((-2 * M_PI * static_cast<double>(i * j)) / static_cast<double>(fft_size)));
    }
  }
  desc.queue.copy(temp_host, global_pointer, 2 * fft_size);
  sycl::free(temp_host, desc.queue);
  desc.queue.wait();
  return res;
}

}

#endif