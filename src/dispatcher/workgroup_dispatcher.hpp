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
#include <descriptor.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {
// specialization constants
constexpr static sycl::specialization_id<std::size_t> workgroup_spec_const_fft_size{};

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
  // TODO should this really be just a copy of workitem's?
  std::size_t maximum_n_sgs = 8 * n_compute_units * 64;
  std::size_t maximum_n_wgs = maximum_n_sgs / PORTFFT_SGS_IN_WG;
  std::size_t wg_size = subgroup_size * PORTFFT_SGS_IN_WG;

  std::size_t n_wgs_we_can_utilize = divideCeil(n_transforms, wg_size);
  return wg_size * sycl::min(maximum_n_wgs, n_wgs_we_can_utilize);
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
 * @param loc_twiddles pointer to twiddles residing in the local memory
 * @param n_transforms number of fft batch size
 * @param it Associated Iterator
 * @param twiddles Pointer to twiddles residing in the global memory
 * @param scaling_factor scaling factor applied to the result
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, std::size_t FFTSize,
          int SubgroupSize, bool ApplyLoadCallback = false, bool ApplyStoreCallback = true, typename T>
__attribute__((always_inline)) inline void workgroup_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                          std::size_t n_transforms, sycl::nd_item<1> it,
                                                          const T* twiddles, T scaling_factor,
                                                          T* callback_data_array = nullptr) {
  std::size_t num_workgroups = it.get_group_range(0);
  std::size_t wg_id = it.get_group(0);
  std::size_t max_global_offset = 2 * (n_transforms - 1) * FFTSize;
  std::size_t global_offset = 2 * FFTSize * wg_id;
  constexpr std::size_t N = detail::factorize(FFTSize);
  constexpr std::size_t M = FFTSize / N;
  const T* wg_twiddles = twiddles + 2 * (M + N);

  std::size_t max_num_batches_in_local_mem = [=]() {
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      return it.get_local_range(0) / 2;
    } else {
      return 1;
    }
  }();
  std::size_t offset_increment = 2 * FFTSize * num_workgroups * max_num_batches_in_local_mem;
  global2local<pad::DONT_PAD, level::WORKGROUP, SubgroupSize>(it, twiddles, loc_twiddles, 2 * (M + N));

  for (std::size_t offset = global_offset; offset <= max_global_offset; offset += offset_increment) {
    std::size_t num_batches_in_local_mem = [=]() {
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
        if (offset + it.get_local_range(0) / 2 < n_transforms) {
          return it.get_local_range(0) / 2;
        } else {
          return n_transforms - offset / (2 * FFTSize);
        }
      } else {
        return 1;
      }
    }();
    if constexpr (TransposeIn == detail::transpose::TRANSPOSED) {
      // Load in a transposed manner, similar to subgroup impl.
      global2local_transposed<pad::DO_PAD, level::WORKGROUP, T>(it, input, loc, 2 * offset, FFTSize, n_transforms,
                                                                num_batches_in_local_mem);
    } else {
      global2local<pad::DO_PAD, level::WORKGROUP, SubgroupSize>(it, input, loc, 2 * FFTSize, offset);
    }
    sycl::group_barrier(it.get_group());
    for (std::size_t i = 0; i < num_batches_in_local_mem; i++) {
      wg_dft<Dir, FFTSize, N, M, SubgroupSize, ApplyLoadCallback, ApplyStoreCallback>(
          loc + i * 2 * FFTSize, loc_twiddles, wg_twiddles, it, scaling_factor, callback_data_array);
      sycl::group_barrier(it.get_group());
      if constexpr (TransposeIn == detail::transpose::TRANSPOSED && TransposeOut == detail::transpose::NOT_TRANSPOSED) {
        // Once all batches in local memory have been processed, store all of them back to global memory in one go
        // Viewing it as a rectangle of height as problem size and length as the number of batches in local memory
        // Which needs to read in a transposed manner and stored in a contiguous one.
        local2global_transposed<detail::pad::DO_PAD>(it, N * M, num_batches_in_local_mem, max_num_batches_in_local_mem,
                                                     loc, output, offset);
      } else {
        if constexpr (TransposeOut == detail::transpose::NOT_TRANSPOSED) {
          local2global_transposed<detail::pad::DO_PAD>(it, N, M, M, loc, output, offset);
        } else {
          local2global<detail::pad::DO_PAD, detail::level::WORKGROUP>(it, loc, output, 2 * FFTSize, 0, offset);
        }
      }
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
 * @param n_transforms number of fft batch size
 * @param it Associated Iterator
 * @param twiddles Pointer to twiddles residing in the global memory
 * @param scaling_factor scaling factor applied to the result
 * @tparam fft_size Problem size
 */
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, int SubgroupSize,
          typename SizeList, bool ApplyLoadCallback = false, bool ApplyStoreCallback = true, typename T>
__attribute__((always_inline)) void workgroup_dispatch_impl(const T* input, T* output, T* loc, T* loc_twiddles,
                                                            std::size_t n_transforms, sycl::nd_item<1> it,
                                                            const T* twiddles, T scaling_factor, std::size_t fft_size,
                                                            T* callback_data_array = nullptr) {
  if constexpr (!SizeList::list_end) {
    constexpr size_t this_size = SizeList::size;
    if (fft_size == this_size) {
      if constexpr (!fits_in_sg<T>(this_size, SubgroupSize)) {
        workgroup_impl<Dir, TransposeIn, TransposeOut, this_size, SubgroupSize, ApplyLoadCallback, ApplyStoreCallback>(
            input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor, callback_data_array);
      }
    } else {
      workgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, typename SizeList::child_t,
                              ApplyLoadCallback, ApplyStoreCallback, T>(
          input, output, loc, loc_twiddles, n_transforms, it, twiddles, scaling_factor, fft_size, callback_data_array);
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
                                                               T_out>::inner<detail::level::WORKGROUP, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
    std::size_t n_transforms = desc.params.number_of_transforms;
    Scalar* twiddles = desc.twiddles_forward.get();
    std::size_t global_size =
        detail::get_global_size_workgroup<Scalar>(n_transforms, SubgroupSize, desc.n_compute_units);
    std::size_t local_elements =
        num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP, TransposeIn, Dummy>::execute(desc);
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(desc.exec_bundle);
      auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
      auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      cgh.parallel_for<detail::workgroup_kernel<Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                                                ApplyStoreCallback, SubgroupSize, 0>>(
          sycl::nd_range<1>{{global_size}, {SubgroupSize * PORTFFT_SGS_IN_WG}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t fft_size = kh.get_specialization_constant<detail::workgroup_spec_const_fft_size>();
            detail::workgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize,
                                            detail::cooley_tukey_size_list_t, ApplyLoadCallback, ApplyStoreCallback,
                                            Scalar>(&in_acc_or_usm[0], &out_acc_or_usm[0], &loc[0],
                                                    &loc[detail::pad_local(2 * fft_size)], n_transforms, it, twiddles,
                                                    scale_factor, fft_size);
          });
    });
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::WORKGROUP, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::workgroup_spec_const_fft_size>(desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::WORKGROUP,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    std::size_t N = static_cast<std::size_t>(desc.factors[0] * desc.factors[1]);
    std::size_t M = static_cast<std::size_t>(desc.factors[2] * desc.factors[3]);
    // working memory + twiddles for subgroup impl for the two sizes
    if (TransposeIn == detail::transpose::TRANSPOSED) {
      std::size_t num_batches_in_local_mem = static_cast<std::size_t>(desc.used_sg_size) * PORTFFT_SGS_IN_WG / 2;
      return detail::pad_local(2 * fft_size * num_batches_in_local_mem) + 2 * (M + N);
    } else {
      return detail::pad_local(2 * fft_size) + 2 * (M + N);
    }
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
    desc.queue.wait();
    sycl::free(temp_host, desc.queue);
    return res;
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_WORKGROUP_DISPATCHER_HPP
