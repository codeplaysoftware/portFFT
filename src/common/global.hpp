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

#ifndef PORTFFT_COMMON_GLOBAL_HPP
#define PORTFFT_COMMON_GLOBAL_HPP

#include <defines.hpp>
#include <descriptor.hpp>
#include <dispatcher/subgroup_dispatcher.hpp>
#include <dispatcher/workitem_dispatcher.hpp>
#include <enums.hpp>

namespace portfft {
namespace detail {

/**
 * Gets the inclusive scan of the factors at a particular index.
 *
 * @tparam KernelID  Recursion Level
 * @param device_factors device array containing, factors, and their inclusive scan
 * @param num_factors Number of factors
 * @return Outer batch product
 */

PORTFFT_INLINE std::size_t get_outer_batch_product(const std::size_t* device_factors, std::size_t num_factors,
                                                   std::size_t level_num) {
  if (level_num == 0) {
    return static_cast<std::size_t>(1);
  }
  if (level_num == num_factors - 1 && level_num != 1) {
    return device_factors[2 * num_factors + level_num - 2];
  }
  return device_factors[2 * num_factors + level_num - 1];
}

/**
 * Calculate the n-1'th dimensional array offset where N = KernelID, where
 * offset = dim_1 * stride_1 + ..... dim_{n-1} * stride_{n-1}
 *
 * @tparam KernelID Recursion Level
 * @param device_factors device_factors device array containing, factors, and their inclusive scan
 * @param num_factors Number of factors
 * @param iter_value Current iterator value of the flattened n-dimensional loop
 * @param outer_batch_product Inclusive Scan of factors at position KernelID-1
 * @return
 */
PORTFFT_INLINE std::size_t get_outer_batch_offset(const std::size_t* device_factors, std::size_t num_factors,
                                                  std::size_t level_num, std::size_t iter_value,
                                                  std::size_t outer_batch_product) {
  auto get_outer_batch_offset_impl = [&](std::size_t N) -> std::size_t {
    std::size_t outer_batch_offset = 0;
    for (std::size_t j = 0; j < N; j++) {
      if (j == N - 1) {
        outer_batch_offset += 2 * (iter_value % device_factors[j]) * device_factors[num_factors + j];
      }
      outer_batch_offset +=
          2 * ((iter_value / (outer_batch_product / device_factors[2 * num_factors + j])) % device_factors[j]) *
          device_factors[num_factors + j];
    }
    return outer_batch_offset;
  };
  if (level_num == 0) {
    return static_cast<std::size_t>(0);
  }
  if (level_num == 1) {
    return 2 * iter_value * device_factors[num_factors];
  }
  if (level_num == num_factors - 1) {
    return get_outer_batch_offset_impl(level_num - 1);
  }
  return get_outer_batch_offset_impl(level_num);
}

template <typename Scalar, domain Domain, int SubgroupSize, typename TOut>
void dispatch_transpose_kernels(committed_descriptor<Scalar, Domain>& desc, const Scalar* input, TOut& output,
                                std::size_t level_num, std::size_t batch, std::vector<sycl::event>& dependencies) {
  constexpr detail::memory Mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
  std::size_t N = desc.factors[level_num];
  std::size_t M = desc.sub_batches[level_num];
  std::size_t committed_size = desc.params.lengths[0];
  const std::size_t* device_factors = static_cast<const std::size_t*>(desc.dev_factors.get());
  std::vector<sycl::event> dependency_copy(dependencies);
  dependencies.clear();
  for (std::size_t batch_in_l2 = 0;
       batch_in_l2 < desc.num_batches_in_l2 && batch_in_l2 + batch < desc.params.number_of_transforms; batch_in_l2++) {
    dependencies.push_back(desc.queue.submit([&](sycl::handler& cgh) {
      auto out_acc_or_usm = get_access<Scalar>(output, cgh);
      sycl::local_accessor<Scalar, 2> loc({16, 32}, cgh);
      cgh.depends_on(dependency_copy);
      cgh.use_kernel_bundle(desc.transpose_kernel_bundle[level_num]);
      cgh.parallel_for<transpose_kernel<Scalar, Domain, Mem, SubgroupSize>>(
          sycl::nd_range<2>({round_up_to_multiple(N, static_cast<std::size_t>(16)),
                             round_up_to_multiple(M, static_cast<std::size_t>(16))},
                            {16, 16}),
          [=](sycl::nd_item<2> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            std::size_t level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
            std::size_t num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
            std::size_t outer_batch_product = get_outer_batch_product(device_factors, num_factors, level_num);
            for (std::size_t iter_value = 0; iter_value < outer_batch_product; iter_value++) {
              std::size_t outer_batch_offset =
                  get_outer_batch_offset(device_factors, num_factors, level_num, iter_value, outer_batch_product);
              generic_transpose(N, M, 16, input + (batch_in_l2 + batch) * 2 * committed_size + outer_batch_offset,
                                &out_acc_or_usm[0] + (batch_in_l2 + batch) * 2 * committed_size + outer_batch_offset,
                                loc, it);
            }
          });
    }));
  }
  if (level_num != 0) {
    std::vector<sycl::event> dependency_copy_2(dependencies);
    dependencies.clear();
    dependencies.push_back(desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependency_copy_2);
      cgh.host_task([&]() { desc.scratch_1.swap(desc.scratch_2); });
    }));
  }
}

template <direction Dir, typename Scalar, transpose TransposeIn, transpose TransposeOut,
          apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize>
PORTFFT_INLINE void dispatch_level(const Scalar* input, Scalar* output, const Scalar* implementation_twiddles,
                                   const Scalar* load_modifier_data, const Scalar* store_modifier_data,
                                   Scalar* input_loc, Scalar* twiddles_loc, Scalar* load_modifier_loc,
                                   Scalar* store_modifier_loc, const std::size_t* device_factors, Scalar scaling_factor,
                                   std::size_t batch_size, sycl::nd_item<1> it, sycl::kernel_handler kh) {
  auto level = kh.get_specialization_constant<GlobalSpecConstLevel>();
  std::size_t level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
  std::size_t num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
  std::size_t outer_batch_product = get_outer_batch_product(device_factors, num_factors, level_num);
  for (std::size_t iter_value = 0; iter_value < outer_batch_product; iter_value++) {
    std::size_t outer_batch_offset =
        get_outer_batch_offset(device_factors, num_factors, level_num, iter_value, outer_batch_product);
    if (level == detail::level::WORKITEM) {
      std::size_t fft_size = kh.get_specialization_constant<GlobalSpecConstFftSize>();
      workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor,
                             SubgroupSize, cooley_tukey_size_list_t, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, input_loc, batch_size, it, scaling_factor, fft_size,
          load_modifier_data, store_modifier_data, load_modifier_loc, store_modifier_loc);
    } else if (level == detail::level::SUBGROUP) {
      int factor_wi = kh.get_specialization_constant<GlobalSpecConstSGFactorWI>();
      int factor_sg = kh.get_specialization_constant<GlobalSpecConstSGFactorSG>();
      subgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier, ApplyScaleFactor,
                             SubgroupSize, Scalar, cooley_tukey_size_list_t>(
          factor_wi, factor_sg, input + outer_batch_offset, output + outer_batch_offset, input_loc, twiddles_loc,
          batch_size, it, implementation_twiddles, scaling_factor, load_modifier_data, store_modifier_data,
          load_modifier_loc, store_modifier_loc);
    }
  }
}

template <typename Scalar, direction Dir, domain Domain, detail::transpose TransposeIn, detail::transpose TransposeOut,
          apply_load_modifier ApplyLoadModifier, apply_store_modifier ApplyStoreModifier,
          apply_scale_factor ApplyScaleFactor, int SubgroupSize, typename TIn>
void dispatch_compute_kernels(committed_descriptor<Scalar, Domain>& desc, const TIn& input, Scalar scale_factor,
                              std::size_t level_num, std::size_t implementation_twiddle_offset,
                              std::size_t intermediate_twiddle_offset, std::size_t batch,
                              std::vector<sycl::event>& dependencies) {
  std::vector<sycl::event> dependency_copy(dependencies);
  constexpr detail::memory Mem = std::is_pointer<TIn>::value ? detail::memory::USM : detail::memory::BUFFER;
  dependencies.clear();
  std::size_t fft_size = desc.factors[level_num];
  std::size_t batch_size = desc.sub_batches[level_num];
  std::size_t committed_size = desc.params.lengths[0];
  auto [global_range, local_range] = desc.launch_configurations[level_num];
  std::size_t local_mem_usage = desc.local_mem_per_factor[level_num];
  const Scalar* twiddles_ptr = static_cast<const Scalar*>(desc.twiddles_forward.get());
  detail::level level_id = desc.levels[0];
  Scalar* scratch_1 = desc.scratch_1.get();
  const std::size_t* device_factors = static_cast<const std::size_t*>(desc.dev_factors.get());
  for (std::size_t batch_in_l2 = 0;
       batch_in_l2 < desc.num_batches_in_l2 && batch_in_l2 + batch < desc.params.number_of_transforms; batch_in_l2++) {
    dependencies.push_back(desc.queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<Scalar, 1> loc(local_mem_usage, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(
          [=]() {
            if (level_id == detail::level::WORKITEM || level_id == detail::level::WORKGROUP) {
              return static_cast<std::size_t>(0);
            }
            return 2 * fft_size;
          }(),
          cgh);
      sycl::local_accessor<Scalar, 1> loc_store_modifier(
          [=]() {
            if (level_num == desc.factors.size() - 1) {
              return static_cast<std::size_t>(0);
            }
            return local_mem_usage;
          }(),
          cgh);
      auto in_acc_or_usm = get_access<const Scalar>(input, cgh);
      cgh.use_kernel_bundle(desc.exec_bundle[level_num]);
      cgh.depends_on(dependency_copy);
      cgh.parallel_for<global_kernel<Scalar, Domain, Dir, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                                     ApplyStoreModifier, ApplyScaleFactor, SubgroupSize>>(
          sycl::nd_range<1>(global_range, local_range),
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
            dispatch_level<Dir, Scalar, transpose::NOT_TRANSPOSED, transpose::NOT_TRANSPOSED,
                           apply_load_modifier::NOT_APPLIED, apply_store_modifier::NOT_APPLIED,
                           apply_scale_factor::APPLIED, SubgroupSize>(
                &in_acc_or_usm[0] + 2 * (batch_in_l2 + batch) * committed_size,
                scratch_1 + 2 * (batch_in_l2 + batch) * committed_size, twiddles_ptr + implementation_twiddle_offset,
                static_cast<const Scalar*>(nullptr), twiddles_ptr + intermediate_twiddle_offset, &loc[0],
                &loc_twiddles[0], static_cast<Scalar*>(nullptr), &loc_store_modifier[0], device_factors, scale_factor,
                batch_size, it, kh);
          });
    }));
  }
}

}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_COMMON_GLOBAL_HPP
