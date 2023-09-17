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

constexpr static sycl::specialization_id<std::size_t> SpecConstFftSize{};
constexpr static sycl::specialization_id<int> SpecConstSGFactorWI{};
constexpr static sycl::specialization_id<int> SpecConstSGFactorSG{};
constexpr static sycl::specialization_id<level> SpecConstLevel{};

template <int KernelID, direction Dir, typename Scalar, domain Domain, memory Mem, transpose TransposeIn,
          transpose TransposeOut, bool ApplyLoadModifier, bool ApplyStoreModifier, bool ApplyScaleFactor,
          int SubgroupSize, typename TIn, typename TOut>
struct dispatch_kernel_struct {
  static sycl::event execute(TIn input_pointer, TOut output_pointer, committed_descriptor<Scalar, Domain>& desc,
                             std::size_t intermediate_twiddles_offset, std::size_t local_twiddles_offset,
                             Scalar scale_factor, std::size_t base_offset, std::size_t batch_num,
                             const std::vector<sycl::event>& dependencies = {}) {
    sycl::event event;
    std::size_t fft_size = desc.factors[KernelID];
    std::size_t committed_size = desc.params.lengths[0];
    std::size_t batch_size = desc.sub_batches[KernelID];
    Scalar* scratch_pointer = desc.scratch_1.get();
    Scalar* scratch_pointer_2 = desc.scratch_2.get();
    auto global_range = desc.launch_configurations[KernelID].first;
    auto local_range = desc.launch_configurations[KernelID].second;
    std::size_t local_mem_usage = desc.local_mem_per_factor[KernelID];
    auto level_id = desc.levels[KernelID];
    const Scalar* twiddles_ptr = static_cast<const Scalar*>(desc.twiddles_forward.get());
    const std::size_t* device_factors = static_cast<const std::size_t*>(desc.dev_factors.get());
    std::size_t num_factors = desc.factors.size();
    desc.queue.wait();
    for (std::size_t i = 0; i < desc.num_batches_in_l2 && i + batch_num < desc.params.number_of_transforms; i++) {
      desc.queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Scalar, 1> loc(local_mem_usage, cgh);
        sycl::local_accessor<Scalar, 1> loc_twiddles(
            [=]() {
              if (level_id == detail::level::WORKITEM || level_id == detail::level::WORKGROUP) {
                return static_cast<std::size_t>(0);
              }
              return 2 * fft_size;
            }(),
            cgh);
        cgh.depends_on(dependencies);
        sycl::local_accessor<Scalar, 1> loc_store_modifier(local_mem_usage, cgh);
        cgh.use_kernel_bundle(desc.exec_bundle[KernelID]);
        cgh.parallel_for<global_kernel<Scalar, Domain, Dir, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                                       ApplyStoreModifier, ApplyScaleFactor, SubgroupSize, KernelID>>(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
              auto level_spec_const = kh.get_specialization_constant<SpecConstLevel>();
              auto sub_batches_product = device_factors[2 * num_factors + KernelID - 1];
              if (KernelID == (num_factors - 1)) {
                sub_batches_product = device_factors[2 * num_factors + KernelID - 2];
              }
              for (std::size_t sub_batch = 0; sub_batch < sub_batches_product; sub_batch++) {
                std::size_t sub_batch_offset = 0;
                if constexpr (KernelID == 1) {
                  sub_batch_offset = sub_batch * device_factors[num_factors];
                } else {
                  unrolled_loop<0, KernelID, 1>([&](std::size_t j) PORTFFT_ALWAYS_INLINE {
                    if (j == KernelID - 1) {
                      sub_batch_offset += (sub_batch % device_factors[j]) * device_factors[num_factors + j];
                    } else {
                      // sub_batch_offset += (sub_batch / (sub_batch_product / inclusive_scan[j]) % factor[j]) *
                      // sub_batch[j]
                      sub_batch_offset += ((sub_batch / (sub_batches_product / device_factors[2 * num_factors + j])) %
                                           device_factors[j]) *
                                          device_factors[num_factors + j];
                    }
                  });
                }
                if (level_spec_const == detail::level::WORKITEM) {
                  std::size_t problem_size = kh.get_specialization_constant<SpecConstFftSize>();
                  workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier,
                                         ApplyScaleFactor, SubgroupSize, cooley_tukey_size_list_t>(
                      static_cast<const Scalar*>(&scratch_pointer[0]) + base_offset + 2 * i * committed_size +
                          sub_batch_offset,
                      &scratch_pointer[0] + base_offset + 2 * i * committed_size + sub_batch_offset, &loc[0],
                      batch_size, it, scale_factor, problem_size, static_cast<const Scalar*>(nullptr),
                      twiddles_ptr + intermediate_twiddles_offset, static_cast<Scalar*>(nullptr),
                      &loc_store_modifier[0]);
                } else if (level_spec_const == detail::level::SUBGROUP) {
                  int factor_wi = kh.get_specialization_constant<SpecConstSGFactorWI>();
                  int factor_sg = kh.get_specialization_constant<SpecConstSGFactorSG>();
                  subgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier,
                                         ApplyScaleFactor, SubgroupSize, Scalar, cooley_tukey_size_list_t>(
                      factor_wi, factor_sg,
                      static_cast<const Scalar*>(&scratch_pointer[0]) + base_offset + 2 * committed_size +
                          sub_batch_offset,
                      &scratch_pointer[0] + base_offset + 2 * committed_size + sub_batch_offset, &loc[0],
                      &loc_twiddles[0], batch_size, it, twiddles_ptr + local_twiddles_offset, scale_factor,
                      static_cast<const Scalar*>(nullptr), twiddles_ptr + intermediate_twiddles_offset,
                      static_cast<Scalar*>(nullptr), &loc_store_modifier[0]);
                }
              }
            });
      });
    }
    if (KernelID == (desc.factors.size() - 1)) {
      return event;
    }
    std::size_t incremented_local_twiddles_offset;
    if (level_id == detail::level::SUBGROUP) {
      incremented_local_twiddles_offset = local_twiddles_offset + 2 * fft_size;
    }
    std::size_t incremented_intermediate_twiddles_offset = intermediate_twiddles_offset + 2 * fft_size * batch_size;
    if (KernelID == (desc.factors.size() - 2)) {
      dispatch_kernel_struct<KernelID + 1, Dir, Scalar, Domain, Mem, detail::transpose::NOT_TRANSPOSED,
                             detail::transpose::NOT_TRANSPOSED, false, false, true, SubgroupSize, TIn,
                             TOut>::execute(input_pointer, output_pointer, desc,
                                            incremented_intermediate_twiddles_offset, incremented_local_twiddles_offset,
                                            scale_factor, base_offset, batch_num, dependencies);
    } else {
      dispatch_kernel_struct<KernelID + 1, Dir, Scalar, Domain, Mem, detail::transpose::TRANSPOSED,
                             detail::transpose::TRANSPOSED, false, true, false, SubgroupSize, TIn,
                             TOut>::execute(input_pointer, output_pointer, desc,
                                            incremented_intermediate_twiddles_offset, incremented_local_twiddles_offset,
                                            scale_factor, base_offset, batch_num, dependencies);
    }

    desc.queue.wait();
    for (std::size_t i = 0; i < desc.num_batches_in_l2 && i + batch_num < desc.params.number_of_transforms; i++) {
      desc.queue.submit([&](sycl::handler& cgh) {
        cgh.use_kernel_bundle(desc.transpose_kernel_bundle[KernelID]);
        auto out_acc_or_usm = &get_access<Scalar>(output_pointer, cgh)[0];
        out_acc_or_usm = scratch_pointer_2;
        sycl::local_accessor<Scalar, 2> loc({16, 32}, cgh);
        cgh.parallel_for<transpose_kernel<Scalar, Domain, Dir, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                                          ApplyStoreModifier, ApplyScaleFactor, SubgroupSize, KernelID>>(
            sycl::nd_range<2>({round_up_to_multiple(fft_size, static_cast<std::size_t>(16)),
                               round_up_to_multiple(batch_size, static_cast<std::size_t>(16))},
                              {16, 16}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
              auto sub_batches_product = device_factors[2 * num_factors + KernelID - 1];
              for (std::size_t sub_batch = 0; sub_batch < sub_batches_product; sub_batch++) {
                std::size_t sub_batch_offset = 0;
                if constexpr (KernelID == 1) {
                  sub_batch_offset = sub_batch * device_factors[num_factors];
                } else {
                  unrolled_loop<0, KernelID, 1>([&](std::size_t j) PORTFFT_ALWAYS_INLINE {
                    if (j == KernelID - 1) {
                      sub_batch_offset += (sub_batch % device_factors[j]) * device_factors[num_factors + j];
                    } else {
                      // sub_batch_offset += ((sub_batch_product / inclusive_scan[j]) % factor[j]) * sub_batch[j]
                      sub_batch_offset += ((sub_batch / (sub_batches_product / device_factors[2 * num_factors + j])) %
                                           device_factors[j]) *
                                          device_factors[num_factors + j];
                    }
                  });
                }
                generic_transpose(fft_size, batch_size, 16,
                                  scratch_pointer + 2 * i * committed_size + base_offset + sub_batch_offset,
                                  out_acc_or_usm + base_offset + 2 * i * committed_size + sub_batch_offset, loc, it);
              }
            });
      });
    }
    desc.queue.wait();
    desc.scratch_1.swap(desc.scratch_2);
    return event;
  }
};

template <direction Dir, typename Scalar, domain Domain, memory Mem, transpose TransposeIn, transpose TransposeOut,
          bool ApplyLoadModifier, bool ApplyStoreModifier, bool ApplyScaleFactor, int SubgroupSize, typename TIn,
          typename TOut>
struct dispatch_kernel_struct<0, Dir, Scalar, Domain, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                              ApplyStoreModifier, ApplyScaleFactor, SubgroupSize, TIn, TOut> {
  static sycl::event execute(TIn input_pointer, TOut output_pointer, committed_descriptor<Scalar, Domain>& desc,
                             std::size_t intermediate_twiddles_offset, std::size_t local_twiddles_offset,
                             Scalar scale_factor, std::size_t base_offset, std::size_t batch_num,
                             const std::vector<sycl::event>& dependencies = {}) {
    sycl::event event;
    std::size_t fft_size = desc.factors[0];
    std::size_t committed_size = desc.params.lengths[0];
    std::size_t batch_size = desc.sub_batches[0];
    Scalar* scratch_pointer = desc.scratch_1.get();
    auto global_range = desc.launch_configurations[0].first;
    auto local_range = desc.launch_configurations[0].second;
    std::size_t local_mem_usage = desc.local_mem_per_factor[0];
    const Scalar* twiddles_ptr = static_cast<const Scalar*>(desc.twiddles_forward.get());
    detail::level level_id = desc.levels[0];
    for (std::size_t i = 0; i < desc.num_batches_in_l2 && i + batch_num < desc.params.number_of_transforms; i++) {
      desc.queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Scalar, 1> loc(local_mem_usage, cgh);
        sycl::local_accessor<Scalar, 1> loc_twiddles(
            [=]() {
              if (level_id == detail::level::WORKITEM || level_id == detail::level::WORKGROUP) {
                return static_cast<std::size_t>(0);
              }
              return 2 * fft_size;
            }(),
            cgh);
        sycl::local_accessor<Scalar, 1> loc_store_modifier(local_mem_usage, cgh);
        cgh.use_kernel_bundle(desc.exec_bundle[0]);
        auto in_ptr_or_acc = get_access<const Scalar>(input_pointer, cgh);
        cgh.depends_on(dependencies);
        cgh.parallel_for<global_kernel<Scalar, Domain, Dir, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                                       ApplyStoreModifier, ApplyScaleFactor, SubgroupSize>>(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
              auto level_spec_const = kh.get_specialization_constant<SpecConstLevel>();
              if (level_spec_const == detail::level::WORKITEM) {
                std::size_t problem_size = kh.get_specialization_constant<SpecConstFftSize>();
                workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier,
                                       ApplyScaleFactor, SubgroupSize, cooley_tukey_size_list_t, Scalar>(
                    static_cast<const Scalar*>(&in_ptr_or_acc[0] + base_offset + 2 * i * committed_size),
                    &scratch_pointer[0] + base_offset + 2 * i * committed_size, &loc[0], batch_size, it, scale_factor,
                    problem_size, static_cast<const Scalar*>(nullptr), twiddles_ptr + intermediate_twiddles_offset,
                    static_cast<Scalar*>(nullptr), &loc_store_modifier[0]);
              } else if (level_spec_const == detail::level::SUBGROUP) {
                int factor_wi = kh.get_specialization_constant<SpecConstSGFactorWI>();
                int factor_sg = kh.get_specialization_constant<SpecConstSGFactorSG>();
                subgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, ApplyLoadModifier, ApplyStoreModifier,
                                       ApplyScaleFactor, SubgroupSize, Scalar, cooley_tukey_size_list_t>(
                    factor_wi, factor_sg, &in_ptr_or_acc[0] + base_offset + 2 * i * committed_size,
                    &scratch_pointer[0] + base_offset + 2 * i * committed_size, &loc[0], &loc_twiddles[0], batch_size,
                    it, twiddles_ptr + local_twiddles_offset, scale_factor, static_cast<const Scalar*>(nullptr),
                    twiddles_ptr + intermediate_twiddles_offset, static_cast<Scalar*>(nullptr), &loc_store_modifier[0]);
              }
            });
      });
    }
    std::size_t incremented_local_twiddles_offset;
    if (level_id == detail::level::SUBGROUP) {
      incremented_local_twiddles_offset = local_twiddles_offset + 2 * fft_size;
    }
    std::size_t incremented_intermediate_twiddles_offset = intermediate_twiddles_offset + 2 * fft_size * batch_size;
    if (0 == (desc.factors.size() - 2)) {
      dispatch_kernel_struct<1, Dir, Scalar, Domain, Mem, detail::transpose::NOT_TRANSPOSED,
                             detail::transpose::NOT_TRANSPOSED, false, false, true, SubgroupSize, TIn,
                             TOut>::execute(input_pointer, output_pointer, desc,
                                            incremented_intermediate_twiddles_offset, incremented_local_twiddles_offset,
                                            scale_factor, base_offset, batch_num, dependencies);
    } else {
      dispatch_kernel_struct<1, Dir, Scalar, Domain, Mem, detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                             false, true, false, SubgroupSize, TIn,
                             TOut>::execute(input_pointer, output_pointer, desc,
                                            incremented_intermediate_twiddles_offset, incremented_local_twiddles_offset,
                                            scale_factor, base_offset, batch_num, dependencies);
    }
    desc.queue.wait();
    // TODO: This is not a good way to do it,
    //  This should be a single kernel, a batched matrix tranpose routine is the way to go
    if ((desc.factors.size() - 1) % 2 != 0 && 0 != (desc.factors.size() - 2)) {
      desc.scratch_1.swap(desc.scratch_2);
    }
    for (std::size_t i = 0; i < desc.num_batches_in_l2 && i + batch_num < desc.params.number_of_transforms; i++) {
      desc.queue.submit([&](sycl::handler& cgh) {
        cgh.use_kernel_bundle(desc.transpose_kernel_bundle[0]);
        auto out_acc_or_usm = get_access<Scalar>(output_pointer, cgh);
        sycl::local_accessor<Scalar, 2> loc({16, 32}, cgh);
        cgh.parallel_for<transpose_kernel<Scalar, Domain, Dir, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                                          ApplyStoreModifier, ApplyScaleFactor, SubgroupSize, 0>>(
            sycl::nd_range<2>({round_up_to_multiple(fft_size, static_cast<std::size_t>(16)),
                               round_up_to_multiple(batch_size, static_cast<std::size_t>(16))},
                              {16, 16}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
              generic_transpose(fft_size, batch_size, 16, scratch_pointer + 2 * i * committed_size + base_offset,
                                &out_acc_or_usm[0] + base_offset + 2 * i * committed_size, loc, it);
            });
      });
    }
    return event;
  }
};

template <direction Dir, typename Scalar, domain Domain, memory Mem, transpose TransposeIn, transpose TransposeOut,
          bool ApplyLoadModifier, bool ApplyStoreModifier, bool ApplyScaleFactor, int SubgroupSize, typename TIn,
          typename TOut>
struct dispatch_kernel_struct<MaxFactors, Dir, Scalar, Domain, Mem, TransposeIn, TransposeOut, ApplyLoadModifier,
                              ApplyStoreModifier, ApplyScaleFactor, SubgroupSize, TIn, TOut> {
  static sycl::event execute(TIn, TOut, committed_descriptor<Scalar, Domain>&, std::size_t, std::size_t, Scalar,
                             std::size_t, std::size_t, const std::vector<sycl::event>&) {
    sycl::event event;
    return event;
  }
};
}  // namespace detail
}  // namespace portfft

#endif  // PORTFFT_COMMON_GLOBAL_HPP