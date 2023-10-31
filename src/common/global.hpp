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

#include <common/helpers.hpp>
#include <defines.hpp>
#include <dispatcher/subgroup_dispatcher.hpp>
#include <dispatcher/workgroup_dispatcher.hpp>
#include <dispatcher/workitem_dispatcher.hpp>

#include <sycl/sycl.hpp>

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

PORTFFT_INLINE IdxGlobal get_outer_batch_product(const IdxGlobal* device_factors, Idx num_factors, Idx level_num) {
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
PORTFFT_INLINE IdxGlobal get_outer_batch_offset(const IdxGlobal* device_factors, Idx num_factors, Idx level_num,
                                                IdxGlobal iter_value, IdxGlobal outer_batch_product) {
  auto get_outer_batch_offset_impl = [&](Idx N) -> IdxGlobal {
    IdxGlobal outer_batch_offset = 0;
    for (Idx j = 0; j < N; j++) {
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

template <direction Dir, typename Scalar, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize>
PORTFFT_INLINE void dispatch_level(const Scalar* input, Scalar* output, const Scalar* implementation_twiddles,
                                   const Scalar* store_modifier_data, Scalar* input_loc, Scalar* twiddles_loc,
                                   Scalar* store_modifier_loc, const IdxGlobal* device_factors, IdxGlobal batch_size,
                                   Scalar scale_factor, detail::global_data_struct global_data,
                                   sycl::kernel_handler& kh) {
  auto level = kh.get_specialization_constant<GlobalSubImplSpecConst>();
  Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
  Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
  IdxGlobal outer_batch_product = get_outer_batch_product(device_factors, num_factors, level_num);
  global_data.log_message_global(__func__, "Level num = ", level_num, "num_factors = ", num_factors);
  for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
    IdxGlobal outer_batch_offset =
        get_outer_batch_offset(device_factors, num_factors, level_num, iter_value, outer_batch_product);
    if (level == detail::level::WORKITEM) {
      workitem_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, input_loc, batch_size, scale_factor, global_data, kh,
          static_cast<const Scalar*>(nullptr), store_modifier_data, static_cast<Scalar*>(nullptr), store_modifier_loc);
    } else if (level == detail::level::SUBGROUP) {
      subgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, input_loc, twiddles_loc, batch_size,
          implementation_twiddles, scale_factor, global_data, kh, static_cast<const Scalar*>(nullptr),
          store_modifier_data, static_cast<Scalar*>(nullptr), store_modifier_loc);
    } else if (level == detail::level::WORKGROUP) {
      workgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, input_loc, twiddles_loc, batch_size,
          implementation_twiddles, scale_factor, global_data, kh, static_cast<Scalar*>(nullptr), store_modifier_data);
    }
  }
}

template <typename Scalar, direction Dir, domain Domain, detail::layout LayoutIn, detail::layout LayoutOut,
          int SubgroupSize>
void launch_kernel(sycl::accessor<const Scalar, 1, sycl::access::mode::read>& input, Scalar* output,
                   sycl::local_accessor<Scalar, 1>& loc_for_input, sycl::local_accessor<Scalar, 1>& loc_for_twiddles,
                   sycl::local_accessor<Scalar, 1>& loc_for_store_modifier, const Scalar* multipliers_between_factors,
                   const Scalar* impl_twiddles, const IdxGlobal* factors_and_scans, IdxGlobal n_transforms,
                   Scalar scale_factor, IdxGlobal input_batch_offset,
                   std::pair<sycl::range<1>, sycl::range<1>> launch_params, sycl::handler& cgh) {
  auto [global_range, local_range] = launch_params;
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  cgh.parallel_for<global_kernel<Scalar, Domain, Dir, memory::BUFFER, LayoutIn, LayoutOut, SubgroupSize>>(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        dispatch_level<Dir, Scalar, LayoutIn, LayoutOut, SubgroupSize>(
            &input[0] + input_batch_offset, output, impl_twiddles, multipliers_between_factors, &loc_for_input[0],
            &loc_for_twiddles[0], &loc_for_store_modifier[0], factors_and_scans, n_transforms, scale_factor,
            global_data, kh);
      });
}

template <typename Scalar, direction Dir, domain Domain, detail::layout LayoutIn, detail::layout LayoutOut,
          int SubgroupSize>
void launch_kernel(const Scalar* input, Scalar* output, sycl::local_accessor<Scalar, 1>& loc_for_input,
                   sycl::local_accessor<Scalar, 1>& loc_for_twiddles,
                   sycl::local_accessor<Scalar, 1>& loc_for_store_modifier, const Scalar* multipliers_between_factors,
                   const Scalar* impl_twiddles, const IdxGlobal* factors_and_scans, IdxGlobal n_transforms,
                   Scalar scale_factor, IdxGlobal input_batch_offset,
                   std::pair<sycl::range<1>, sycl::range<1>> launch_params, sycl::handler& cgh) {
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  auto [global_range, local_range] = launch_params;
  cgh.parallel_for<global_kernel<Scalar, Domain, Dir, memory::USM, LayoutIn, LayoutOut, SubgroupSize>>(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        dispatch_level<Dir, Scalar, LayoutIn, LayoutOut, SubgroupSize>(
            &input[0] + input_batch_offset, output, impl_twiddles, multipliers_between_factors, &loc_for_input[0],
            &loc_for_twiddles[0], &loc_for_store_modifier[0], factors_and_scans, n_transforms, scale_factor,
            global_data, kh);
      });
}

template <typename Scalar>
static void dispatch_transpose_kernel_impl(const Scalar* input,
                                           sycl::accessor<const Scalar, 1, sycl::access::mode::write>& output,
                                           sycl::local_accessor<Scalar, 2>& loc, const IdxGlobal* device_factors,
                                           IdxGlobal N, IdxGlobal M, sycl::handler& cgh) {
  cgh.parallel_for<detail::transpose_kernel<Scalar, memory::BUFFER>>(
      sycl::nd_range<2>({detail::round_up_to_multiple(static_cast<std::size_t>(N), static_cast<std::size_t>(16)),
                         detail::round_up_to_multiple(static_cast<std::size_t>(M), static_cast<std::size_t>(16))},
                        {16, 16}),
      [=](sycl::nd_item<2> it, sycl::kernel_handler kh) {
        Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
        Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
        IdxGlobal outer_batch_product = get_outer_batch_product(device_factors, num_factors, level_num);
        for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
          IdxGlobal outer_batch_offset =
              get_outer_batch_offset(device_factors, num_factors, level_num, iter_value, outer_batch_product);
          generic_transpose(N, M, 16, input + outer_batch_offset, &output[0] + outer_batch_offset, loc, it);
        }
      });
}

template <typename Scalar>
static void dispatch_transpose_kernel_impl(const Scalar* input, Scalar* output, sycl::local_accessor<Scalar, 2>& loc,
                                           const IdxGlobal* device_factors, IdxGlobal output_offset, IdxGlobal N,
                                           IdxGlobal M, sycl::handler& cgh) {
  cgh.parallel_for<detail::transpose_kernel<Scalar, memory::USM>>(
      sycl::nd_range<2>({detail::round_up_to_multiple(static_cast<std::size_t>(N), static_cast<std::size_t>(16)),
                         detail::round_up_to_multiple(static_cast<std::size_t>(M), static_cast<std::size_t>(16))},
                        {16, 16}),
      [=](sycl::nd_item<2> it, sycl::kernel_handler kh) {
        Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
        Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
        IdxGlobal outer_batch_product = get_outer_batch_product(device_factors, num_factors, level_num);
        for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
          IdxGlobal outer_batch_offset =
              get_outer_batch_offset(device_factors, num_factors, level_num, iter_value, outer_batch_product);
          generic_transpose(N, M, 16, input + outer_batch_offset, &output[0] + outer_batch_offset + output_offset, loc,
                            it);
        }
      });
}

}  // namespace detail

template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct,
                            const Scalar* input, TOut output, const IdxGlobal* device_factors, IdxGlobal committed_size,
                            Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_num,
                            IdxGlobal output_offset, sycl::queue& queue, std::shared_ptr<Scalar>& ptr1,
                            std::shared_ptr<Scalar>& ptr2, const std::vector<sycl::event>& events) {
  std::vector<sycl::event> transpose_events;
  IdxGlobal ld_input = kd_struct.factors.at(1);
  IdxGlobal ld_output = kd_struct.factors.at(0);
  for (Idx batch_in_l2 = 0;
       batch_in_l2 < num_batches_in_l2 && (static_cast<IdxGlobal>(batch_in_l2) + batch_start) < n_transforms;
       batch_in_l2++) {
    transpose_events.push_back(queue.submit([&](sycl::handler& cgh) {
      auto out_acc_or_usm = detail::get_access<Scalar>(output, cgh);
      sycl::local_accessor<Scalar, 2> loc({16, 32}, cgh);
      if (events.size() < num_batches_in_l2) {
        cgh.depends_on(events);
      } else {
        cgh.depends_on(events.at(static_cast<std::size_t>(batch_in_l2)));
      }
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
      detail::dispatch_transpose_kernel_impl<Scalar>(input + 2 * committed_size * batch_in_l2, out_acc_or_usm, loc,
                                                     device_factors, output_offset + 2 * committed_size * batch_in_l2,
                                                     ld_output, ld_input, cgh);
    }));
  }
  if (factor_num != 0) {
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(transpose_events);
      cgh.host_task([&]() { ptr1.swap(ptr2); });
    });
  } else
    return queue.submit([&](sycl::handler& cgh) { cgh.depends_on(transpose_events); });
  ;
}

template <typename Scalar, domain Domain, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut,
          Idx SubgroupSize, typename TIn>
std::vector<sycl::event> compute_level(
    const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct, const TIn input, Scalar* output,
    const Scalar* twiddles_ptr, const IdxGlobal* factors_and_scans, Scalar scale_factor,
    IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset, IdxGlobal input_global_offset,
    IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_id,
    Idx total_factors, const std::vector<sycl::event> dependencies, sycl::queue& queue) {
  IdxGlobal local_range = kd_struct.local_range;
  IdxGlobal global_range = kd_struct.global_range;
  std::cout << "local range = " << local_range << " Global Range = " << global_range << std::endl;
  IdxGlobal batch_size = kd_struct.batch_size;
  std::size_t local_memory_for_input = kd_struct.local_mem_required;
  std::size_t local_mem_for_store_modifier = [&]() {
    if (factor_id < total_factors - 1) {
      if (kd_struct.level == detail::level::WORKITEM) {
        return 2 * kd_struct.length * static_cast<std::size_t>(local_range);
      }
      if (kd_struct.level == detail::level::SUBGROUP) {
        return 2 * kd_struct.length * static_cast<std::size_t>(local_range / 2);
      }
    }
    return std::size_t(1);
  }();
  std::size_t loc_mem_for_twiddles = [&]() {
    if (factor_id < total_factors - 1) {
      if (kd_struct.level == detail::level::WORKITEM) {
        return std::size_t(1);
      }
      if (kd_struct.level == detail::level::SUBGROUP) {
        return 2 * kd_struct.length;
      }
    }
    return std::size_t(1);
  }();
  std::vector<sycl::event> events;
  for (Idx batch_in_l2 = 0; batch_in_l2 < num_batches_in_l2 && ((batch_in_l2 + batch_start) < n_transforms);
       batch_in_l2++) {
    events.push_back(queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<Scalar, 1> loc_for_input(local_memory_for_input, cgh);
      sycl::local_accessor<Scalar, 1> loc_for_twiddles(loc_mem_for_twiddles, cgh);
      sycl::local_accessor<Scalar, 1> loc_for_modifier(local_mem_for_store_modifier, cgh);
      auto in_acc_or_usm = detail::get_access<const Scalar>(input, cgh);
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
      if (dependencies.size() < num_batches_in_l2) {
        cgh.depends_on(dependencies);
      } else {
        cgh.depends_on(dependencies.at(static_cast<std::size_t>(batch_in_l2)));
      }
      detail::launch_kernel<Scalar, Dir, Domain, LayoutIn, LayoutOut, SubgroupSize>(
          in_acc_or_usm, output + 2 * batch_in_l2 * committed_size, loc_for_input, loc_for_twiddles, loc_for_modifier,
          twiddles_ptr + intermediate_twiddle_offset, twiddles_ptr + subimpl_twiddle_offset, factors_and_scans,
          batch_size, scale_factor, 2 * committed_size * batch_in_l2 + input_global_offset,
          {sycl::range<1>(static_cast<std::size_t>(global_range)),
           sycl::range<1>(static_cast<std::size_t>(local_range))},
          cgh);
    }));
  }
  return events;
}
}  // namespace portfft

#endif