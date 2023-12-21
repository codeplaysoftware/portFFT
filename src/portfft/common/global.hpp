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

#include <sycl/sycl.hpp>

#include "portfft/common/helpers.hpp"
#include "portfft/common/transpose.hpp"
#include "portfft/defines.hpp"
#include "portfft/descriptor.hpp"
#include "portfft/dispatcher/subgroup_dispatcher.hpp"
#include "portfft/dispatcher/workgroup_dispatcher.hpp"
#include "portfft/dispatcher/workitem_dispatcher.hpp"

namespace portfft {
namespace detail {

/**
 * inner batches refers to the batches associated per factor which will be computed in a single implementation call
 * corresponding to that factor. Optimization note: currently the factors_triple pointer is in global memory, however
 * since all threads in the subgroup will access the same address, at least on Nvidia, this will turn into an optimal
 * broadcast operation. If such behaviour is not observed on other vendors, use constant memory once there is support
 * for it in SYCL
 */

/**
 * Gets the precomputed inclusive scan of the factors at a particular index.
 *
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param num_factors Number of factors
 * @param level_num factor number
 * @return Outer batch product
 */
PORTFFT_INLINE inline IdxGlobal get_outer_batch_product(const IdxGlobal* inclusive_scan, Idx num_factors,
                                                        Idx level_num) {
  // Edge case to handle 2 factor  case, in which it should equivalent to the Bailey 4 step method
  if (level_num == 0 || (level_num == 1 && (level_num == num_factors - 1))) {
    return static_cast<IdxGlobal>(1);
  }
  if (level_num == num_factors - 1 && level_num != 1) {
    return inclusive_scan[level_num - 2];
  }
  return inclusive_scan[level_num - 1];
}

/**
 * Calculate the n-1'th dimensional array offset where N = KernelID, where
 * offset = dim_1 * stride_1 + ..... dim_{n-1} * stride_{n-1}
 *
 * In the multi-factor algorithm, the input can be assumed as an n-dimensional tensor,
 * and computing the mth factor is equivalent to computing FFTs along the mth dimension.
 * To calculate the offset require to get to the start of mth dimension, this implementation function flattens the
 * required m-dimensional loop into the single loop (dispatch level), and this function calculates the offset.
 * Precomputed inclusive scans are used to further reduce the number of calculations required.
 *
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param num_factors Number of factors
 * @param iter_value Current iterator value of the flattened n-dimensional loop
 * @param outer_batch_product Inclusive Scan of factors at position level_num-1
 * @return outer batch offset to be applied for the current iteration
 */
PORTFFT_INLINE inline IdxGlobal get_outer_batch_offset(const IdxGlobal* factors, const IdxGlobal* inner_batches,
                                                       const IdxGlobal* inclusive_scan, Idx num_factors, Idx level_num,
                                                       IdxGlobal iter_value, IdxGlobal outer_batch_product) {
  auto get_outer_batch_offset_impl = [&](Idx N) -> IdxGlobal {
    IdxGlobal outer_batch_offset = 0;
    for (Idx j = 0; j < N; j++) {
      if (j == N - 1) {
        outer_batch_offset += 2 * (iter_value % factors[j]) * inner_batches[j];
      } else {
        outer_batch_offset +=
            2 * ((iter_value / (outer_batch_product / inclusive_scan[j])) % factors[j]) * inner_batches[j];
      }
    }
    return outer_batch_offset;
  };
  // Edge case when there are only two factors;
  if (level_num == 0 || (level_num == 1 && (level_num == num_factors - 1))) {
    return static_cast<std::size_t>(0);
  }
  if (level_num == 1) {
    return 2 * iter_value * inner_batches[0];
  }
  if (level_num == num_factors - 1) {
    return get_outer_batch_offset_impl(level_num - 1);
  }
  return get_outer_batch_offset_impl(level_num);
}

/**
 * Device function responsible for calling the corresponding sub-implementation
 *
 * @tparam Dir Direction of the FFT
 * @tparam Scalar  Scalar type
 * @tparam LayoutIn Input layout
 * @tparam LayoutOut Output layout
 * @tparam SubgroupSize Subgroup size
 * @param input input pointer
 * @param output output pointer
 * @param implementation_twiddles global twiddles pointer containing twiddles for the sub implementation
 * @param store_modifier store modifier data
 * @param input_loc local memory for storing the input
 * @param twiddles_loc local memory for storing the twiddles for sub-implementation
 * @param store_modifier_loc local memory for store modifier data
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param batch_size Batch size for the corresponding input
 * @param scale_factor scale_factor
 * @param global_data global data
 * @param kh kernel handler
 */
template <direction Dir, typename Scalar, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize>
PORTFFT_INLINE void dispatch_level(const Scalar* input, Scalar* output, const Scalar* implementation_twiddles,
                                   const Scalar* store_modifier_data, Scalar* input_loc, Scalar* twiddles_loc,
                                   Scalar* store_modifier_loc, const IdxGlobal* factors, const IdxGlobal* inner_batches,
                                   const IdxGlobal* inclusive_scan, IdxGlobal batch_size, Scalar scale_factor,
                                   detail::global_data_struct<1> global_data, sycl::kernel_handler& kh) {
  auto level = kh.get_specialization_constant<GlobalSubImplSpecConst>();
  Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
  Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
  global_data.log_message_global(__func__, "dispatching sub implementation for factor num = ", level_num);
  IdxGlobal outer_batch_product = get_outer_batch_product(inclusive_scan, num_factors, level_num);
  for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
    IdxGlobal outer_batch_offset = get_outer_batch_offset(factors, inner_batches, inclusive_scan, num_factors,
                                                          level_num, iter_value, outer_batch_product);
    if (level == detail::level::WORKITEM) {
      workitem_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, nullptr, nullptr, input_loc, batch_size,
          scale_factor, global_data, kh, static_cast<const Scalar*>(nullptr), store_modifier_data,
          static_cast<Scalar*>(nullptr), store_modifier_loc);
    } else if (level == detail::level::SUBGROUP) {
      subgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, nullptr, nullptr, input_loc, twiddles_loc,
          batch_size, implementation_twiddles, scale_factor, global_data, kh, static_cast<const Scalar*>(nullptr),
          store_modifier_data, static_cast<Scalar*>(nullptr), store_modifier_loc);
    } else if (level == detail::level::WORKGROUP) {
      workgroup_impl<Dir, SubgroupSize, LayoutIn, LayoutOut, Scalar>(
          input + outer_batch_offset, output + outer_batch_offset, nullptr, nullptr, input_loc, twiddles_loc,
          batch_size, implementation_twiddles, scale_factor, global_data, kh, static_cast<Scalar*>(nullptr),
          store_modifier_data);
    }
    sycl::group_barrier(global_data.it.get_group());
  }
}

/**
 * Utility function to launch the kernel when the input is a buffer
 * @tparam Scalar Scalar type
 * @tparam Dir Direction of the FFT
 * @tparam Domain Domain of the compute
 * @tparam LayoutIn Input layout
 * @tparam LayoutOut Output layout
 * @tparam SubgroupSize Subgroup size
 * @param input input of type sycl::accessor
 * @param output USM output pointer
 * @param loc_for_input local memory for input
 * @param loc_for_twiddles local memory for twiddles
 * @param loc_for_store_modifier local memory for store modifier data
 * @param multipliers_between_factors twiddles to be multiplied between factors
 * @param impl_twiddles twiddles required for sub implementation
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param n_transforms batch size corresposding to the factor
 * @param scale_factor scale factor
 * @param input_batch_offset offset for the input pointer
 * @param launch_params launch configuration, the global and local range with which the kernel will get launched
 * @param cgh associated command group handler
 */
template <typename Scalar, direction Dir, domain Domain, detail::layout LayoutIn, detail::layout LayoutOut,
          int SubgroupSize>
void launch_kernel(sycl::accessor<const Scalar, 1, sycl::access::mode::read>& input, Scalar* output,
                   sycl::local_accessor<Scalar, 1>& loc_for_input, sycl::local_accessor<Scalar, 1>& loc_for_twiddles,
                   sycl::local_accessor<Scalar, 1>& loc_for_store_modifier, const Scalar* multipliers_between_factors,
                   const Scalar* impl_twiddles, const IdxGlobal* factors, const IdxGlobal* inner_batches,
                   const IdxGlobal* inclusive_scan, IdxGlobal n_transforms, Scalar scale_factor,
                   IdxGlobal input_batch_offset, std::pair<sycl::range<1>, sycl::range<1>> launch_params,
                   sycl::handler& cgh) {
  auto [global_range, local_range] = launch_params;
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  cgh.parallel_for<global_kernel<Scalar, Domain, Dir, memory::BUFFER, LayoutIn, LayoutOut, SubgroupSize>>(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        dispatch_level<Dir, Scalar, LayoutIn, LayoutOut, SubgroupSize>(
            &input[0] + input_batch_offset, output, impl_twiddles, multipliers_between_factors, &loc_for_input[0],
            &loc_for_twiddles[0], &loc_for_store_modifier[0], factors, inner_batches, inclusive_scan, n_transforms,
            scale_factor, global_data, kh);
      });
}

/**
 * TODO: Launch the kernel directly from compute_level and remove the duplicated launch_kernel
 * Utility function to launch the kernel when the input is an USM
 * @tparam Scalar Scalar type
 * @tparam Dir Direction of the FFT
 * @tparam Domain Domain of the compute
 * @tparam LayoutIn Input layout
 * @tparam LayoutOut Output layout
 * @tparam SubgroupSize Subgroup size
 * @param input input pointer
 * @param output output pointer
 * @param loc_for_input local memory for input
 * @param loc_for_twiddles local memory for twiddles
 * @param loc_for_store_modifier local memory for store modifier data
 * @param multipliers_between_factors twiddles to be multiplied between factors
 * @param impl_twiddles twiddles required for sub implementation
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param n_transforms batch size corresposding to the factor
 * @param scale_factor scale factor
 * @param input_batch_offset offset for the input pointer
 * @param launch_params launch configuration, the global and local range with which the kernel will get launched
 * @param cgh associated command group handler
 */
template <typename Scalar, direction Dir, domain Domain, detail::layout LayoutIn, detail::layout LayoutOut,
          int SubgroupSize>
void launch_kernel(const Scalar* input, Scalar* output, sycl::local_accessor<Scalar, 1>& loc_for_input,
                   sycl::local_accessor<Scalar, 1>& loc_for_twiddles,
                   sycl::local_accessor<Scalar, 1>& loc_for_store_modifier, const Scalar* multipliers_between_factors,
                   const Scalar* impl_twiddles, const IdxGlobal* factors, const IdxGlobal* inner_batches,
                   const IdxGlobal* inclusive_scan, IdxGlobal n_transforms, Scalar scale_factor,
                   IdxGlobal input_batch_offset, std::pair<sycl::range<1>, sycl::range<1>> launch_params,
                   sycl::handler& cgh) {
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  auto [global_range, local_range] = launch_params;
  cgh.parallel_for<global_kernel<Scalar, Domain, Dir, memory::USM, LayoutIn, LayoutOut, SubgroupSize>>(
      sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        dispatch_level<Dir, Scalar, LayoutIn, LayoutOut, SubgroupSize>(
            &input[0] + input_batch_offset, output, impl_twiddles, multipliers_between_factors, &loc_for_input[0],
            &loc_for_twiddles[0], &loc_for_store_modifier[0], factors, inner_batches, inclusive_scan, n_transforms,
            scale_factor, global_data, kh);
      });
}

/**
 * TODO: Launch the kernel directly from transpose_level and remove the duplicated dispatch_transpose_kernel_impl
 * Utility function to launch the transpose kernel, when the output is a buffer
 * @tparam Scalar Scalar type
 * @param input input pointer
 * @param output output accessor
 * @param loc 2D local memory
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param output_offset offset to output pointer
 * @param ldb leading dimension of the output
 * @param lda leading dimension of the input
 * @param cgh associated command group handler
 */
template <typename Scalar>
static void dispatch_transpose_kernel_impl(const Scalar* input,
                                           sycl::accessor<Scalar, 1, sycl::access::mode::write>& output,
                                           sycl::local_accessor<Scalar, 2>& loc, const IdxGlobal* factors,
                                           const IdxGlobal* inner_batches, const IdxGlobal* inclusive_scan,
                                           IdxGlobal output_offset, IdxGlobal lda, IdxGlobal ldb, sycl::handler& cgh) {
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  cgh.parallel_for<detail::transpose_kernel<Scalar, memory::BUFFER>>(
      sycl::nd_range<2>({detail::round_up_to_multiple(static_cast<std::size_t>(lda), static_cast<std::size_t>(16)),
                         detail::round_up_to_multiple(static_cast<std::size_t>(ldb), static_cast<std::size_t>(16))},
                        {16, 16}),
      [=](sycl::nd_item<2> it, sycl::kernel_handler kh) {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
        Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
        IdxGlobal outer_batch_product = get_outer_batch_product(inclusive_scan, num_factors, level_num);
        for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
          IdxGlobal outer_batch_offset = get_outer_batch_offset(factors, inner_batches, inclusive_scan, num_factors,
                                                                level_num, iter_value, outer_batch_product);
          detail::generic_transpose(lda, ldb, 16, input + outer_batch_offset,
                                    &output[0] + outer_batch_offset + output_offset, loc, global_data);
        }
      });
}

/**
 * Utility function to launch the transpose kernel, when the output is a buffer
 * @tparam Scalar Scalar type
 * @param input input pointer
 * @param output output pointer
 * @param loc 2D local memory
 * @param factors global memory pointer containing factors of the input
 * @param inner_batches global memory pointer containing the inner batch for each factor
 * @param inclusive_scan global memory pointer containing the inclusive scan of the factors
 * @param output_offset offset to output pointer
 * @param ldb leading dimension of the output
 * @param lda leading dimension of the input
 * @param cgh associated command group handler
 */
template <typename Scalar>
static void dispatch_transpose_kernel_impl(const Scalar* input, Scalar* output, sycl::local_accessor<Scalar, 2>& loc,
                                           const IdxGlobal* factors, const IdxGlobal* inner_batches,
                                           const IdxGlobal* inclusive_scan, IdxGlobal output_offset, IdxGlobal lda,
                                           IdxGlobal ldb, sycl::handler& cgh) {
#ifdef PORTFFT_LOG
  sycl::stream s{1024 * 16, 1024, cgh};
#endif
  cgh.parallel_for<detail::transpose_kernel<Scalar, memory::USM>>(
      sycl::nd_range<2>({detail::round_up_to_multiple(static_cast<std::size_t>(lda), static_cast<std::size_t>(16)),
                         detail::round_up_to_multiple(static_cast<std::size_t>(ldb), static_cast<std::size_t>(16))},
                        {16, 16}),
      [=](sycl::nd_item<2> it, sycl::kernel_handler kh) {
        detail::global_data_struct global_data{
#ifdef PORTFFT_LOG
            s,
#endif
            it};
        Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
        Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
        IdxGlobal outer_batch_product = get_outer_batch_product(inclusive_scan, num_factors, level_num);
        for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
          IdxGlobal outer_batch_offset = get_outer_batch_offset(factors, inner_batches, inclusive_scan, num_factors,
                                                                level_num, iter_value, outer_batch_product);
          detail::generic_transpose(lda, ldb, 16, input + outer_batch_offset,
                                    &output[0] + outer_batch_offset + output_offset, loc, global_data);
        }
      });
}

/**
 * Prepares the launch of transposition at a particular level
 * @tparam Scalar Scalar type
 * @tparam Domain Domain of the FFT
 * @tparam TOut Output type
 * @param kd_struct kernel data struct
 * @param input input pointer
 * @param output output usm/buffer
 * @param factors_triple global memory pointer containing factors, inner batches corresponding per factor, and the
 * inclusive scan of the factors
 * @param committed_size committed size of the FFT
 * @param num_batches_in_l2 number of batches in l2
 * @param n_transforms number of transforms as set in the descriptor
 * @param batch_start start of the current global batch being processed
 * @param factor_num Current transpose level being run
 * @param total_factors total number of factors of the committed size
 * @param output_offset offset to the output pointer
 * @param queue queue associated with the commit
 * @param ptr1 shared_ptr for the first scratch pointer
 * @param ptr2 shared_ptr for the second scratch pointer
 * @param events event dependencies
 * @return sycl::event
 */
template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct,
                            const Scalar* input, TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size,
                            Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_num,
                            Idx total_factors, IdxGlobal output_offset, sycl::queue& queue,
                            std::shared_ptr<Scalar>& ptr1, std::shared_ptr<Scalar>& ptr2,
                            const std::vector<sycl::event>& events) {
  std::vector<sycl::event> transpose_events;
  IdxGlobal ld_input = kd_struct.factors.at(1);
  IdxGlobal ld_output = kd_struct.factors.at(0);
  const IdxGlobal* inner_batches = factors_triple + total_factors;
  const IdxGlobal* inclusive_scan = factors_triple + 2 * total_factors;
  for (Idx batch_in_l2 = 0;
       batch_in_l2 < num_batches_in_l2 && (static_cast<IdxGlobal>(batch_in_l2) + batch_start) < n_transforms;
       batch_in_l2++) {
    transpose_events.push_back(queue.submit([&](sycl::handler& cgh) {
      auto out_acc_or_usm = detail::get_access<Scalar>(output, cgh);
      sycl::local_accessor<Scalar, 2> loc({16, 32}, cgh);
      if (static_cast<Idx>(events.size()) < num_batches_in_l2) {
        cgh.depends_on(events);
      } else {
        // If events is a vector, the order of events is assumed to correspond to the order batches present in last
        // level cache.
        cgh.depends_on(events.at(static_cast<std::size_t>(batch_in_l2)));
      }
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
      detail::dispatch_transpose_kernel_impl<Scalar>(
          input + 2 * committed_size * batch_in_l2, out_acc_or_usm, loc, factors_triple, inner_batches, inclusive_scan,
          output_offset + 2 * committed_size * batch_in_l2, ld_output, ld_input, cgh);
    }));
  }
  if (factor_num != 0) {
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(transpose_events);
      cgh.host_task([&]() { ptr1.swap(ptr2); });
    });
  }
  return queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(transpose_events);
    cgh.host_task([&]() {});
  });
}

/**
 * Prepares the launch of fft compute at a particular level
 * @tparam Scalar Scalar type
 * @tparam Domain Domain of FFT
 * @tparam Dir Direction of the FFT
 * @tparam LayoutIn Input layout
 * @tparam LayoutOut output layout
 * @tparam SubgroupSize subgroup size
 * @tparam TIn input type
 * @param kd_struct associated kernel data struct with the factor
 * @param input input usm/buffer
 * @param output output pointer
 * @param twiddles_ptr global pointer containing the input
 * @param factors_triple global memory pointer containing factors, inner batches corresponding per factor, and the
 * inclusive scan of the factors
 * @param scale_factor scale factor
 * @param intermediate_twiddle_offset offset value to the global pointer for twiddles in between factors
 * @param subimpl_twiddle_offset offset value to to the global pointer for obtaining the twiddles required for sub
 * implementation
 * @param input_global_offset offset value for the input pointer
 * @param committed_size committed size
 * @param num_batches_in_l2 number of batches in l2
 * @param n_transforms number of transforms as set in the descriptor
 * @param batch_start start of the current global batch being processed
 * @param factor_id current factor being proccessed
 * @param total_factors total number of factors
 * @param dependencies even dependencies
 * @param queue queue
 * @return vector events, one for each batch in l2
 */
template <typename Scalar, domain Domain, direction Dir, detail::layout LayoutIn, detail::layout LayoutOut,
          Idx SubgroupSize, typename TIn>
std::vector<sycl::event> compute_level(
    const typename committed_descriptor<Scalar, Domain>::kernel_data_struct& kd_struct, const TIn input, Scalar* output,
    const Scalar* twiddles_ptr, const IdxGlobal* factors_triple, Scalar scale_factor,
    IdxGlobal intermediate_twiddle_offset, IdxGlobal subimpl_twiddle_offset, IdxGlobal input_global_offset,
    IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx factor_id,
    Idx total_factors, const std::vector<sycl::event>& dependencies, sycl::queue& queue) {
  IdxGlobal local_range = kd_struct.local_range;
  IdxGlobal global_range = kd_struct.global_range;
  IdxGlobal batch_size = kd_struct.batch_size;
  std::size_t local_memory_for_input = kd_struct.local_mem_required;
  std::size_t local_mem_for_store_modifier = [&]() -> std::size_t {
    if (factor_id < total_factors - 1) {
      if (kd_struct.level == detail::level::WORKITEM || kd_struct.level == detail::level::WORKGROUP) {
        return 1;
      }
      if (kd_struct.level == detail::level::SUBGROUP) {
        return 2 * kd_struct.length * static_cast<std::size_t>(local_range / 2);
      }
    }
    return std::size_t(1);
  }();
  std::size_t loc_mem_for_twiddles = [&]() {
    if (kd_struct.level == detail::level::WORKITEM) {
      return std::size_t(1);
    }
    if (kd_struct.level == detail::level::SUBGROUP) {
      return 2 * kd_struct.length;
    }
    if (kd_struct.level == detail::level::WORKGROUP) {
      return std::size_t(1);
    }
    throw internal_error("illegal level encountered");
  }();
  const IdxGlobal* inner_batches = factors_triple + total_factors;
  const IdxGlobal* inclusive_scan = factors_triple + 2 * total_factors;
  std::vector<sycl::event> events;
  for (Idx batch_in_l2 = 0; batch_in_l2 < num_batches_in_l2 && batch_in_l2 + batch_start < n_transforms;
       batch_in_l2++) {
    events.push_back(queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<Scalar, 1> loc_for_input(local_memory_for_input, cgh);
      sycl::local_accessor<Scalar, 1> loc_for_twiddles(loc_mem_for_twiddles, cgh);
      sycl::local_accessor<Scalar, 1> loc_for_modifier(local_mem_for_store_modifier, cgh);
      auto in_acc_or_usm = detail::get_access<const Scalar>(input, cgh);
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
      if (static_cast<Idx>(dependencies.size()) < num_batches_in_l2) {
        cgh.depends_on(dependencies);
      } else {
        // If events is a vector, the order of events is assumed to correspond to the order batches present in last
        // level cache.
        cgh.depends_on(dependencies.at(static_cast<std::size_t>(batch_in_l2)));
      }
      detail::launch_kernel<Scalar, Dir, Domain, LayoutIn, LayoutOut, SubgroupSize>(
          in_acc_or_usm, output + 2 * batch_in_l2 * committed_size, loc_for_input, loc_for_twiddles, loc_for_modifier,
          twiddles_ptr + intermediate_twiddle_offset, twiddles_ptr + subimpl_twiddle_offset, factors_triple,
          inner_batches, inclusive_scan, batch_size, scale_factor,
          2 * committed_size * batch_in_l2 + input_global_offset,
          {sycl::range<1>(static_cast<std::size_t>(global_range)),
           sycl::range<1>(static_cast<std::size_t>(local_range))},
          cgh);
    }));
  }
  return events;
}
}  // namespace detail
}  // namespace portfft

#endif
