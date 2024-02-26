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
 * Helper function to determine the increment of twiddle pointer between factors
 * @param level Corresponding implementation for the previous factor
 * @param factor_size length of the factor
 * @return value to increment the pointer by
 */
inline IdxGlobal increment_twiddle_offset(detail::level level, Idx factor_size) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  if (level == detail::level::SUBGROUP) {
    return 2 * factor_size;
  }
  if (level == detail::level::WORKGROUP) {
    Idx n = detail::factorize(factor_size);
    Idx m = factor_size / n;
    return 2 * (factor_size + m + n);
  }
  return 0;
}

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
 * @param inclusive_scan pointer to global memory containing the inclusive scan of the factors
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
 * @param factors pointer to global memory containing factors of the input
 * @param inner_batches pointer to global memory containing the inner batch for each factor
 * @param inclusive_scan pointer to global memory containing the inclusive scan of the factors
 * @param num_factors Number of factors
 * @param iter_value Current iterator value of the flattened n-dimensional loop
 * @param outer_batch_product Inclusive Scan of factors at position level_num-1
 * @param storage complex storage: interleaved or split
 * @return outer batch offset to be applied for the current iteration
 */
PORTFFT_INLINE inline IdxGlobal get_outer_batch_offset(const IdxGlobal* factors, const IdxGlobal* inner_batches,
                                                       const IdxGlobal* inclusive_scan, Idx num_factors, Idx level_num,
                                                       IdxGlobal iter_value, IdxGlobal outer_batch_product,
                                                       complex_storage storage) {
  const Idx vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
  auto get_outer_batch_offset_impl = [&](Idx N) -> IdxGlobal {
    IdxGlobal outer_batch_offset = 0;
    for (Idx j = 0; j < N; j++) {
      if (j == N - 1) {
        outer_batch_offset += vec_size * (iter_value % factors[j]) * inner_batches[j];
      } else {
        outer_batch_offset +=
            vec_size * ((iter_value / (outer_batch_product / inclusive_scan[j])) % factors[j]) * inner_batches[j];
      }
    }
    return outer_batch_offset;
  };
  // Edge case when there are only two factors;
  if (level_num == 0 || (level_num == 1 && (level_num == num_factors - 1))) {
    return static_cast<std::size_t>(0);
  }
  if (level_num == 1) {
    return vec_size * iter_value * inner_batches[0];
  }
  if (level_num == num_factors - 1) {
    return get_outer_batch_offset_impl(level_num - 1);
  }
  return get_outer_batch_offset_impl(level_num);
}

/**
 * Device function responsible for calling the corresponding sub-implementation
 *
 * @tparam Scalar  Scalar type
 * @tparam SubgroupSize Subgroup size
 * @param input input pointer
 * @param output output pointer
 * @param input_imag input pointer for imaginary data
 * @param output_imag output pointer for imaginary data
 * @param implementation_twiddles pointer to global memory containing twiddles for the sub implementation
 * @param store_modifier store modifier data
 * @param input_loc pointer to local memory for storing the input
 * @param twiddles_loc pointer to local memory for storing the twiddles for sub-implementation
 * @param store_modifier_loc pointer to local memory for store modifier data
 * @param factors pointer to global memory containing factors of the input
 * @param inner_batches pointer to global memory containing the inner batch for each factor
 * @param inclusive_scan pointer to global memory containing the inclusive scan of the factors
 * @param batch_size Batch size for the corresponding input
 * @param global_data global data
 * @param kh kernel handler
 */
template <typename Scalar, Idx SubgroupSize>
PORTFFT_INLINE void dispatch_level(const Scalar* input, Scalar* output, const Scalar* input_imag, Scalar* output_imag,
                                   const Scalar* implementation_twiddles, const Scalar* load_modifier_data,
                                   const Scalar* store_modifier_data, Scalar* input_loc, Scalar* twiddles_loc,
                                   const IdxGlobal* factors, const IdxGlobal* inner_batches,
                                   const IdxGlobal* inclusive_scan, IdxGlobal batch_size,
                                   detail::global_data_struct<1> global_data, sycl::kernel_handler& kh) {
  complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
  auto level = kh.get_specialization_constant<GlobalSubImplSpecConst>();
  Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
  Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
  detail::elementwise_multiply multiply_on_store = kh.get_specialization_constant<detail::SpecConstMultiplyOnStore>();
  global_data.log_message_global(__func__, "dispatching sub implementation for factor num = ", level_num);
  IdxGlobal outer_batch_product = get_outer_batch_product(inclusive_scan, num_factors, level_num);
  for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
    IdxGlobal outer_batch_offset = get_outer_batch_offset(factors, inner_batches, inclusive_scan, num_factors,
                                                          level_num, iter_value, outer_batch_product, storage);
    IdxGlobal store_modifier_offset = [&]() {
      if (level_num == num_factors - 1 && multiply_on_store == detail::elementwise_multiply::APPLIED) {
        return outer_batch_offset;
      }
      return static_cast<IdxGlobal>(0);
    }();
    if (level == detail::level::WORKITEM) {
      workitem_impl<SubgroupSize, Scalar>(input + outer_batch_offset, output + outer_batch_offset,
                                          input_imag + outer_batch_offset, output_imag + outer_batch_offset, input_loc,
                                          batch_size, global_data, kh, load_modifier_data,
                                          store_modifier_data + store_modifier_offset);
    } else if (level == detail::level::SUBGROUP) {
      subgroup_impl<SubgroupSize, Scalar>(input + outer_batch_offset, output + outer_batch_offset,
                                          input_imag + outer_batch_offset, output_imag + outer_batch_offset, input_loc,
                                          twiddles_loc, batch_size, implementation_twiddles, global_data, kh,
                                          load_modifier_data, store_modifier_data + store_modifier_offset);
    } else if (level == detail::level::WORKGROUP) {
      workgroup_impl<SubgroupSize, Scalar>(input + outer_batch_offset, output + outer_batch_offset,
                                           input_imag + outer_batch_offset, output_imag + outer_batch_offset, input_loc,
                                           twiddles_loc, batch_size, implementation_twiddles, global_data, kh,
                                           load_modifier_data, store_modifier_data + store_modifier_offset);
    }
    sycl::group_barrier(global_data.it.get_group());
  }
}

/**
 * Prepares the launch of transposition at a particular level
 * @tparam Scalar Scalar type
 * @tparam Domain Domain of the FFT
 * @tparam TOut Output type
 * @param kd_struct kernel data struct
 * @param input input pointer
 * @param output output usm/buffer
 * @param factors_triple pointer to global memory containing factors, inner batches corresponding per factor, and the
 * inclusive scan of the factors
 * @param committed_size committed size of the FFT
 * @param num_batches_in_l2 number of batches in l2
 * @param n_transforms number of transforms as set in the descriptor
 * @param batch_start start of the current global batch being processed
 * @param total_factors total number of factors of the committed size
 * @param output_offset offset to the output pointer
 * @param queue queue associated with the commit
 * @param events event dependencies
 * @return sycl::event
 */
template <typename Scalar, domain Domain, typename TOut>
sycl::event transpose_level(const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct& kd_struct,
                            const Scalar* input, TOut output, const IdxGlobal* factors_triple, IdxGlobal committed_size,
                            Idx num_batches_in_l2, IdxGlobal n_transforms, IdxGlobal batch_start, Idx total_factors,
                            IdxGlobal output_offset, sycl::queue& queue, const std::vector<sycl::event>& events,
                            complex_storage storage) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  constexpr detail::memory Mem = std::is_pointer_v<TOut> ? detail::memory::USM : detail::memory::BUFFER;
  const IdxGlobal vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
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
      sycl::local_accessor<Scalar, 2> loc({16, 16 * static_cast<std::size_t>(vec_size)}, cgh);
      if (static_cast<Idx>(events.size()) < num_batches_in_l2) {
        cgh.depends_on(events);
      } else {
        // If events is a vector, the order of events is assumed to correspond to the order batches present in last
        // level cache.
        cgh.depends_on(events.at(static_cast<std::size_t>(batch_in_l2)));
      }
      const Scalar* offset_input = input + vec_size * committed_size * batch_in_l2;
      IdxGlobal output_offset_inner = output_offset + vec_size * committed_size * batch_in_l2;
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
#ifdef PORTFFT_KERNEL_LOG
      sycl::stream s{1024 * 16, 1024, cgh};
#endif
      std::size_t ld_output_rounded =
          detail::round_up_to_multiple(static_cast<std::size_t>(ld_output), static_cast<std::size_t>(16));
      std::size_t ld_input_rounded =
          detail::round_up_to_multiple(static_cast<std::size_t>(ld_input), static_cast<std::size_t>(16));
      PORTFFT_LOG_TRACE("Launching transpose kernel with global_size", ld_output_rounded, ld_input_rounded,
                        "local_size", 16, 16);
      cgh.parallel_for<detail::transpose_kernel<Scalar, Mem>>(
          sycl::nd_range<2>({ld_output_rounded, ld_input_rounded}, {16, 16}),
          [=
#ifdef PORTFFT_KERNEL_LOG
               ,
           global_logging_config = detail::global_logging_config
#endif
      ](sycl::nd_item<2> it, sycl::kernel_handler kh) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_KERNEL_LOG
                s, global_logging_config,
#endif
                it};
            global_data.log_message_global("entering transpose kernel - buffer impl");
            complex_storage storage = kh.get_specialization_constant<detail::SpecConstComplexStorage>();
            Idx level_num = kh.get_specialization_constant<GlobalSpecConstLevelNum>();
            Idx num_factors = kh.get_specialization_constant<GlobalSpecConstNumFactors>();
            IdxGlobal outer_batch_product = get_outer_batch_product(inclusive_scan, num_factors, level_num);
            for (IdxGlobal iter_value = 0; iter_value < outer_batch_product; iter_value++) {
              global_data.log_message_subgroup("iter_value: ", iter_value);
              IdxGlobal outer_batch_offset =
                  get_outer_batch_offset(factors_triple, inner_batches, inclusive_scan, num_factors, level_num,
                                         iter_value, outer_batch_product, storage);
              if (storage == complex_storage::INTERLEAVED_COMPLEX) {
                detail::generic_transpose<2>(ld_output, ld_input, 16, offset_input + outer_batch_offset,
                                             &out_acc_or_usm[0] + outer_batch_offset + output_offset_inner, loc,
                                             global_data);
              } else {
                detail::generic_transpose<1>(ld_output, ld_input, 16, offset_input + outer_batch_offset,
                                             &out_acc_or_usm[0] + outer_batch_offset + output_offset_inner, loc,
                                             global_data);
              }
            }
            global_data.log_message_global("exiting transpose kernel - buffer impl");
          });
    }));
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
 * @tparam SubgroupSize subgroup size
 * @tparam TIn input type
 * @param kd_struct associated kernel data struct with the factor
 * @param input input usm/buffer
 * @param output output pointer
 * @param input_imag input usm/buffer for imaginary data
 * @param output_imag output pointer for imaginary data
 * @param twiddles_ptr pointer to global memory containing the input
 * @param factors_triple pointer to global memory containing factors, inner batches corresponding per factor, and the
 * inclusive scan of the factors
 * @param intermediate_twiddle_offset offset value to the global pointer for twiddles in between factors
 * @param subimpl_twiddle_offset offset value to to the global pointer for obtaining the twiddles required for sub
 * implementation
 * @param input_global_offset offset value for the input pointer
 * @param committed_size committed size
 * @param num_batches_in_l2 number of batches in l2
 * @param n_transforms number of transforms as set in the descriptor
 * @param batch_start start of the current global batch being processed
 * @param total_factors total number of factors
 * @param storage complex storage: interleaved or split
 * @param dependencies dependent events
 * @param queue queue
 * @return vector events, one for each batch in l2
 */
template <typename Scalar, domain Domain, Idx SubgroupSize, typename TIn>
std::vector<sycl::event> compute_level(
    const typename committed_descriptor_impl<Scalar, Domain>::kernel_data_struct& kd_struct, const TIn& input,
    Scalar* output, const TIn& input_imag, Scalar* output_imag, const Scalar* load_modifier_data,
    const Scalar* store_modifier_data, const Scalar* subimpl_twiddles, const IdxGlobal* factors_triple,
    IdxGlobal input_global_offset, IdxGlobal committed_size, Idx num_batches_in_l2, IdxGlobal n_transforms,
    IdxGlobal batch_start, Idx total_factors, complex_storage storage, const std::vector<sycl::event>& dependencies,
    sycl::queue& queue) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  constexpr detail::memory Mem = std::is_pointer_v<TIn> ? detail::memory::USM : detail::memory::BUFFER;
  IdxGlobal local_range = kd_struct.local_range;
  IdxGlobal global_range = kd_struct.global_range;
  IdxGlobal batch_size = kd_struct.batch_size;
  std::size_t local_memory_for_input = kd_struct.local_mem_required;

  std::size_t loc_mem_for_twiddles = [&]() {
    if (kd_struct.level == detail::level::WORKITEM) {
      return std::size_t(0);
    }
    if (kd_struct.level == detail::level::SUBGROUP) {
      return 2 * kd_struct.length;
    }
    if (kd_struct.level == detail::level::WORKGROUP) {
      return std::size_t(0);
    }
    throw internal_error("illegal level encountered");
  }();

  const IdxGlobal* inner_batches = factors_triple + total_factors;
  const IdxGlobal* inclusive_scan = factors_triple + 2 * total_factors;
  const Idx vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
  std::vector<sycl::event> events;
  PORTFFT_LOG_TRACE("Local mem requirement - input:", local_memory_for_input, "twiddles", loc_mem_for_twiddles, "total",
                    local_memory_for_input + loc_mem_for_twiddles);
  for (Idx batch_in_l2 = 0; batch_in_l2 < num_batches_in_l2 && batch_in_l2 + batch_start < n_transforms;
       batch_in_l2++) {
    events.push_back(queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<Scalar, 1> loc_for_input(local_memory_for_input, cgh);
      sycl::local_accessor<Scalar, 1> loc_for_twiddles(loc_mem_for_twiddles, cgh);
      auto in_acc_or_usm = detail::get_access<const Scalar>(input, cgh);
      auto in_imag_acc_or_usm = detail::get_access<const Scalar>(input_imag, cgh);
      cgh.use_kernel_bundle(kd_struct.exec_bundle);
      if (static_cast<Idx>(dependencies.size()) < num_batches_in_l2) {
        cgh.depends_on(dependencies);
      } else {
        // If events is a vector, the order of events is assumed to correspond to the order batches present in last
        // level cache.
        cgh.depends_on(dependencies.at(static_cast<std::size_t>(batch_in_l2)));
      }

      Scalar* offset_output_imag = storage == complex_storage::INTERLEAVED_COMPLEX
                                       ? nullptr
                                       : output_imag + vec_size * batch_in_l2 * committed_size;
      Scalar* offset_output = output + vec_size * batch_in_l2 * committed_size;
      IdxGlobal input_batch_offset = vec_size * committed_size * batch_in_l2 + input_global_offset;
#ifdef PORTFFT_KERNEL_LOG
      sycl::stream s{1024 * 16, 1024, cgh};
#endif
      PORTFFT_LOG_TRACE("Launching kernel for global implementation with global_size", global_range, "local_size",
                        local_range);
      cgh.parallel_for<global_kernel<Scalar, Domain, Mem, SubgroupSize>>(
          sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(global_range)),
                            sycl::range<1>(static_cast<std::size_t>(local_range))),
          [=
#ifdef PORTFFT_KERNEL_LOG
               ,
           global_logging_config = detail::global_logging_config
#endif
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) PORTFFT_REQD_SUBGROUP_SIZE(SubgroupSize) {
            detail::global_data_struct global_data{
#ifdef PORTFFT_KERNEL_LOG
                s, global_logging_config,
#endif
                it};
            dispatch_level<Scalar, SubgroupSize>(
                &in_acc_or_usm[0] + input_batch_offset, offset_output, &in_imag_acc_or_usm[0] + input_batch_offset,
                offset_output_imag, subimpl_twiddles, load_modifier_data, store_modifier_data, &loc_for_input[0],
                &loc_for_twiddles[0], factors_triple, inner_batches, inclusive_scan, batch_size, global_data, kh);
          });
    }));
  }
  return events;
}

/**
 * Run the global implementation
 * @tparam Scalar Scalar type of committed descriptor
 * @tparam TIn Input type
 * @tparam TOut Output Type
 * @tparam SubgroupSize Subgroup Size
 * @tparam Domain Domain of the committed descriptor
 * @param input sycl::buffer / pointer containing the input data. In the case SPLIT_COMPLEX storage, it contains only
 * the real part
 * @param input_imag sycl::buffer / pointer containing the imaginary part of the input in the case where storage is
 * SPLIT_COMPLEX
 * @param output sycl::buffer / pointer containing the output data. In the case SPLIT_COMPLEX storage, it contains only
 * the real part
 * @param output_imag sycl::buffer / pointer containing the imaginary part of the output in the case where storage is
 * SPLIT_COMPLEX
 * @param desc committed descriptor
 * @param dimension_data Dimension struct pertaining to the dimension being dispatched
 * @param kernels vector containing the kernels for the computation
 * @param transpose_kernels vector containing transpose kernels as required by the global implementation
 * @param num_factors Number of factors
 * @param ptr_offset Offset applied to the twiddles pointer to obtain the start of twiddles applied between factors.
 * @param subimpl_twiddles_offset Offset applied to the twiddles pointer to obtain the start of twiddles required by the
 * level specific implementation.
 * @param kd_struct_offset offset applied to vector of kernels
 * @param i Batch being processed
 * @param num_batches number of transforms
 * @param batch_offset_input offset applied to the input
 * @param batch_offset_output offset applied to the output
 * @param storage complex storage scheme: split_complex / complex_interleaved
 * @param first_uses_load_modifier whether or not the very first kernel modifies data before computation
 * @param last_kernel_store_modifier_data whether or not the very last kernel modifies the data after computation
 * @return sycl::event waiting on the last transposes
 */
template <Idx SubgroupSize, typename Scalar, domain Domain, typename TIn, typename TOut>
sycl::event global_impl_driver(const TIn& input, const TIn& input_imag, TOut output, TOut output_imag,
                               committed_descriptor_impl<Scalar, Domain>& desc,
                               typename committed_descriptor_impl<Scalar, Domain>::dimension_struct& dimension_data,
                               const kernels_vec<Scalar, Domain>& kernels,
                               const kernels_vec<Scalar, Domain>& transpose_kernels, Idx num_factors,
                               IdxGlobal ptr_offset, IdxGlobal subimpl_twiddles_offset, std::size_t kd_struct_offset,
                               std::size_t i, IdxGlobal num_batches, IdxGlobal batch_offset_input,
                               IdxGlobal batch_offset_output, complex_storage storage,
                               detail::elementwise_multiply first_uses_load_modifier,
                               const Scalar* last_kernel_store_modifier_data) {
  std::vector<sycl::event> l2_events;
  sycl::event event;

  IdxGlobal intermediate_twiddles_offset = ptr_offset;
  IdxGlobal impl_twiddle_offset = subimpl_twiddles_offset;
  const IdxGlobal vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
  auto imag_offset = static_cast<IdxGlobal>(dimension_data.length) * vec_size;
  const Scalar* twiddles_ptr = static_cast<const Scalar*>(kernels.at(0).twiddles_forward.get());
  const IdxGlobal* factors_and_scan =
      static_cast<const IdxGlobal*>(dimension_data.factors_and_scan.get()) + 3 * kd_struct_offset;
  IdxGlobal dimension_size = static_cast<IdxGlobal>(dimension_data.length);
  Idx max_batches_in_l2 = dimension_data.num_batches_in_l2;

  auto& kernel0 = kernels.at(kd_struct_offset + 0);
  const Scalar* load_modifier_data = first_uses_load_modifier == detail::elementwise_multiply::APPLIED
                                         ? twiddles_ptr + dimension_data.bluestein_modifiers_offset
                                         : static_cast<const Scalar*>(nullptr);
  l2_events = detail::compute_level<Scalar, Domain, SubgroupSize>(
      kernel0, input, desc.scratch_ptr_1.get(), input_imag, desc.scratch_ptr_1.get() + imag_offset, load_modifier_data,
      twiddles_ptr + intermediate_twiddles_offset, twiddles_ptr + impl_twiddle_offset, factors_and_scan,
      batch_offset_input, dimension_size, max_batches_in_l2, num_batches, static_cast<IdxGlobal>(i), num_factors,
      storage, {event}, desc.queue);
  intermediate_twiddles_offset += 2 * kernel0.batch_size * static_cast<IdxGlobal>(kernel0.length);
  impl_twiddle_offset += increment_twiddle_offset(kernel0.level, static_cast<Idx>(kernel0.length));

  for (std::size_t factor_num = 1; factor_num < static_cast<std::size_t>(num_factors); factor_num++) {
    auto& current_kernel = kernels.at(kd_struct_offset + factor_num);
    if (static_cast<Idx>(factor_num) == num_factors - 1) {
      l2_events = detail::compute_level<Scalar, Domain, SubgroupSize>(
          current_kernel, static_cast<const Scalar*>(desc.scratch_ptr_1.get()), desc.scratch_ptr_1.get(),
          static_cast<const Scalar*>(desc.scratch_ptr_1.get() + imag_offset), desc.scratch_ptr_1.get() + imag_offset,
          static_cast<const Scalar*>(nullptr), last_kernel_store_modifier_data, twiddles_ptr + impl_twiddle_offset,
          factors_and_scan, 0, dimension_size, max_batches_in_l2, num_batches, static_cast<IdxGlobal>(i), num_factors,
          storage, l2_events, desc.queue);
    } else {
      l2_events = detail::compute_level<Scalar, Domain, SubgroupSize>(
          current_kernel, static_cast<const Scalar*>(desc.scratch_ptr_1.get()), desc.scratch_ptr_1.get(),
          static_cast<const Scalar*>(desc.scratch_ptr_1.get() + imag_offset), desc.scratch_ptr_1.get() + imag_offset,
          static_cast<const Scalar*>(nullptr), twiddles_ptr + intermediate_twiddles_offset,
          twiddles_ptr + impl_twiddle_offset, factors_and_scan, 0, dimension_size, max_batches_in_l2, num_batches,
          static_cast<IdxGlobal>(i), num_factors, storage, l2_events, desc.queue);
      intermediate_twiddles_offset += 2 * current_kernel.batch_size * static_cast<IdxGlobal>(current_kernel.length);
      impl_twiddle_offset += increment_twiddle_offset(current_kernel.level, static_cast<Idx>(current_kernel.length));
    }
  }

  event = desc.queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(l2_events);
    cgh.host_task([&]() {});
  });

  for (Idx num_transpose = num_factors - 2; num_transpose > 0; num_transpose--) {
    event = detail::transpose_level<Scalar, Domain>(
        transpose_kernels.at(static_cast<std::size_t>(num_transpose) + kd_struct_offset - 1), desc.scratch_ptr_1.get(),
        desc.scratch_ptr_2.get(), factors_and_scan, dimension_size, static_cast<Idx>(max_batches_in_l2), num_batches,
        static_cast<IdxGlobal>(i), num_factors, 0, desc.queue, {event}, storage);
    if (storage == complex_storage::SPLIT_COMPLEX) {
      event = detail::transpose_level<Scalar, Domain>(
          transpose_kernels.at(static_cast<std::size_t>(num_transpose)), desc.scratch_ptr_1.get() + imag_offset,
          desc.scratch_ptr_2.get() + imag_offset, factors_and_scan, dimension_size, static_cast<Idx>(max_batches_in_l2),
          num_batches, static_cast<IdxGlobal>(i), num_factors, 0, desc.queue, {event}, storage);
    }
    desc.scratch_ptr_1.swap(desc.scratch_ptr_2);
  }

  std::size_t transpose_kernel_pos = kd_struct_offset == 0 ? 0 : kd_struct_offset - 1;
  event = detail::transpose_level<Scalar, Domain>(
      transpose_kernels.at(transpose_kernel_pos), desc.scratch_ptr_1.get(), output, factors_and_scan, dimension_size,
      static_cast<Idx>(max_batches_in_l2), num_batches, static_cast<IdxGlobal>(i), num_factors, batch_offset_output,
      desc.queue, {event}, storage);
  if (storage == complex_storage::SPLIT_COMPLEX) {
    event = detail::transpose_level<Scalar, Domain>(
        transpose_kernels.at(transpose_kernel_pos), desc.scratch_ptr_1.get() + imag_offset, output_imag,
        factors_and_scan, dimension_size, static_cast<Idx>(max_batches_in_l2), num_batches, static_cast<IdxGlobal>(i),
        num_factors, batch_offset_output, desc.queue, {event}, storage);
  }
  return event;
}

}  // namespace detail
}  // namespace portfft

#endif
