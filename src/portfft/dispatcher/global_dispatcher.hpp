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

#ifndef PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP

#include <sycl/sycl.hpp>

#include <cstring>

#include "portfft/common/global.hpp"
#include "portfft/common/subgroup.hpp"
#include "portfft/defines.hpp"
#include "portfft/enums.hpp"
#include "portfft/specialization_constant.hpp"

namespace portfft {
namespace detail {

/**
 * Helper function to obtain the global and local range for kernel corresponding to the factor
 * @param fft_size length of the factor
 * @param num_batches number of corresposing batches
 * @param level The implementation for the factor
 * @param n_compute_units compute_units available
 * @param subgroup_size Subgroup size chosen
 * @param n_sgs_in_wg Number of subgroups in a workgroup.
 * @return std::pair containing global and local range
 */
inline std::pair<IdxGlobal, IdxGlobal> get_launch_params(IdxGlobal fft_size, IdxGlobal num_batches, detail::level level,
                                                         Idx n_compute_units, Idx subgroup_size, Idx n_sgs_in_wg) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  IdxGlobal n_available_sgs = 8 * n_compute_units * 64;
  IdxGlobal wg_size = n_sgs_in_wg * subgroup_size;
  if (level == detail::level::WORKITEM) {
    IdxGlobal n_ffts_per_wg = wg_size;
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(std::min(n_wgs_required * wg_size, n_available_sgs), wg_size);
  }
  if (level == detail::level::SUBGROUP) {
    IdxGlobal n_ffts_per_sg = static_cast<IdxGlobal>(subgroup_size) / detail::factorize_sg(fft_size, subgroup_size);
    IdxGlobal n_ffts_per_wg = n_ffts_per_sg * n_sgs_in_wg;
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(std::min(n_wgs_required * wg_size, n_available_sgs), wg_size);
  }
  if (level == detail::level::WORKGROUP) {
    return std::make_pair(std::min(num_batches * wg_size, n_available_sgs), wg_size);
  }
  throw internal_error("illegal level encountered");
}

/**
 * Transposes A into B, for complex inputs only
 * @param a Input pointer
 * @param b Output pointer
 * @param lda leading dimension of `a`
 * @param ldb leading dimension of `b`
 * @param num_elements Total number of complex values in the matrix
 */
template <typename T>
void complex_transpose(const T* a, T* b, IdxGlobal lda, IdxGlobal ldb, IdxGlobal num_elements) {
  PORTFFT_LOG_FUNCTION_ENTRY();
  for (IdxGlobal i = 0; i < num_elements; i++) {
    IdxGlobal j = i / ldb;
    IdxGlobal k = i % ldb;
    b[2 * i] = a[2 * k * lda + 2 * j];
    b[2 * i + 1] = a[2 * k * lda + 2 * j + 1];
  }
}

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

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor_impl& desc, dimension_struct& /*dimension_data*/,
                         std::vector<kernel_data_struct>& kernels) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    std::vector<IdxGlobal> factors_idx_global;
    // Get factor sizes per level;
    for (const auto& kernel_data : kernels) {
      factors_idx_global.push_back(static_cast<IdxGlobal>(
          std::accumulate(kernel_data.factors.begin(), kernel_data.factors.end(), 1, std::multiplies<Idx>())));
    }

    std::vector<IdxGlobal> sub_batches;
    // Get sub batches
    for (std::size_t i = 0; i < factors_idx_global.size() - 1; i++) {
      sub_batches.push_back(std::accumulate(factors_idx_global.begin() + static_cast<long>(i + 1),
                                            factors_idx_global.end(), IdxGlobal(1), std::multiplies<IdxGlobal>()));
    }
    sub_batches.push_back(factors_idx_global.at(factors_idx_global.size() - 2));
    // calculate total memory required for twiddles;
    IdxGlobal mem_required_for_twiddles = 0;
    // First calculate mem required for twiddles between factors;
    for (std::size_t i = 0; i < factors_idx_global.size() - 1; i++) {
      mem_required_for_twiddles += 2 * factors_idx_global.at(i) * sub_batches.at(i);
    }
    // Now calculate mem required for twiddles per implementation
    std::size_t counter = 0;
    for (const auto& kernel_data : kernels) {
      if (kernel_data.level == detail::level::SUBGROUP) {
        mem_required_for_twiddles += 2 * factors_idx_global.at(counter);
      } else if (kernel_data.level == detail::level::WORKGROUP) {
        IdxGlobal factor_1 = detail::factorize(factors_idx_global.at(counter));
        IdxGlobal factor_2 = factors_idx_global.at(counter) / factor_1;
        mem_required_for_twiddles += 2 * (factor_1 * factor_2) + 2 * (factor_1 + factor_2);
      }
      counter++;
    }
    std::vector<Scalar> host_memory(static_cast<std::size_t>(mem_required_for_twiddles));
    std::vector<Scalar> scratch_space(static_cast<std::size_t>(mem_required_for_twiddles));
    PORTFFT_LOG_TRACE("Allocating global memory for twiddles for workgroup implementation. Allocation size",
                      mem_required_for_twiddles);
    Scalar* device_twiddles =
        sycl::malloc_device<Scalar>(static_cast<std::size_t>(mem_required_for_twiddles), desc.queue);

    // Helper Lambda to calculate twiddles
    auto calculate_twiddles = [](IdxGlobal N, IdxGlobal M, IdxGlobal& offset, Scalar* ptr) {
      for (IdxGlobal i = 0; i < N; i++) {
        for (IdxGlobal j = 0; j < M; j++) {
          double theta = -2 * M_PI * static_cast<double>(i * j) / static_cast<double>(N * M);
          ptr[offset++] = static_cast<Scalar>(std::cos(theta));
          ptr[offset++] = static_cast<Scalar>(std::sin(theta));
        }
      }
    };

    IdxGlobal offset = 0;
    // calculate twiddles to be multiplied between factors
    for (std::size_t i = 0; i < factors_idx_global.size() - 1; i++) {
      calculate_twiddles(sub_batches.at(i), factors_idx_global.at(i), offset, host_memory.data());
    }
    // Now calculate per twiddles.
    counter = 0;
    for (const auto& kernel_data : kernels) {
      if (kernel_data.level == detail::level::SUBGROUP) {
        for (Idx i = 0; i < kernel_data.factors.at(0); i++) {
          for (Idx j = 0; j < kernel_data.factors.at(1); j++) {
            double theta = -2 * M_PI * static_cast<double>(i * j) /
                           static_cast<double>(kernel_data.factors.at(0) * kernel_data.factors.at(1));
            auto twiddle =
                std::complex<Scalar>(static_cast<Scalar>(std::cos(theta)), static_cast<Scalar>(std::sin(theta)));
            host_memory[static_cast<std::size_t>(offset + static_cast<IdxGlobal>(j * kernel_data.factors.at(0) + i))] =
                twiddle.real();
            host_memory[static_cast<std::size_t>(
                offset + static_cast<IdxGlobal>((j + kernel_data.factors.at(1)) * kernel_data.factors.at(0) + i))] =
                twiddle.imag();
          }
        }
        offset += 2 * kernel_data.factors.at(0) * kernel_data.factors.at(1);
      } else if (kernel_data.level == detail::level::WORKGROUP) {
        Idx factor_n = kernel_data.factors.at(0) * kernel_data.factors.at(1);
        Idx factor_m = kernel_data.factors.at(2) * kernel_data.factors.at(3);
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(0)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(1)), offset, host_memory.data());
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(2)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(3)), offset, host_memory.data());
        // Calculate wg twiddles and transpose them
        calculate_twiddles(static_cast<IdxGlobal>(factor_n), static_cast<IdxGlobal>(factor_m), offset,
                           host_memory.data());
        for (Idx j = 0; j < factor_n; j++) {
          detail::complex_transpose(host_memory.data() + offset + 2 * j * factor_n, scratch_space.data(), factor_m,
                                    factor_n, factor_n * factor_m);
        }
      }
      counter++;
    }

    // Rearrage the twiddles between factors for optimal access patters in shared memory
    // Also take this opportunity to populate local memory size, and batch size, and launch params and local memory
    // usage Note, global impl only uses store modifiers
    // TODO: there is a heap corruption in workitem's access of loaded modifiers, hence loading from global directly for
    // now.
    counter = 0;
    for (auto& kernel_data : kernels) {
      kernel_data.batch_size = sub_batches.at(counter);
      kernel_data.length = static_cast<std::size_t>(factors_idx_global.at(counter));
      if (kernel_data.level == detail::level::WORKITEM) {
        // See comments in workitem_dispatcher for layout requirments.
        Idx num_sgs_in_wg = PORTFFT_SGS_IN_WG;
        if (counter < kernels.size() - 1) {
          kernel_data.local_mem_required = static_cast<std::size_t>(1);
        } else {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem(
              detail::level::WORKITEM, static_cast<std::size_t>(factors_idx_global.at(counter)),
              kernel_data.used_sg_size, {static_cast<Idx>(factors_idx_global.at(counter))}, num_sgs_in_wg,
              layout::PACKED);
        }
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idx_global.at(counter), sub_batches.at(counter), detail::level::WORKITEM,
                                      desc.n_compute_units, kernel_data.used_sg_size, num_sgs_in_wg);
        kernel_data.global_range = global_range;
        kernel_data.local_range = local_range;
      } else if (kernel_data.level == detail::level::SUBGROUP) {
        Idx num_sgs_in_wg = PORTFFT_SGS_IN_WG;
        // See comments in subgroup_dispatcher for layout requirements.
        IdxGlobal factor_sg = detail::factorize_sg(factors_idx_global.at(counter), kernel_data.used_sg_size);
        IdxGlobal factor_wi = factors_idx_global.at(counter) / factor_sg;
        if (counter < kernels.size() - 1) {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idx_global.at(counter)),
              kernel_data.used_sg_size, {static_cast<Idx>(factor_wi), static_cast<Idx>(factor_sg)}, num_sgs_in_wg,
              layout::BATCH_INTERLEAVED);
        } else {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idx_global.at(counter)),
              kernel_data.used_sg_size, {static_cast<Idx>(factor_wi), static_cast<Idx>(factor_sg)}, num_sgs_in_wg,
              layout::PACKED);
        }
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idx_global.at(counter), sub_batches.at(counter), detail::level::SUBGROUP,
                                      desc.n_compute_units, kernel_data.used_sg_size, num_sgs_in_wg);
        kernel_data.global_range = global_range;
        kernel_data.local_range = local_range;
      }
      counter++;
    }
    desc.queue.copy(host_memory.data(), device_twiddles, static_cast<std::size_t>(mem_required_for_twiddles)).wait();
    return device_twiddles;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::GLOBAL, Dummy> {
  static void execute(committed_descriptor_impl& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      Idx length, const std::vector<Idx>& factors, detail::level level, Idx factor_num, Idx num_factors,
                      Idx ffts_in_local) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    PORTFFT_LOG_TRACE("GlobalSubImplSpecConst:", level);
    in_bundle.template set_specialization_constant<detail::GlobalSubImplSpecConst>(level);
    PORTFFT_LOG_TRACE("GlobalSpecConstNumFactors:", num_factors);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(num_factors);
    PORTFFT_LOG_TRACE("GlobalSpecConstLevelNum:", factor_num);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(factor_num);
    if (level == detail::level::WORKITEM || level == detail::level::WORKGROUP) {
      PORTFFT_LOG_TRACE("SpecConstFftSize:", length);
      in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length);
    } else if (level == detail::level::SUBGROUP) {
      PORTFFT_LOG_TRACE("SubgroupFactorWISpecConst:", factors[1]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[1]);
      PORTFFT_LOG_TRACE("SubgroupFactorSGSpecConst:", factors[0]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[0]);
      // TODO set for workgroup if that is used
      PORTFFT_LOG_TRACE("SpecConstTransformsInLocal:", ffts_in_local);
      in_bundle.template set_specialization_constant<detail::SpecConstTransformsInLocal>(ffts_in_local);
    }
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::GLOBAL, Dummy> {
  static std::size_t execute(committed_descriptor_impl& /*desc*/, std::size_t /*length*/, Idx /*used_sg_size*/,
                             const std::vector<Idx>& /*factors*/, Idx& /*num_sgs_per_wg*/, layout /*input_layout*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    // No work required as all work done in calculate_twiddles;
    return 0;
  }
};

template <typename Scalar, domain Domain>
template <Idx SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::run_kernel_struct<SubgroupSize, TIn,
                                                                    TOut>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor_impl& desc, const TIn& in, TOut& out, const TIn& in_imag,
                             TOut& out_imag, const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, dimension_struct& dimension_data,
                             direction compute_direction, layout /*input_layout*/) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    complex_storage storage = desc.params.complex_storage;
    const IdxGlobal vec_size = storage == complex_storage::INTERLEAVED_COMPLEX ? 2 : 1;
    const auto& kernels =
        compute_direction == direction::FORWARD ? dimension_data.forward_kernels : dimension_data.backward_kernels;
    const Scalar* twiddles_ptr = static_cast<const Scalar*>(kernels.at(0).twiddles_forward.get());
    const IdxGlobal* factors_and_scan = static_cast<const IdxGlobal*>(dimension_data.factors_and_scan.get());
    std::size_t num_batches = desc.params.number_of_transforms;
    std::size_t max_batches_in_l2 = static_cast<std::size_t>(dimension_data.num_batches_in_l2);
    std::size_t imag_offset = dimension_data.length * max_batches_in_l2;
    IdxGlobal initial_impl_twiddle_offset = 0;
    Idx num_factors = dimension_data.num_factors;
    IdxGlobal committed_size = static_cast<IdxGlobal>(desc.params.lengths[0]);
    Idx num_transposes = num_factors - 1;
    std::vector<sycl::event> l2_events;
    sycl::event event = desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.host_task([&]() {});
    });
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_factors - 1); i++) {
      initial_impl_twiddle_offset += 2 * kernels.at(i).batch_size * static_cast<IdxGlobal>(kernels.at(i).length);
    }
    for (std::size_t i = 0; i < num_batches; i += max_batches_in_l2) {
      PORTFFT_LOG_TRACE("Global implementation working on batches", i, "through", i + max_batches_in_l2, "out of",
                        num_batches);
      IdxGlobal intermediate_twiddles_offset = 0;
      IdxGlobal impl_twiddle_offset = initial_impl_twiddle_offset;
      auto& kernel0 = kernels.at(0);
      PORTFFT_LOG_TRACE("Dispatching the kernel for factor 0 of global implementation");
      l2_events = detail::compute_level<Scalar, Domain, SubgroupSize>(
          kernel0, in, desc.scratch_ptr_1.get(), in_imag, desc.scratch_ptr_1.get() + imag_offset, twiddles_ptr,
          factors_and_scan, intermediate_twiddles_offset, impl_twiddle_offset,
          vec_size * static_cast<IdxGlobal>(i) * committed_size + input_offset, committed_size,
          static_cast<Idx>(max_batches_in_l2), static_cast<IdxGlobal>(num_batches), static_cast<IdxGlobal>(i), 0,
          dimension_data.num_factors, storage, {event}, desc.queue);
      detail::dump_device(desc.queue, "after factor 0:", desc.scratch_ptr_1.get(),
                          desc.params.number_of_transforms * dimension_data.length * 2, l2_events);
      intermediate_twiddles_offset += 2 * kernel0.batch_size * static_cast<IdxGlobal>(kernel0.length);
      impl_twiddle_offset += detail::increment_twiddle_offset(kernel0.level, static_cast<Idx>(kernel0.length));
      for (std::size_t factor_num = 1; factor_num < static_cast<std::size_t>(dimension_data.num_factors);
           factor_num++) {
        auto& current_kernel = kernels.at(factor_num);
        PORTFFT_LOG_TRACE("Dispatching the kernel for factor", factor_num, "of global implementation");
        if (static_cast<Idx>(factor_num) == dimension_data.num_factors - 1) {
          PORTFFT_LOG_TRACE("This is the last kernel");
        }
        l2_events = detail::compute_level<Scalar, Domain, SubgroupSize, const Scalar*>(
            current_kernel, desc.scratch_ptr_1.get(), desc.scratch_ptr_1.get(), desc.scratch_ptr_1.get() + imag_offset,
            desc.scratch_ptr_1.get() + imag_offset, twiddles_ptr, factors_and_scan, intermediate_twiddles_offset,
            impl_twiddle_offset, 0, committed_size, static_cast<Idx>(max_batches_in_l2),
            static_cast<IdxGlobal>(num_batches), static_cast<IdxGlobal>(i), static_cast<Idx>(factor_num),
            dimension_data.num_factors, storage, l2_events, desc.queue);
        intermediate_twiddles_offset += 2 * current_kernel.batch_size * static_cast<IdxGlobal>(current_kernel.length);
        impl_twiddle_offset +=
            detail::increment_twiddle_offset(current_kernel.level, static_cast<Idx>(current_kernel.length));
        detail::dump_device(desc.queue, "after factor:", desc.scratch_ptr_1.get(),
                            desc.params.number_of_transforms * dimension_data.length * 2, l2_events);
      }
      event = desc.queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(l2_events);
        cgh.host_task([&]() {});
      });
      for (Idx num_transpose = num_transposes - 1; num_transpose > 0; num_transpose--) {
        PORTFFT_LOG_TRACE("Dispatching the transpose kernel", num_transpose);
        event = detail::transpose_level<Scalar, Domain>(
            dimension_data.transpose_kernels.at(static_cast<std::size_t>(num_transpose)), desc.scratch_ptr_1.get(),
            desc.scratch_ptr_2.get(), factors_and_scan, committed_size, static_cast<Idx>(max_batches_in_l2),
            n_transforms, static_cast<IdxGlobal>(i), num_factors, 0, desc.queue, {event}, storage);
        if (storage == complex_storage::SPLIT_COMPLEX) {
          event = detail::transpose_level<Scalar, Domain>(
              dimension_data.transpose_kernels.at(static_cast<std::size_t>(num_transpose)),
              desc.scratch_ptr_1.get() + imag_offset, desc.scratch_ptr_2.get() + imag_offset, factors_and_scan,
              committed_size, static_cast<Idx>(max_batches_in_l2), n_transforms, static_cast<IdxGlobal>(i), num_factors,
              0, desc.queue, {event}, storage);
        }
        desc.scratch_ptr_1.swap(desc.scratch_ptr_2);
      }
      PORTFFT_LOG_TRACE("Dispatching the transpose kernel 0");
      event = detail::transpose_level<Scalar, Domain>(
          dimension_data.transpose_kernels.at(0), desc.scratch_ptr_1.get(), out, factors_and_scan, committed_size,
          static_cast<Idx>(max_batches_in_l2), n_transforms, static_cast<IdxGlobal>(i), num_factors,
          vec_size * static_cast<IdxGlobal>(i) * committed_size + output_offset, desc.queue, {event}, storage);
      if (storage == complex_storage::SPLIT_COMPLEX) {
        event = detail::transpose_level<Scalar, Domain>(
            dimension_data.transpose_kernels.at(0), desc.scratch_ptr_1.get() + imag_offset, out_imag, factors_and_scan,
            committed_size, static_cast<Idx>(max_batches_in_l2), n_transforms, static_cast<IdxGlobal>(i), num_factors,
            vec_size * static_cast<IdxGlobal>(i) * committed_size + output_offset, desc.queue, {event}, storage);
      }
    }
    return event;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor_impl<Scalar, Domain>::num_transforms_in_local_mem_struct::inner<detail::level::GLOBAL,
                                                                                            Dummy> {
  static Idx execute(committed_descriptor_impl&, Idx, layout, Idx, const std::vector<Idx>&) {
    PORTFFT_LOG_FUNCTION_ENTRY();
    return 1;
  }
};

}  // namespace detail
}  // namespace portfft

#endif
