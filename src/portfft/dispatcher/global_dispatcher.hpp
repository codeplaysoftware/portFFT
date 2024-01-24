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
 * @param a Input pointer a
 * @param b Input pointer b
 * @param lda leading dimension A
 * @param ldb leading Dimension B
 * @param num_elements Total number of complex values in the matrix
 */
template <typename T>
void complex_transpose(const T* a, T* b, IdxGlobal lda, IdxGlobal ldb, IdxGlobal num_elements) {
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

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor& desc, dimension_struct& dimension_data) {
    auto& kernels = dimension_data.kernels;
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
      Idx used_sg_size = detail::sg_size(kernel_data.used_ct_profile);
      if (kernel_data.level == detail::level::WORKITEM) {
        // See comments in workitem_dispatcher for layout requirments.
        Idx num_sgs_in_wg = desc.rt_configuration.sgs_per_wg;
        if (counter < kernels.size() - 1) {
          kernel_data.local_mem_required = static_cast<std::size_t>(1);
        } else {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem<detail::layout::PACKED>(
              detail::level::WORKITEM, static_cast<std::size_t>(factors_idx_global.at(counter)), used_sg_size,
              {static_cast<Idx>(factors_idx_global.at(counter))}, num_sgs_in_wg);
        }
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idx_global.at(counter), sub_batches.at(counter), detail::level::WORKITEM,
                                      desc.dev_info.n_compute_units, used_sg_size, num_sgs_in_wg);
        kernel_data.global_range = global_range;
        kernel_data.local_range = local_range;
      } else if (kernel_data.level == detail::level::SUBGROUP) {
        Idx num_sgs_in_wg = desc.rt_configuration.sgs_per_wg;
        // See comments in subgroup_dispatcher for layout requirements.
        IdxGlobal factor_sg = detail::factorize_sg(factors_idx_global.at(counter), used_sg_size);
        IdxGlobal factor_wi = factors_idx_global.at(counter) / factor_sg;
        if (counter < kernels.size() - 1) {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idx_global.at(counter)), used_sg_size,
              {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, num_sgs_in_wg);
        } else {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem<detail::layout::PACKED>(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idx_global.at(counter)), used_sg_size,
              {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, num_sgs_in_wg);
        }
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idx_global.at(counter), sub_batches.at(counter), detail::level::SUBGROUP,
                                      desc.dev_info.n_compute_units, used_sg_size, num_sgs_in_wg);
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
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::GLOBAL, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t length, const std::vector<Idx>& factors, detail::level level, Idx factor_num,
                      Idx num_factors) {
    Idx length_idx = static_cast<Idx>(length);
    in_bundle.template set_specialization_constant<detail::GlobalSubImplSpecConst>(level);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(num_factors);
    in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(factor_num);
    if (level == detail::level::WORKITEM || level == detail::level::WORKGROUP) {
      in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(length_idx);
    } else if (level == detail::level::SUBGROUP) {
      in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[1]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[0]);
    }
  }
};

template <typename Scalar, domain Domain>
template <detail::layout LayoutIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::GLOBAL, LayoutIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& /*desc*/, std::size_t /*length*/, Idx /*used_sg_size*/,
                             const std::vector<Idx>& /*factors*/, Idx& /*num_sgs_per_wg*/) {
    // No work required as all work done in calculate_twiddles;
    return 0;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, detail::ct_profile Config, typename TIn,
          typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, LayoutOut, Config, TIn,
                                                               TOut>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, const TIn& in_imag, TOut& out_imag,
                             const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, Scalar scale_factor,
                             dimension_struct& dimension_data) {
    (void)in_imag;
    (void)out_imag;
    const auto& kernels = dimension_data.kernels;
    const Scalar* twiddles_ptr = static_cast<const Scalar*>(kernels.at(0).twiddles_forward.get());
    const IdxGlobal* factors_and_scan = static_cast<const IdxGlobal*>(dimension_data.factors_and_scan.get());
    std::size_t num_batches = desc.params.number_of_transforms;
    std::size_t max_batches_in_l2 = static_cast<std::size_t>(dimension_data.num_batches_in_l2);
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
      IdxGlobal intermediate_twiddles_offset = 0;
      IdxGlobal impl_twiddle_offset = initial_impl_twiddle_offset;
      l2_events = detail::compute_level<Scalar, Domain, Dir, detail::layout::BATCH_INTERLEAVED,
                                        detail::layout::BATCH_INTERLEAVED, Config>(
          kernels.at(0), in, desc.scratch_ptr_1.get(), twiddles_ptr, factors_and_scan, scale_factor,
          intermediate_twiddles_offset, impl_twiddle_offset,
          2 * static_cast<IdxGlobal>(i) * committed_size + input_offset, committed_size,
          static_cast<Idx>(max_batches_in_l2), static_cast<IdxGlobal>(num_batches), static_cast<IdxGlobal>(i), 0,
          dimension_data.num_factors, {event}, desc.queue);
      intermediate_twiddles_offset += 2 * kernels.at(0).batch_size * static_cast<IdxGlobal>(kernels.at(0).length);
      impl_twiddle_offset +=
          detail::increment_twiddle_offset(kernels.at(0).level, static_cast<Idx>(kernels.at(0).length));
      for (std::size_t factor_num = 1; factor_num < static_cast<std::size_t>(dimension_data.num_factors);
           factor_num++) {
        if (static_cast<Idx>(factor_num) == dimension_data.num_factors - 1) {
          l2_events =
              detail::compute_level<Scalar, Domain, Dir, detail::layout::PACKED, detail::layout::PACKED, Config>(
                  kernels.at(factor_num), static_cast<const Scalar*>(desc.scratch_ptr_1.get()),
                  desc.scratch_ptr_1.get(), twiddles_ptr, factors_and_scan, scale_factor, intermediate_twiddles_offset,
                  impl_twiddle_offset, 0, committed_size, static_cast<Idx>(max_batches_in_l2),
                  static_cast<IdxGlobal>(num_batches), static_cast<IdxGlobal>(i), static_cast<Idx>(factor_num),
                  dimension_data.num_factors, l2_events, desc.queue);
        } else {
          l2_events = detail::compute_level<Scalar, Domain, Dir, detail::layout::BATCH_INTERLEAVED,
                                            detail::layout::BATCH_INTERLEAVED, Config>(
              kernels.at(factor_num), static_cast<const Scalar*>(desc.scratch_ptr_1.get()), desc.scratch_ptr_1.get(),
              twiddles_ptr, factors_and_scan, scale_factor, intermediate_twiddles_offset, impl_twiddle_offset, 0,
              committed_size, static_cast<Idx>(max_batches_in_l2), static_cast<IdxGlobal>(num_batches),
              static_cast<IdxGlobal>(i), static_cast<Idx>(factor_num), dimension_data.num_factors, l2_events,
              desc.queue);
          intermediate_twiddles_offset +=
              2 * kernels.at(factor_num).batch_size * static_cast<IdxGlobal>(kernels.at(factor_num).length);
          impl_twiddle_offset += detail::increment_twiddle_offset(kernels.at(factor_num).level,
                                                                  static_cast<Idx>(kernels.at(factor_num).length));
        }
      }
      event = desc.queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(l2_events);
        cgh.host_task([&]() {});
      });
      for (Idx num_transpose = num_transposes - 1; num_transpose > 0; num_transpose--) {
        event = detail::transpose_level<Scalar, Domain>(
            kernels.at(static_cast<std::size_t>(num_transpose) + static_cast<std::size_t>(num_factors)),
            static_cast<const Scalar*>(desc.scratch_ptr_1.get()), desc.scratch_ptr_2.get(), factors_and_scan,
            committed_size, static_cast<Idx>(max_batches_in_l2), n_transforms, static_cast<IdxGlobal>(i), num_transpose,
            num_factors, 0, desc.queue, desc.scratch_ptr_1, desc.scratch_ptr_2, {event});
        event.wait();
      }
      event = detail::transpose_level<Scalar, Domain>(
          kernels.at(static_cast<std::size_t>(num_factors)), static_cast<const Scalar*>(desc.scratch_ptr_1.get()), out,
          factors_and_scan, committed_size, static_cast<Idx>(max_batches_in_l2), n_transforms,
          static_cast<IdxGlobal>(i), 0, num_factors, 2 * static_cast<IdxGlobal>(i) * committed_size + output_offset,
          desc.queue, desc.scratch_ptr_1, desc.scratch_ptr_2, {event});
    }
    return event;
  }
};

}  // namespace portfft

#endif
