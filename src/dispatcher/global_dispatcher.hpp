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

#include <common/subgroup.hpp>
#include <defines.hpp>
#include <descriptor.hpp>
#include <enums.hpp>
#include <specialization_constant.hpp>

#include <sycl/sycl.hpp>

#include <cstring>

namespace portfft {
namespace detail {
std::pair<sycl::range<1>, sycl::range<1>> get_launch_params(IdxGlobal fft_size, IdxGlobal num_batches,
                                                            detail::level level, Idx n_compute_units,
                                                            Idx subgroup_size) {
  IdxGlobal n_available_sgs = static_cast<IdxGlobal>(8 * n_compute_units * 64);
  if (level == detail::level::WORKITEM) {
    IdxGlobal n_ffts_per_wg = static_cast<IdxGlobal>(PORTFFT_SGS_IN_WG * subgroup_size);
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(sycl::range<1>(std::min(n_wgs_required * PORTFFT_SGS_IN_WG, n_available_sgs)),
                          sycl::range<1>(PORTFFT_SGS_IN_WG * subgroup_size));
  }
  if (level == detail::level::SUBGROUP) {
    IdxGlobal n_ffts_per_sg = static_cast<IdxGlobal>(subgroup_size) / detail::factorize_sg(fft_size, subgroup_size);
    IdxGlobal n_ffts_per_wg = n_ffts_per_sg * PORTFFT_SGS_IN_WG * subgroup_size;
    IdxGlobal n_wgs_required = divide_ceil(num_batches, n_ffts_per_wg);
    return std::make_pair(sycl::range<1>(std::min(n_wgs_required * PORTFFT_SGS_IN_WG, n_available_sgs)),
                          sycl::range<1>(PORTFFT_SGS_IN_WG * subgroup_size));
  }
  if (level == detail::level::WORKGROUP) {
    return std::make_pair(sycl::range<1>(std::min(num_batches * PORTFFT_SGS_IN_WG, n_available_sgs)),
                          sycl::range<1>(PORTFFT_SGS_IN_WG * subgroup_size));
  }
}

/**
 * Transposes A into B
 * @param a Input pointer a
 * @param b Input pointer b
 * @param lda leading dimension A
 * @param ldb leading Dimension B
 * @param num_elements Num elements
 */
template <typename T>
void complex_transpose(T* a, T* b, int lda, int ldb, int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    int j = i / ldb;
    int k = i % ldb;
    b[2 * i] = a[2 * k * lda + 2 * j];
    b[2 * i + 1] = a[2 * k * lda + 2 * j + 1];
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor& desc, kernel_data_struct& /*kernel_data*/) {
    std::vector<Idx> factors;
    // Get factor sizes per level;
    for (const auto& kernel_data : desc.dimensions.back().kernels) {
      factors.push_back(
          std::accumulate(kernel_data.factors.begin(), kernel_data.factors.end(), 1, std::multiplies<Idx>()));
    }
    std::vector<IdxGlobal> factors_idxGlobal;
    std::vector<IdxGlobal> sub_batches;
    std::copy(factors.begin(), factors.end(), factors_idxGlobal.begin());
    // Get sub batches
    for (std::size_t i = 0; i < factors.size() - 1; i++) {
      sub_batches.push_back(std::accumulate(factors_idxGlobal.begin() + static_cast<long>(i + 1),
                                            factors_idxGlobal.end(), IdxGlobal(1), std::multiplies<IdxGlobal>()));
    }
    sub_batches.push_back(factors_idxGlobal.at(factors.size() - 2));

    // calculate total memory required for twiddles;
    IdxGlobal mem_required_for_twiddles = 0;
    // First calculate mem required for twiddles between factors;
    for (std::size_t i = 0; i < factors.size() - 1; i++) {
      mem_required_for_twiddles += 2 * factors_idxGlobal.at(i) * sub_batches.at(i);
    }
    // Now calculate mem required for twiddles per implementation
    std::size_t counter = 0;
    for (const auto& kernel_data : desc.dimensions.back().kernels) {
      if (kernel_data.level == detail::level::SUBGROUP) {
        mem_required_for_twiddles += 2 * factors_idxGlobal.at(counter);
      } else if (kernel_data.level == detail::level::WORKGROUP) {
        IdxGlobal factor_1 = detail::factorize(factors_idxGlobal.at(counter));
        IdxGlobal factor_2 = factors_idxGlobal.at(counter) / factor_1;
        mem_required_for_twiddles += 2 * (factor_1 + factor_2);
      }
      counter++;
    }
    Scalar* host_memory = (Scalar*)malloc(static_cast<std::size_t>(mem_required_for_twiddles) * sizeof(Scalar));
    Scalar* scratch_space = (Scalar*)malloc(static_cast<std::size_t>(mem_required_for_twiddles) * sizeof(Scalar));
    Scalar* device_twiddles = sycl::malloc_device<Scalar>(mem_required_for_twiddles, desc.queue);

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
    for (std::size_t i = 0; i < factors.size(); i++) {
      calculate_twiddles(sub_batches.at(i), factors_idxGlobal.at(i), offset, host_memory);
    }
    // Now calculate per twiddles.
    counter = 0;
    for (const auto& kernel_data : desc.dimensions.back().kernels) {
      if (kernel_data.level == detail::level::SUBGROUP) {
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(0)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(1)), offset, host_memory);
      } else if (kernel_data.level == detail::level::WORKGROUP) {
        Idx factor_n = kernel_data.factors.at(0) * kernel_data.factors.at(1);
        Idx factor_m = kernel_data.factors.at(2) * kernel_data.factors.at(3);
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(0)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(1)), offset, host_memory);
        calculate_twiddles(static_cast<IdxGlobal>(kernel_data.factors.at(2)),
                           static_cast<IdxGlobal>(kernel_data.factors.at(3)), offset, host_memory);
        // Calculate wg twiddles and transpose them
        calculate_twiddles(static_cast<IdxGlobal>(factor_n), static_cast<IdxGlobal>(factor_m), offset, host_memory);
        for (Idx j = 0; j < factor_n; j++) {
          detail::complex_transpose(host_memory + offset + 2 * j * factor_n, scratch_space, factor_m, factor_n,
                                    factor_n * factor_m);
        }
      }
      counter++;
    }

    // Rearrage the twiddles between factors for optimal access patters in shared memory
    // Also take this opportunity to populate local memory size, and batch size, and launch params and local memory
    // usage Note, global impl only uses store modifiers
    counter = 0;
    for (auto& kernel_data : desc.dimensions.back().kernels) {
      IdxGlobal host_ptr_base_offset = 0;
      for (std::size_t i = 0; i < counter; i++) {
        host_ptr_base_offset += 2 * sub_batches.at(i) * factors_idxGlobal.at(counter);
      }
      kernel_data.batch_size = sub_batches.at(counter);
      if (kernel_data.level == detail::level::WORKITEM) {
        // See comments in workitem_dispatcher for layout requirments.
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idxGlobal.at(counter), sub_batches.at(counter), detail::level::WORKITEM,
                                      desc.n_compute_units, kernel_data.used_sg_size);
        kernel_data.global_range = global_range[0];
        kernel_data.local_range = local_range[0];
        if (counter < desc.dimensions.back().kernels.size() - 1) {
          IdxGlobal n_batches_per_wg = local_range[0];
          for (IdxGlobal j = 0; j < sub_batches.at(counter); j += n_batches_per_wg) {
            auto host_ptr_offset = host_ptr_base_offset + 2 * j * factors_idxGlobal.at(counter);
            IdxGlobal n_batches_to_transpose = [&]() {
              if ((j + local_range[0]) < sub_batches.at(counter))
                return static_cast<IdxGlobal>(local_range[0]);
              else
                return sub_batches.at(counter) - j;
            }();
            detail::complex_transpose(host_memory + host_ptr_offset, scratch_space, factors_idxGlobal.at(counter),
                                      n_batches_to_transpose, factors_idxGlobal.at(counter) * n_batches_to_transpose);
            std::memcpy(host_memory + host_ptr_offset, scratch_space,
                        2 * factors_idxGlobal.at(counter) * n_batches_to_transpose * sizeof(Scalar));
          }
          kernel_data.local_mem_required = 1;
        } else {
          kernel_data.local_mem_required = 2 * local_range[0] * factors_idxGlobal.at(counter);
        }
      } else if (kernel_data.level == detail::level::SUBGROUP) {
        // See comments in subgroup_dispatcher for layout requirements.
        auto [global_range, local_range] =
            detail::get_launch_params(factors_idxGlobal.at(counter), sub_batches.at(counter), detail::level::SUBGROUP,
                                      desc.n_compute_units, kernel_data.used_sg_size);
        kernel_data.global_range = global_range[0];
        kernel_data.local_range = local_range[0];
        IdxGlobal factor_sg = detail::factorize_sg(factors_idxGlobal.at(counter), kernel_data.used_sg_size);
        IdxGlobal factor_wi = factors_idxGlobal.at(counter) / factor_sg;
        Idx tmp;
        if (counter < desc.dimensions.back().kernels.size() - 1) {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem<detail::layout::BATCH_INTERLEAVED>(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idxGlobal.at(counter)),
              kernel_data.used_sg_size, {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, tmp);
        } else {
          kernel_data.local_mem_required = desc.num_scalars_in_local_mem<detail::layout::PACKED>(
              detail::level::SUBGROUP, static_cast<std::size_t>(factors_idxGlobal.at(counter)),
              kernel_data.used_sg_size, {static_cast<Idx>(factor_sg), static_cast<Idx>(factor_wi)}, tmp);
        }
      }
      counter++;
    }
    desc.queue.copy(host_memory, device_twiddles, mem_required_for_twiddles).wait();
    free(host_memory);
    free(scratch_space);
    return device_twiddles;
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::GLOBAL, Dummy> {
  static void execute(committed_descriptor& /*desc*/, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle,
                      std::size_t /*length*/, const std::vector<Idx>& factors,
                      detail::elementwise_multiply multiply_on_load, detail::elementwise_multiply multiply_on_store,
                      detail::apply_scale_factor scale_factor_applied, detail::level level) {
    in_bundle.template set_specialization_constant<detail::GlobalSubImplSpecConst>(level);
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnLoad>(multiply_on_load);
    in_bundle.template set_specialization_constant<detail::SpecConstMultiplyOnStore>(multiply_on_store);
    in_bundle.template set_specialization_constant<detail::SpecConstApplyScaleFactor>(scale_factor_applied);

    if (level == detail::level::WORKITEM) {
      in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(factors.at(0));
    } else if (level == detail::level::SUBGROUP) {
      in_bundle.template set_specialization_constant<detail::SubgroupFactorWISpecConst>(factors[0]);
      in_bundle.template set_specialization_constant<detail::SubgroupFactorSGSpecConst>(factors[1]);
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
template <direction Dir, detail::layout LayoutIn, detail::layout LayoutOut, Idx SubgroupSize, typename TIn,
          typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, LayoutIn, LayoutOut, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out,
                             const std::vector<sycl::event>& dependencies, IdxGlobal n_transforms,
                             IdxGlobal input_offset, IdxGlobal output_offset, Scalar scale_factor,
                             std::vector<kernel_data_struct>& kernel_data) {}
};

}  // namespace portfft

#endif