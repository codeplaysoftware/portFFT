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
 *
 **************************************************************************/

#ifndef PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/global.hpp>
#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <enums.hpp>
#include <utils.hpp>

namespace portfft {
namespace detail {

/**
 * prepare launch configuration for different levels
 * @param Level level of the FFT for which launch config needs to be prepared
 * @param fft_size factor length
 * @param n_transforms batch size corresposing to the factor
 * @param n_compute_units Number of SMs/EU available in the device
 * @param subgroup_size Subgroup Size the kernel will be using
 * @return std::pair containing range<1> global and local sizes
 */
std::pair<sycl::range<1>, sycl::range<1>> inline get_launch_configuration(level Level, std::size_t fft_size,
                                                                          std::size_t n_transforms,
                                                                          std::size_t n_compute_units,
                                                                          std::size_t subgroup_size) {
  // ensure maximum parallelism per batch, do not allocate more resources than required to acheive as many running (not
  // just scheduled) kernels. Ideally the number of batches processed concurrently also depends on launch params (and
  // not just L2 size and hardware limitations) to avoid scheduling stalls per level. For now, this is a TODO, as well
  // tuning of these params
  std::size_t max_concurrent_subgroups = 64 * 8 * n_compute_units;  // Just a heuristic, not true for all hardware

  if (Level == level::WORKITEM) {
    std::size_t num_wgs_required =
        std::min(max_concurrent_subgroups, detail::divide_ceil(n_transforms, subgroup_size * PORTFFT_SGS_IN_WG));
    return std::pair(sycl::range<1>(num_wgs_required * subgroup_size * PORTFFT_SGS_IN_WG),
                     sycl::range<1>(subgroup_size * PORTFFT_SGS_IN_WG));
  }
  if (Level == level::SUBGROUP) {
    std::size_t factor_sg =
        static_cast<std::size_t>(factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size)));
    std::size_t num_batches_per_sg = subgroup_size / factor_sg;
    std::size_t num_wgs_required =
        std::min(max_concurrent_subgroups, detail::divide_ceil(n_transforms, num_batches_per_sg * PORTFFT_SGS_IN_WG));
    return std::pair(sycl::range<1>(num_wgs_required * subgroup_size * PORTFFT_SGS_IN_WG),
                     sycl::range<1>(subgroup_size * PORTFFT_SGS_IN_WG));
  }
  if (Level == level::WORKGROUP) {
    std::size_t wg_size = subgroup_size * PORTFFT_SGS_IN_WG;
    std::size_t num_wgs_required = detail::divide_ceil(n_transforms, wg_size);
    return std::pair(
        sycl::range<1>(std::min(max_concurrent_subgroups, num_wgs_required * PORTFFT_SGS_IN_WG) * subgroup_size),
        sycl::range<1>(subgroup_size * PORTFFT_SGS_IN_WG));
  }
  throw std::logic_error("Invalid Level");
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    auto calc_total_mem_for_twiddles = [=]() -> std::size_t {
      std::size_t num_scalars = 0;
      std::size_t index = 0;
      for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
        std::size_t num_batches = std::accumulate(desc.factors.begin() + static_cast<long>(i + 1), desc.factors.end(),
                                                  static_cast<std::size_t>(1), std::multiplies<std::size_t>());
        num_scalars += num_batches * desc.factors[i];
      }
      for (detail::level level_name : desc.levels) {
        if (level_name == detail::level::WORKITEM) {
        }
        if (level_name == detail::level::SUBGROUP) {
          num_scalars += desc.factors[index];
        }
        if (level_name == detail::level::WORKGROUP) {
          num_scalars += desc.factors[index];
          auto n = detail::factorize(desc.factors[index]);
          num_scalars += n + desc.factors[index] / n;
        }
        index++;
      }
      return 2 * num_scalars;
    };

    auto calculate_twiddles = [](std::size_t N, std::size_t M, std::size_t& offset, Scalar* ptr) {
      for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
          double thetha = (-2 * M_PI * static_cast<double>(i * j) / static_cast<double>(N * M));
          ptr[offset++] = static_cast<Scalar>(std::cos(thetha));
          ptr[offset++] = static_cast<Scalar>(std::sin(thetha));
        }
      }
    };

    auto memory_for_twiddles = calc_total_mem_for_twiddles();
    Scalar* host_twiddles_ptr = sycl::malloc_host<Scalar>(memory_for_twiddles, desc.queue);
    Scalar* device_twiddles_ptr = sycl::malloc_device<Scalar>(memory_for_twiddles, desc.queue);

    // first calculate all for Intermediate twiddles
    //  Then iff level is subgroup, calculate twiddles required for subgroup.
    std::size_t offset = 0;
    std::size_t index = 0;
    for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
      std::size_t n = desc.factors[i];
      std::size_t m = std::accumulate(desc.factors.begin() + static_cast<long>(i + 1), desc.factors.end(),
                                      static_cast<std::size_t>(1), std::multiplies<std::size_t>());
      // store twiddles for global memory in a transposed fashion to ensure coalesced accesses.
      calculate_twiddles(m, n, offset, host_twiddles_ptr);
    }

    for (detail::level level_name : desc.levels) {
      // TODO: Refactor this and dispatch to correct execute specialization
      if (level_name == detail::level::WORKITEM) {
      }
      if (level_name == detail::level::SUBGROUP) {
        std::size_t factor = desc.factors[index];
        auto n = static_cast<std::size_t>(detail::factorize_sg(static_cast<int>(factor), desc.used_sg_size));
        auto m = factor / static_cast<std::size_t>(n);
        calculate_twiddles(m, n, offset, host_twiddles_ptr);
      }
      if (level_name == detail::level::WORKGROUP) {
        std::size_t factor = desc.factors[index];
        std::size_t n = detail::factorize(factor);
        std::size_t m = factor / n;
        std::size_t n_sg = static_cast<std::size_t>(detail::factorize_sg(static_cast<int>(n), desc.used_sg_size));
        std::size_t n_wi = n / n_sg;
        std::size_t m_sg = static_cast<std::size_t>(detail::factorize_sg(static_cast<int>(m), desc.used_sg_size));
        std::size_t m_wi = m / m_sg;

        calculate_twiddles(n, m, offset, host_twiddles_ptr);
        calculate_twiddles(n_sg, n_wi, offset, host_twiddles_ptr);
        calculate_twiddles(m_sg, m_wi, offset, host_twiddles_ptr);
      }
      index++;
    }

    desc.queue.copy(host_twiddles_ptr, device_twiddles_ptr, memory_for_twiddles).wait();
    desc.queue.prefetch(device_twiddles_ptr, memory_for_twiddles * sizeof(Scalar));
    desc.queue.wait();
    return device_twiddles_ptr;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, int SubgroupSize, typename TIn, typename TOut>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, SubgroupSize, TIn,
                                                               TOut>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const TIn& in, TOut& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    auto increment_twiddle_offset = [&](std::size_t level_num, std::size_t& offset) {
      if (desc.levels[level_num] == detail::level::SUBGROUP) {
        offset += 2 * desc.factors[level_num];
      } else if (desc.levels[level_num] == detail::level::WORKGROUP) {
        offset +=
            2 *
            (desc.factors[level_num] +
             static_cast<std::size_t>(detail::factorize_sg(static_cast<int>(desc.factors[level_num]), SubgroupSize)) +
             desc.factors[level_num] / static_cast<std::size_t>(detail::factorize_sg(
                                           static_cast<int>(desc.factors[level_num]), SubgroupSize)));
      }
    };

    std::vector<sycl::event> dependency_copy(dependencies);
    num_scalars_in_local_mem_struct::template inner<detail::level::GLOBAL, TransposeIn, Dummy>::execute(desc);
    std::size_t local_mem_twiddle_offset = 0;
    const Scalar* scratch_input = static_cast<const Scalar*>(desc.scratch_1.get());
    Scalar* scratch_output = static_cast<Scalar*>(desc.scratch_2.get());
    for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
      local_mem_twiddle_offset += static_cast<std::size_t>(desc.factors[i] * desc.sub_batches[i]);
    }
    for (std::size_t batch = 0; batch < desc.params.number_of_transforms; batch += desc.num_batches_in_l2) {
      std::size_t twiddle_between_factors_offset = 0;
      std::size_t impl_twiddles_offset = local_mem_twiddle_offset;
      detail::dispatch_compute_kernels<Scalar, Dir, Domain, detail::transpose::TRANSPOSED,
                                       detail::transpose::TRANSPOSED, detail::apply_load_modifier::NOT_APPLIED,
                                       detail::apply_store_modifier::APPLIED, detail::apply_scale_factor::NOT_APPLIED,
                                       SubgroupSize>(desc, in, scale_factor, 0, impl_twiddles_offset,
                                                     twiddle_between_factors_offset, batch, dependency_copy);
      twiddle_between_factors_offset += 2 * desc.factors[0] * desc.sub_batches[0];
      increment_twiddle_offset(0, impl_twiddles_offset);
      for (std::size_t level_num = 1; level_num < desc.factors.size(); level_num++) {
        if (level_num == desc.factors.size() - 1) {
          detail::dispatch_compute_kernels<Scalar, Dir, Domain, detail::transpose::NOT_TRANSPOSED,
                                           detail::transpose::NOT_TRANSPOSED, detail::apply_load_modifier::NOT_APPLIED,
                                           detail::apply_store_modifier::NOT_APPLIED,
                                           detail::apply_scale_factor::APPLIED, SubgroupSize>(
              desc, scratch_input, scale_factor, level_num, impl_twiddles_offset, twiddle_between_factors_offset, batch,
              dependency_copy);
        } else {
          detail::dispatch_compute_kernels<Scalar, Dir, Domain, detail::transpose::TRANSPOSED,
                                           detail::transpose::TRANSPOSED, detail::apply_load_modifier::NOT_APPLIED,
                                           detail::apply_store_modifier::APPLIED,
                                           detail::apply_scale_factor::NOT_APPLIED, SubgroupSize>(
              desc, scratch_input, scale_factor, level_num, impl_twiddles_offset, twiddle_between_factors_offset, batch,
              dependency_copy);
          twiddle_between_factors_offset += 2 * desc.factors[level_num] * desc.sub_batches[level_num];
          increment_twiddle_offset(level_num, impl_twiddles_offset);
        }
      }
      for (std::size_t level_num = desc.factors.size() - 2; level_num > 0; level_num--) {
        detail::dispatch_transpose_kernels<Scalar, Domain, SubgroupSize>(desc, scratch_input, scratch_output, level_num,
                                                                         batch, dependency_copy);
      }
      detail::dispatch_transpose_kernels<Scalar, Domain, SubgroupSize>(desc, scratch_input, out, 0, batch,
                                                                       dependency_copy);
    }
    return desc.queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependency_copy);
      cgh.host_task([&]() {});
    });
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::GLOBAL,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    auto get_local_mem_usage_per_level = [](committed_descriptor<Scalar, Domain> committed_descriptor,
                                            std::size_t factor, detail::level level_id,
                                            bool transposed) -> std::size_t {
      if (level_id == detail::level::WORKITEM) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, detail::transpose::TRANSPOSED,
                                                                 Dummy, std::size_t>::execute(committed_descriptor,
                                                                                              factor);
        }
        return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM,
                                                               detail::transpose::NOT_TRANSPOSED, Dummy,
                                                               std::size_t>::execute(committed_descriptor, factor);
      }
      if (level_id == detail::level::SUBGROUP) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, detail::transpose::TRANSPOSED,
                                                                 Dummy, std::size_t>::execute(committed_descriptor,
                                                                                              factor);
        }
        return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP,
                                                               detail::transpose::NOT_TRANSPOSED, Dummy,
                                                               std::size_t>::execute(committed_descriptor, factor);
      }
      if (level_id == detail::level::WORKGROUP) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                                 detail::transpose::TRANSPOSED, Dummy,
                                                                 std::size_t>::execute(committed_descriptor, factor);
        }
        return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                               detail::transpose::NOT_TRANSPOSED, Dummy,
                                                               std::size_t>::execute(committed_descriptor, factor);
      }
      return 0;
    };
    if (desc.local_mem_per_factor.empty()) {
      bool transposed = true;
      for (std::size_t i = 0; i < desc.factors.size(); i++) {
        if (i == desc.factors.size() - 1) {
          transposed = false;
        }
        auto factor = desc.factors[i];
        detail::level level_id = desc.levels[i];
        desc.local_mem_per_factor.push_back(get_local_mem_usage_per_level(desc, factor, level_id, transposed));
      }
      std::size_t index = 0;
      desc.launch_configurations.clear();
      for (detail::level level_id : desc.levels) {
        fft_size = desc.factors[index];
        std::size_t batch_size = std::accumulate(desc.factors.begin() + static_cast<long>(index), desc.factors.end(),
                                                 static_cast<std::size_t>(1), std::multiplies<std::size_t>());
        if (index == desc.factors.size() - 1) {
          batch_size = desc.factors[index - 1];
        }
        desc.launch_configurations.push_back(detail::get_launch_configuration(
            level_id, fft_size, batch_size, desc.n_compute_units, static_cast<std::size_t>(desc.used_sg_size)));
        index++;
      }
      return 0;
    }
    return 0;
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::GLOBAL, TransposeIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::GLOBAL, TransposeIn, Dummy>::execute(
        desc, desc.params.lengths[0]);
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::GLOBAL, Dummy> {
  static void execute(committed_descriptor& desc,
                      std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>& in_bundles) {
    for (std::size_t i = 0; i < in_bundles.size(); i++) {
      detail::level level_id = desc.levels[i];
      std::size_t factor = desc.factors[i];
      auto& in_bundle = in_bundles[i];
      in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevel>(level_id);
      in_bundle.template set_specialization_constant<detail::GlobalSpecConstNumFactors>(desc.factors.size());
      in_bundle.template set_specialization_constant<detail::GlobalSpecConstLevelNum>(i);
      switch (level_id) {
        case detail::level::WORKITEM: {
          in_bundle.template set_specialization_constant<detail::GlobalSpecConstFftSize>(factor);
          break;
        }
        case detail::level::SUBGROUP: {
          int factor_sg = detail::factorize_sg(static_cast<int>(factor), static_cast<int>(desc.used_sg_size));
          int factor_wi = static_cast<int>(factor) / factor_sg;
          in_bundle.template set_specialization_constant<detail::GlobalSpecConstSGFactorWI>(factor_wi);
          in_bundle.template set_specialization_constant<detail::GlobalSpecConstSGFactorSG>(factor_sg);
          break;
        }
        case detail::level::WORKGROUP: {
          in_bundle.template set_specialization_constant<detail::GlobalSpecConstFftSize>(factor);
          break;
        }
        case detail::level::GLOBAL:
          throw std::logic_error("Invalid factor level");
      }
    }
  }
};

}  // namespace portfft

#endif  // PORTFFT_DISPATCHER_GLOBAL_DISPATCHER_HPP
