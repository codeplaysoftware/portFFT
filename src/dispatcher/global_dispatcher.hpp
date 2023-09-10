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

std::pair<sycl::range<1>, sycl::range<1>> inline get_launch_configuration(level Level, std::size_t fft_size,
                                                                          std::size_t n_transforms,
                                                                          std::size_t n_compute_units,
                                                                          std::size_t subgroup_size) {
  // ensure maximum parallelism per batch, do not allocate more resources than required to acheive as many running (not
  // just scheduled) kernels. Ideally the number of batches processed concurrently also depends on launch params (and
  // not just L2 size and hardware limitations) to avoid scheduling stalls per level. For now, this is a TODO, as well
  // tuning of these params
  std::size_t max_concurrent_subgroups = 64 * n_compute_units;  // Just a heuristic, not true for all hardware

  if (Level == level::WORKITEM) {
    std::size_t num_wgs_required = std::min(max_concurrent_subgroups, detail::divide_ceil(n_transforms, subgroup_size));
    return std::pair(sycl::range<1>(num_wgs_required * subgroup_size), sycl::range<1>(subgroup_size));
  }
  if (Level == level::SUBGROUP) {
    std::size_t factor_sg =
        static_cast<std::size_t>(factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size)));
    std::size_t num_batches_per_sg = subgroup_size / factor_sg;
    std::size_t num_wgs_required =
        std::min(max_concurrent_subgroups, detail::divide_ceil(n_transforms, num_batches_per_sg));
    return std::pair(sycl::range<1>(num_wgs_required * subgroup_size), sycl::range<1>(subgroup_size));
  }
  if (Level == level::WORKGROUP) {
    std::size_t wg_size = subgroup_size * 4;
    std::size_t num_wgs_required = detail::divide_ceil(n_transforms, wg_size);
    return std::pair(sycl::range<1>(std::min(max_concurrent_subgroups, num_wgs_required * 4) * subgroup_size),
                     sycl::range<1>(subgroup_size * 4));
  }
}

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    auto calc_total_mem_for_twiddles = [=]() -> std::size_t {
      std::size_t num_scalars = 0;
      int index = 0;
      for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
        std::size_t num_batches =
            std::accumulate(desc.factors.begin() + i + 1, desc.factors.end(), 1, std::multiplies<std::size_t>());
        num_scalars += num_batches * desc.factors[i];
      }
      for (detail::level level : desc.levels) {
        if (level == detail::level::WORKITEM) {
        }
        if (level == detail::level::SUBGROUP) {
          num_scalars += desc.factors[index];
        }
        if (level == detail::level::WORKGROUP) {
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
    int index = 0;
    for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
      std::size_t n = desc.factors[i];
      std::size_t m = std::accumulate(desc.factors.begin() + static_cast<long>(i + 1), desc.factors.end(), 1,
                                      std::multiplies<std::size_t>());
      // store twiddles for global memory in a transposed fashion to ensure coalesced accesses.
      calculate_twiddles(m, n, offset, host_twiddles_ptr);
    }

    for (detail::level level : desc.levels) {
      // TODO: Refactor this and dispatch to correct execute specialization
      switch (level) {
        if (level == detail::level::WORKITEM) {
        }
        if (level == detail::level::SUBGROUP) {
          std::size_t factor = desc.factors[index];
          auto n = detail::factorize_sg(factor, desc.used_sg_size);
          auto m = factor / n;
          calculate_twiddles(m, n, offset, host_twiddles_ptr);
        }
        if (level == detail::level::WORKGROUP) {
          std::size_t factor = desc.factors[index];
          std::size_t n = detail::factorize(factor);
          std::size_t m = factor / n;
          std::size_t n_sg = static_cast<std::size_t>(detail::factorize_sg(n, desc.used_sg_size));
          std::size_t n_wi = n / n_sg;
          std::size_t m_sg = static_cast<std::size_t>(detail::factorize_sg(m, desc.used_sg_size));
          std::size_t m_wi = m / m_sg;

          calculate_twiddles(n, m, offset, host_twiddles_ptr);
          calculate_twiddles(n_sg, n_wi, offset, host_twiddles_ptr);
          calculate_twiddles(m_sg, m_wi, offset, host_twiddles_ptr);
        }
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
    constexpr detail::memory mem = std::is_pointer<TOut>::value ? detail::memory::USM : detail::memory::BUFFER;
    num_scalars_in_local_mem_struct::template inner<detail::level::GLOBAL, TransposeIn, Dummy>::execute(desc);
    std::size_t local_mem_twiddle_offset = 0;
    for (std::size_t i = 0; i < desc.factors.size() - 1; i++) {
      local_mem_twiddle_offset += static_cast<std::size_t>(desc.factors[i] * desc.sub_batches[i]);
    }
    std::size_t fft_size = desc.params.lengths[0];
    for (int batch = 0; batch < desc.params.number_of_transforms; batch += desc.num_batches_in_l2) {
      detail::dispatch_kernel_struct<0, Dir, Scalar, Domain, mem, detail::transpose::TRANSPOSED,
                                     detail::transpose::TRANSPOSED, false, true, SubgroupSize, TIn,
                                     TOut>::execute(in, out, desc, 0, 2 * local_mem_twiddle_offset, scale_factor,
                                                    2 * fft_size * batch);
    }
    desc.queue.wait();
    sycl::event Event;
    return Event;
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::GLOBAL,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    auto get_local_mem_usage_per_level = [](committed_descriptor<Scalar, Domain> committed_descriptor,
                                            std::size_t factor, detail::level level, bool transposed) -> std::size_t {
      if (level == detail::level::WORKITEM) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM, detail::transpose::TRANSPOSED,
                                                                 Dummy, std::size_t>::execute(committed_descriptor,
                                                                                              factor);
        } else {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM,
                                                                 detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                 std::size_t>::execute(committed_descriptor, factor);
        }
      }
      if (level == detail::level::SUBGROUP) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP, detail::transpose::TRANSPOSED,
                                                                 Dummy, std::size_t>::execute(committed_descriptor,
                                                                                              factor);
        } else {
          return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP,
                                                                 detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                 std::size_t>::execute(committed_descriptor, factor);
        }
      }
      if (level == detail::level::WORKGROUP) {
        if (transposed) {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                                 detail::transpose::TRANSPOSED, Dummy,
                                                                 std::size_t>::execute(committed_descriptor, factor);
        } else {
          return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                                 detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                 std::size_t>::execute(committed_descriptor, factor);
        }
      }
    };
    if (desc.local_mem_per_factor.empty()) {
      bool transposed = true;
      for (std::size_t i = 0; i < desc.factors.size(); i++) {
        if (i == desc.factors.size() - 1) {
          transposed = false;
        }
        auto factor = desc.factors[i];
        detail::level level = desc.levels[i];
        desc.local_mem_per_factor.push_back(get_local_mem_usage_per_level(desc, factor, level, transposed));
      }
      int index = 0;
      desc.launch_configurations.clear();
      for (detail::level level : desc.levels) {
        std::size_t fft_size = desc.factors[index];
        std::size_t batch_size =
            std::accumulate(desc.factors.begin() + index, desc.factors.end(), 1, std::multiplies<std::size_t>());
        if (index == desc.factors.size() - 1) {
          batch_size = desc.factors[index - 1];
        }
        desc.launch_configurations.push_back(
            detail::get_launch_configuration(level, fft_size, batch_size, desc.n_compute_units, desc.used_sg_size));
        index++;
      }
      return 0;
    } else {
      return 0;
    }
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
      detail::level level = desc.levels[i];
      std::size_t factor = desc.factors[i];
      auto& in_bundle = in_bundles[i];
      in_bundle.template set_specialization_constant<detail::SpecConstLevel>(level);
      switch (level) {
        case detail::level::WORKITEM: {
          in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(factor);
          break;
        }
        case detail::level::SUBGROUP: {
          int factor_sg = detail::factorize_sg(static_cast<int>(factor), static_cast<int>(desc.used_sg_size));
          int factor_wi = static_cast<int>(factor) / factor_sg;
          in_bundle.template set_specialization_constant<detail::SpecConstSGFactorWI>(factor_wi);
          in_bundle.template set_specialization_constant<detail::SpecConstSGFactorSG>(factor_sg);
          break;
        }
        case detail::level::WORKGROUP: {
          in_bundle.template set_specialization_constant<detail::SpecConstFftSize>(factor);
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