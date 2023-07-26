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

#ifndef PORTFFT_DISPATCHER_DEVICE_DISPATCHER_HPP
#define PORTFFT_DISPATCHER_DEVICE_DISPATCHER_HPP

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <common/workitem.hpp>
#include <descriptor.hpp>
#include <enums.hpp>
#include <mutex>

#include <sycl/sycl.hpp>

namespace portfft {
namespace detail {

std::pair<sycl::range<1>, sycl::range<1>> get_launch_configuration(level Level, std::size_t fft_size,
                                                                   std::size_t n_transforms,
                                                                   std::size_t n_compute_units,
                                                                   std::size_t subgroup_size) {
  // ensure maximum parallelism per batch, do not allocate more resources than required to acheive as many running (not
  // just scheduled) kernels. Ideally the number of batches processed concurrently also depends on launch params (and
  // not just L2 size and hardware limitations) to avoid scheduling stalls per level. For now, this is a TODO, as well
  // tuning of these params
  std::size_t max_concurrent_subgroups = 64 * n_compute_units;  // Just a heuristic, not true for all hardware

  switch (Level) {
    case level::WORKITEM: {
      std::size_t num_wgs_required = std::min(max_concurrent_subgroups, divideCeil(n_transforms, subgroup_size));
      return std::pair(sycl::range<1>(num_wgs_required * subgroup_size), sycl::range<1>(subgroup_size));
    } break;
    case level::SUBGROUP: {
      std::size_t factor_sg = factorize_sg(fft_size, static_cast<int>(subgroup_size));
      std::size_t num_batches_per_sg = subgroup_size / factor_sg;
      std::size_t num_wgs_required = std::min(max_concurrent_subgroups, divideCeil(n_transforms, num_batches_per_sg));
      return std::pair(sycl::range<1>(num_wgs_required * subgroup_size), sycl::range<1>(subgroup_size));
    } break;
    case level::WORKGROUP: {
      std::size_t wg_size = subgroup_size * 4;
      std::size_t num_wgs_required = divideCeil(n_transforms, wg_size);
      return std::pair(sycl::range<1>(std::min(max_concurrent_subgroups, num_wgs_required * 4) * subgroup_size),
                       sycl::range<1>(subgroup_size * 4));
    } break;

    default:
      break;
  }
}
}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::DEVICE, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    // first calculate space for Intermediate twiddles
    //  Then iff level is subgroup, calculate twiddles required for subgroup.
    auto calc_total_mem_for_twiddles = [=]() -> std::size_t {
      std::size_t num_scalars = 0;
      int index = 0;
      for (auto iter = desc.factors.begin(); iter + 1 != desc.factors.end(); iter++) {
        num_scalars += *iter * std::accumulate(iter + 1, desc.factors.end(), 1, std::multiplies<std::size_t>());
      }
      for (detail::level Level : desc.levels) {
        switch (Level) {
          case detail::level::WORKITEM:
            break;
          case detail::level::SUBGROUP:
            num_scalars += desc.factors[index];
            break;
          case detail::level::WORKGROUP:
            num_scalars += desc.factors[index];
            auto N = detail::factorize(desc.factors[index]);
            num_scalars += N + desc.factors[index] / N;
            break;
        }
        index++;
      }
      return 2 * num_scalars * sizeof(Scalar);
    };

    auto calculate_twiddles = [](std::size_t N, std::size_t M, std::size_t& offset, Scalar* ptr) {
      for (std::size_t i = 0; i < N; i++) {
        for (std::size_t j = 0; j < M; j++) {
          ptr[offset++] = static_cast<Scalar>(std::cos((-2 * M_PI * i * j) / (N * M)));
          ptr[offset++] = static_cast<Scalar>(std::sin((-2 * M_PI * i * j) / (N * M)));
        }
      }
    };

    // Debatable use of pinned memory
    auto memory_for_twiddles = calc_total_mem_for_twiddles();
    Scalar* host_twiddles_ptr = sycl::malloc_host<Scalar>(memory_for_twiddles, desc.queue);
    Scalar* device_twiddles_ptr = sycl::malloc_device<Scalar>(memory_for_twiddles, desc.queue);

    // first calculate all for Intermediate twiddles
    //  Then iff level is subgroup, calculate twiddles required for subgroup.
    std::size_t offset = 0;
    int index = 0;
    for (auto iter = desc.factors.begin(); iter + 1 != desc.factors.end(); iter++) {
      std::size_t N = *iter;
      std::size_t M = std::accumulate(iter + 1, desc.factors.end(), 1, std::multiplies<std::size_t>());
      // store twiddles for global memory in a transposed fashion to ensure coalesced accesses.
      calculate_twiddles(M, N, offset, host_twiddles_ptr);
    }

    for (detail::level Level : desc.levels) {
      // TODO: Refactor this and dispatch to correct execute specialization
      switch (Level) {
        case detail::level::WORKITEM:
          /* code */
          break;

        case detail::level::SUBGROUP: {
          std::size_t factor = desc.factors[index];
          auto N = detail::factorize_sg(factor, desc.used_sg_size);
          auto M = factor / N;
          calculate_twiddles(N, M, offset, host_twiddles_ptr);
        } break;

        case detail::level::WORKGROUP: {
          std::size_t factor = desc.factors[index];
          std::size_t N = detail::factorize(factor);
          std::size_t M = factor / N;
          std::size_t N_sg = detail::factorize_sg(N, desc.used_sg_size);
          std::size_t N_wi = N / N_sg;
          std::size_t M_sg = detail::factorize_sg(M, desc.used_sg_size);
          std::size_t M_wi = M / M_sg;

          calculate_twiddles(N, M, offset, host_twiddles_ptr);
          calculate_twiddles(N_sg, N_wi, offset, host_twiddles_ptr);
          calculate_twiddles(M_sg, M_wi, offset, host_twiddles_ptr);
          break;
        }
      }
      index++;
    }

    desc.queue.memcpy(host_twiddles_ptr, device_twiddles_ptr, offset).wait();
    sycl::free(host_twiddles_ptr, desc.queue);
    return device_twiddles_ptr;
  }
};

template <typename Scalar, domain Domain>
template <direction Dir, detail::transpose TransposeIn, detail::transpose TransposeOut, int SubgroupSize,
          bool ApplyLoadCallback, bool ApplyStoreCallback, typename T_in, typename T_out>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::run_kernel_struct<Dir, TransposeIn, TransposeOut, SubgroupSize,
                                                               ApplyLoadCallback, ApplyStoreCallback, T_in,
                                                               T_out>::inner<detail::level::DEVICE, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;

    int index = 0;
    sycl::event event;
    bool last_kernel = false;
    num_scalars_in_local_mem_struct::template inner<detail::level::DEVICE, TransposeIn, Dummy>::execute(desc);
    desc.queue.copy(desc.scratch.get(), in, desc.params.lengths[0] * desc.params.number_of_transforms);
    std::size_t local_mem_twiddles_initial_offset = 0;
    for (auto iter = desc.factors.begin(); iter + 1 != desc.factors.end(); iter++) {
      local_mem_twiddles_initial_offset +=
          *iter * std::accumulate(iter + 1, desc.factors.end(), 1, std::multiplies<std::size_t>());
    }
    for (int i = 0; i < desc.params.number_of_transforms; i += desc.num_batches_in_l2) {
      for (std::size_t level_num = 0; level_num < desc.levels.size(); level_num++) {
        std::size_t iter_offset = 1;
        std::size_t local_mem_twiddles_offset = local_mem_twiddles_initial_offset;
        std::size_t interemediate_twiddles_offset = 0;
        std::size_t local_mem_usage_index = 0;
        std::size_t batch_size;
        if (i == desc.levels.size() - 1) {
          batch_size = desc.factors[0];
          last_kernel = true;
        } else {
          std::size_t batch_size = std::accumulate(desc.factors.begin() + level_num + 1, desc.factors.end(), 1,
                                                   std::multiplies<std::size_t>());
        }
        std::size_t factor = desc.factors[level_num];
        detail::level Level = desc.levels[level_num];
        for (int batch_num = i; batch_num < desc.num_batches_in_l2; batch_num++) {
          // TODO:: A better solution to this will be to use a virtual function table, or construct a map.
          //  Let key be <Level, Scalar, TransposeIn, TransposeOut, LoadCallback, StoreCallback ..., kernel_id>, and
          //  the map will be from key ---> Impl. prepare such keys during commit time, and store them. This
          //  can extend to other levels as well.
          //  Get rid of this switch from here
          event = desc.queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<Scalar, 1> loc(desc.local_mem_per_factor[level_num], cgh);
            sycl::local_accessor<Scalar, 1> loc_twiddles(2 * factor, cgh);
            auto global_range = desc.launch_configurations[level_num];
            auto local_range = desc.launch_configurations[level_num];
            switch (Level) {
              case detail::level::WORKITEM:
                if (last_kernel) {
                  cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, mem, detail::transpose::TRANSPOSED,
                                                           detail::transpose::NOT_TRANSPOSED,
                                                           detail::transpose::TRANSPOSED, false, false>>(
                      sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it, sycl::kernel_handler kh) {
                        detail::workitem_dispatch_impl<Dir, detail::transpose::NOT_TRANSPOSED,
                                                       detail::transpose::TRANSPOSED, SubgroupSize,
                                                       detail::cooley_tukey_size_list_t, false, false>(
                            desc.scratch.get(), out, loc, batch_size, it, scale_factor, factor);
                      });
                } else {
                  cgh.parallel_for<detail::workitem_kernel<Scalar, Domain, Dir, mem, detail::transpose::TRANSPOSED,
                                                           detail::transpose::TRANSPOSED, detail::transpose::TRANSPOSED,
                                                           false, true>>(
                      sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> it, sycl::kernel_handler kh) {
                        detail::workitem_dispatch_impl<Dir, detail::transpose::TRANSPOSED,
                                                       detail::transpose::NOT_TRANSPOSED, SubgroupSize,
                                                       detail::cooley_tukey_size_list_t, false, true>(
                            desc.scratch.get(), desc.scratch.get(), loc, batch_size, it, scale_factor, factor,
                            desc.twiddles_forward.get() + interemediate_twiddles_offset);
                      });
                }
                interemediate_twiddles_offset += 2 * factor;
                break;

              case detail::level::SUBGROUP:
                // TODO: use spec constant
                auto factor_sg = detail::factorize_sg(factor, SubgroupSize);
                auto factor_wi = factor / factor_sg;
                if (last_kernel) {
                  cgh.parallel_for <
                      detail::subgroup_kernel<Scalar, Domain, Dir, mem, detail::transpose::NOT_TRANSPOSED,
                                              detail::transpose::TRANSPOSED, false, false, SubgroupSize>(
                          sycl::nd_range<1>(global_range, local_range),
                          [=](sycl::nd_item it, sycl::kernel_handler kh) {
                            detail::subgroup_dispatch_impl<Dir, detail::transpose::NOT_TRANSPOSED,
                                                           detail::transpose::TRANSPOSED, SubgroupSize,
                                                           detail::cooley_tukey_size_list_t, false, false>(
                                factor_wi, factor_sg, desc.scratch.get(), out, loc, loc_twiddles, batch_size, it,
                                desc.twiddles_forward.get() + local_mem_twiddles_offset, scale_factor);
                          });
                } else {
                  cgh.parallel_for <
                      detail::subgroup_kernel<Scalar, Domain, Dir, mem, detail::transpose::NOT_TRANSPOSED,
                                              detail::transpose::TRANSPOSED, false, false, SubgroupSize>(
                          sycl::nd_range<1>(global_range, local_range),
                          [=](sycl::nd_item it, sycl::kernel_handler kh) {
                            detail::subgroup_dispatch_impl<Dir, detail::transpose::NOT_TRANSPOSED,
                                                           detail::transpose::TRANSPOSED, SubgroupSize,
                                                           detail::cooley_tukey_size_list_t, false, true>(
                                factor_wi, factor_sg, desc.scratch.get(), desc.scratch.get(), loc, loc_twiddles,
                                batch_size, it, desc.twiddles_forward.get() + local_mem_twiddles_offset, scale_factor,
                                desc.twiddles_forward.get() + interemediate_twiddles_offset);
                          });
                }
                interemediate_twiddles_offset += 2 * factor;
                break;
            }
          });
        }
        if (Level == detail::level::SUBGROUP) {
          local_mem_twiddles_offset += 2 * factor;
        }
        if (Level == detail::level::WORKGROUP) {
          auto N = detail::factorize(factor);
          local_mem_twiddles_offset += 2 * (factor + N + factor / N);
        }
        desc.queue.wait();
      }
    }
    return event;  // returning misleading event, waiting on this event does not ensure device compute is done, return
                   // vector of events ?
  }
};

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::set_spec_constants_struct::inner<detail::level::DEVICE, Dummy> {
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) {
    in_bundle.template set_specialization_constant<detail::factor_wi_spec_const>(desc.factors[0]);
    in_bundle.template set_specialization_constant<detail::factor_sg_spec_const>(desc.factors[1]);
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::DEVICE,
                                                                                         TransposeIn, Dummy> {
  static std::size_t execute(committed_descriptor& desc, std::size_t fft_size) {
    auto get_local_mem_usage_per_level = [](committed_descriptor<Scalar, Domain> committed_descriptor,
                                            std::size_t factor, detail::level Level, bool transposed) -> std::size_t {
      switch (Level) {
        case detail::level::WORKITEM:
          if (transposed) {
            return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM,
                                                                   detail::transpose::TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          } else {
            return num_scalars_in_local_mem_struct::template inner<detail::level::WORKITEM,
                                                                   detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          }
          break;
        case detail::level::SUBGROUP:
          if (transposed) {
            return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP,
                                                                   detail::transpose::TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          } else {
            return num_scalars_in_local_mem_struct::template inner<detail::level::SUBGROUP,
                                                                   detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          }
          break;
        case detail::level::WORKGROUP:
          if (transposed) {
            return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                                   detail::transpose::TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          } else {
            return num_scalars_in_local_mem_struct::template inner<detail::level::WORKGROUP,
                                                                   detail::transpose::NOT_TRANSPOSED, Dummy,
                                                                   std::size_t>::execute(committed_descriptor, factor);
          }
          break;
        default:
          throw std::logic_error("Invalid factor level for global implementation");
      }
    };
    // use std::call_once ?
    if (desc.factors.empty()) {
      bool transposed = true;
      for (std::size_t i = 0; i < desc.factors.size(); i++) {
        if (i == desc.factors.size() - 1) {
          transposed = false;
        }
        auto factor = desc.factors[i];
        detail::level Level = desc.levels[i];
        desc.local_mem_per_factor.push_back(get_local_mem_usage_per_level(desc, factor, Level, transposed));
      }
      return 0;
      // For now, just let this reside here, refactor later
      int index = 0;
      desc.launch_configurations.clear();
      for (detail::level Level : desc.levels) {
        std::size_t fft_size = desc.factors[index];
        std::size_t batch_size =
            std::accumulate(desc.factors.begin() + index, desc.factors.end(), 1, std::multiplies<std::size_t>());
        desc.launch_configurations.push_back(
            detail::get_launch_configuration(Level, fft_size, batch_size, desc.n_compute_units, desc.used_sg_size));
      }
    } else {
      return 0;
    }
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_struct::inner<detail::level::DEVICE, TransposeIn,
                                                                                    Dummy> {
  static std::size_t execute(committed_descriptor& desc) {
    return num_scalars_in_local_mem_impl_struct::template inner<detail::level::DEVICE, TransposeIn, Dummy>::execute(
        desc, desc.params.lengths[0]);
  }
};

}  // namespace portfft

#endif