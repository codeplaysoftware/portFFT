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

#include <common/cooley_tukey_compiled_sizes.hpp>
#include <common/helpers.hpp>
#include <common/transfers.hpp>
#include <descriptor.hpp>
#include <dispatcher/subgroup_dispatcher.hpp>
#include <dispatcher/workgroup_dispatcher.hpp>
#include <dispatcher/workitem_dispatcher.hpp>
#include <enums.hpp>
#include <mutex>

#include <sycl/sycl.hpp>

namespace portfft {
namespace detail {

constexpr static sycl::specialization_id<std::size_t> fft_size_spec_constant{};
constexpr static sycl::specialization_id<int> factor_1_spec_constant{};
constexpr static sycl::specialization_id<int> factor_2_spec_constant{};

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
      std::size_t factor_sg =
          static_cast<std::size_t>(factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size)));
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

template <int kernel_id, typename Scalar, domain Domain, direction Dir, detail::memory mem,
          detail::transpose TransposeIn, detail::transpose TransposeOut, bool ApplyLoadCallback,
          bool ApplyStoreCallback, int SubgroupSize>
struct dispatch_level_struct {
  static sycl::event execute(const Scalar* input, Scalar* output, const Scalar* twiddles_ptr,
                             const int* dev_factors_and_batches, std::size_t intermediate_twiddle_offset,
                             std::size_t local_mem_offset, std::size_t num_concurrent_batches, Scalar scale_factor,
                             std::vector<int>& factors, std::vector<detail::level>& Levels,
                             std::vector<std::size_t> local_mem_per_factor,
                             std::vector<std::pair<sycl::range<1>, sycl::range<1>>> launch_configurations,
                             sycl::queue& queue) {
    sycl::event Event;

    std::size_t factor = factors[kernel_id];
    level Level = Levels[kernel_id];
    auto global_size = launch_configurations[kernel_id].first;
    auto local_size = launch_configurations[kernel_id].second;
    auto local_mem_usage = local_mem_per_factor[kernel_id];
    auto num_factors = factors.size();
    auto batch_size =
        std::accumulate(factors.begin() + kernel_id + 1, factors.end(), 1, std::multiplies<std::size_t>());

    auto twiddle_local_mem = [=]() {
      if (Level == level::WORKITEM)
        return static_cast<std::size_t>(0);
      else
        return 2 * factor;
    }();
    queue.wait();
    for (std::size_t i = 0; i < num_concurrent_batches; i++) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Scalar, 1> loc(local_mem_usage, cgh);
        sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_local_mem, cgh);
        sycl::local_accessor<int, 1> loc_factors(num_factors, cgh);
        cgh.parallel_for<global_kernel<Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                                       ApplyStoreCallback, SubgroupSize, kernel_id>>(
            sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
              // its technically an inclusive scan viewed as if the array is reversed
              std::size_t factors_inclusive_scan[kernel_id];
              // really missing constant memory support here.
              global2local<pad::DONT_PAD, detail::level::WORKGROUP, SubgroupSize>(it, dev_factors_and_batches,
                                                                                  &loc_factors[0], 2 * num_factors - 1);
              sycl::group_barrier(it.get_group());
              factors_inclusive_scan[0] = loc_factors[kernel_id - 1];
              std::size_t product_sub_batches = 1;

#pragma unroll
              for (int _i = 0; _i < kernel_id; _i++) {
                product_sub_batches *= static_cast<std::size_t>(loc_factors[i]);
              }
#pragma unroll
              for (int _i = 1; _i < kernel_id; _i++) {
                factors_inclusive_scan[i] = factors_inclusive_scan[i - 1] * loc_factors[kernel_id - 1 - i];
              }

              for (std::size_t j = 0; j < product_sub_batches; j++) {
                std::size_t offset = 0;
                for (std::size_t k = 0; k < kernel_id - 1; k++) {
                  offset += ((j / factors_inclusive_scan[k + 1]) % loc_factors[k]) * loc_factors[num_factors + k];
                }
                switch (Level) {
                  case detail::level::WORKITEM:
                    workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, cooley_tukey_size_list_t,
                                           ApplyLoadCallback, ApplyStoreCallback, Scalar>(
                        input + offset, output + offset, &loc[0], batch_size, it, scale_factor, factor, twiddles_ptr);
                    break;
                  case detail::level::SUBGROUP:
                    auto factor_sg = detail::factorize_sg(static_cast<int>(factor), SubgroupSize);
                    subgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, cooley_tukey_size_list_t,
                                           ApplyLoadCallback, ApplyStoreCallback, Scalar>(
                        static_cast<int>(factor) / factor_sg, factor_sg, input + offset, output + offset, &loc[0],
                        &loc_twiddles[0], batch_size, it, twiddles_ptr + local_mem_offset, scale_factor,
                        twiddles_ptr + intermediate_twiddle_offset);
                    break;
                }
              }
            });
      });
    }
    intermediate_twiddle_offset += 2 * factor * batch_size;
    if (Level == level::SUBGROUP) {
      local_mem_offset += 2 * factor;
    }

    if (kernel_id == num_factors - 1) {
      return Event;
    }
    if (kernel_id == num_factors - 2) {
      dispatch_level_struct<kernel_id + 1, Scalar, Domain, Dir, mem, detail::transpose::NOT_TRANSPOSED,
                            detail::transpose::NOT_TRANSPOSED, false, false,
                            SubgroupSize>::execute(input, output, twiddles_ptr, dev_factors_and_batches,
                                                   intermediate_twiddle_offset, local_mem_offset,
                                                   num_concurrent_batches, scale_factor, factors, Levels,
                                                   local_mem_per_factor, launch_configurations, queue);
    } else {
      dispatch_level_struct<kernel_id + 1, Scalar, Domain, Dir, mem, detail::transpose::TRANSPOSED,
                            detail::transpose::TRANSPOSED, false, true,
                            SubgroupSize>::execute(input, output, twiddles_ptr, dev_factors_and_batches,
                                                   intermediate_twiddle_offset, local_mem_offset,
                                                   num_concurrent_batches, scale_factor, factors, Levels,
                                                   local_mem_per_factor, launch_configurations, queue);
    }
  }
}
};

template <typename Scalar, domain Domain, direction Dir, detail::memory mem, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize>
struct dispatch_level_struct<0, Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                             ApplyStoreCallback, SubgroupSize> {
  static sycl::event execute(const Scalar* input, Scalar* output, const Scalar* twiddles_ptr,
                             const int* dev_factors_and_batches, std::size_t intermediate_twiddle_offset,
                             std::size_t local_mem_offset, std::size_t num_concurrent_batches, Scalar scale_factor,
                             std::vector<int>& factors, std::vector<detail::level>& Levels,
                             std::vector<std::size_t> local_mem_per_factor,
                             std::vector<std::pair<sycl::range<1>, sycl::range<1>>> launch_configurations,
                             sycl::queue& queue) {
    sycl::event Event;

    std::size_t factor = factors[0];
    level Level = Levels[0];
    auto global_size = launch_configurations[0].first;
    auto local_size = launch_configurations[0].second;
    auto local_mem_usage = local_mem_per_factor[0];
    auto num_factors = factors.size();
    auto batch_size =
        static_cast<std::size_t>(std::accumulate(factors.begin() + 1, factors.end(), 1, std::multiplies<int>()));

    auto twiddle_local_mem = [=]() {
      if (Level == level::WORKITEM)
        return static_cast<std::size_t>(0);
      else
        return 2 * factor;
    }();
    queue.wait();
    for (std::size_t i = 0; i < num_concurrent_batches; i++) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Scalar, 1> loc(local_mem_usage, cgh);
        sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_local_mem, cgh);
        sycl::local_accessor<int, 1> loc_factors(num_factors, cgh);
        cgh.parallel_for<global_kernel<Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                                       ApplyStoreCallback, SubgroupSize, 0>>(
            sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
              // really missing constant memory support here.
              global2local<pad::DONT_PAD, detail::level::WORKGROUP, SubgroupSize>(it, dev_factors_and_batches,
                                                                                  &loc_factors[0], num_factors);
              sycl::group_barrier(it.get_group());
              switch (Level) {
                case detail::level::WORKITEM:
                  workitem_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, cooley_tukey_size_list_t,
                                         ApplyLoadCallback, ApplyStoreCallback, Scalar>(
                      input, output, &loc[0], batch_size, it, scale_factor, factor, twiddles_ptr);
                  break;
                case detail::level::SUBGROUP:
                  auto factor_sg = detail::factorize_sg(static_cast<int>(factor), SubgroupSize);
                  subgroup_dispatch_impl<Dir, TransposeIn, TransposeOut, SubgroupSize, cooley_tukey_size_list_t,
                                         ApplyLoadCallback, ApplyStoreCallback, Scalar>(
                      static_cast<int>(factor) / factor_sg, factor_sg, input, output, &loc[0], &loc_twiddles[0],
                      batch_size, it, twiddles_ptr + local_mem_offset, scale_factor,
                      twiddles_ptr + intermediate_twiddle_offset);
                  break;
              }
            });
      });
    }
    intermediate_twiddle_offset += 2 * factor * batch_size;
    if (Level == level::SUBGROUP) {
      local_mem_offset += 2 * factor;
    }
    if (0 == num_factors - 2) {
      return dispatch_level_struct<1, Scalar, Domain, Dir, mem, detail::transpose::NOT_TRANSPOSED,
                                   detail::transpose::TRANSPOSED, false, false,
                                   SubgroupSize>::execute(input, output, twiddles_ptr, dev_factors_and_batches,
                                                          intermediate_twiddle_offset, local_mem_offset,
                                                          num_concurrent_batches, scale_factor, factors, Levels,
                                                          local_mem_per_factor, launch_configurations, queue);
    } else {
      return dispatch_level_struct<1, Scalar, Domain, Dir, mem, detail::transpose::TRANSPOSED,
                                   detail::transpose::TRANSPOSED, false, true,
                                   SubgroupSize>::execute(input, output, twiddles_ptr, dev_factors_and_batches,
                                                          intermediate_twiddle_offset, local_mem_offset,
                                                          num_concurrent_batches, scale_factor, factors, Levels,
                                                          local_mem_per_factor, launch_configurations, queue);
    }
  }
};

template <typename Scalar, domain Domain, direction Dir, detail::memory mem, detail::transpose TransposeIn,
          detail::transpose TransposeOut, bool ApplyLoadCallback, bool ApplyStoreCallback, int SubgroupSize>
struct dispatch_level_struct<33, Scalar, Domain, Dir, mem, TransposeIn, TransposeOut, ApplyLoadCallback,
                             ApplyStoreCallback, SubgroupSize> {
  static sycl::event execute(const Scalar*, Scalar*, const Scalar*, const int*, std::size_t, std::size_t, std::size_t,
                             Scalar, std::vector<int>&, std::vector<detail::level>&, std::vector<std::size_t>,
                             std::vector<std::pair<sycl::range<1>, sycl::range<1>>>, sycl::queue&) {
    sycl::event Event;
    return Event;
  }
};

}  // namespace detail

template <typename Scalar, domain Domain>
template <typename Dummy>
struct committed_descriptor<Scalar, Domain>::calculate_twiddles_struct::inner<detail::level::GLOBAL, Dummy> {
  static Scalar* execute(committed_descriptor& desc) {
    // first calculate space for Intermediate twiddles
    //  Then iff level is subgroup, calculate twiddles required for subgroup.
    auto calc_total_mem_for_twiddles = [=]() -> std::size_t {
      std::size_t num_scalars = 0;
      int index = 0;
      for (auto iter = desc.factors.begin(); iter + 1 != desc.factors.end(); iter++) {
        num_scalars +=
            static_cast<std::size_t>(*iter * std::accumulate(iter + 1, desc.factors.end(), 1, std::multiplies<int>()));
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
          double thetha = (-2 * M_PI * static_cast<double>(i * j) / static_cast<double>(N * M));
          ptr[offset++] = static_cast<Scalar>(std::cos(thetha));
          ptr[offset++] = static_cast<Scalar>(std::sin(thetha));
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
                                                               T_out>::inner<detail::level::GLOBAL, Dummy> {
  static sycl::event execute(committed_descriptor& desc, const T_in& in, T_out& out, Scalar scale_factor,
                             const std::vector<sycl::event>& dependencies) {
    // TODO:: A better solution to this will be to use a virtual function table, or construct a map.
    //  Let key be <Level, Scalar, TransposeIn, TransposeOut, LoadCallback, StoreCallback ..., kernel_id>, and
    //  the map will be from key ---> Impl. prepare such keys during commit time, and store them. This
    //  can extend to other levels as well.
    //  Get rid of this switch from here
    sycl::event Event;
    constexpr detail::memory mem = std::is_pointer<T_out>::value ? detail::memory::USM : detail::memory::BUFFER;
    num_scalars_in_local_mem_struct::template inner<detail::level::GLOBAL, TransposeIn, Dummy>::execute(desc);
    std::size_t local_mem_twiddles_initial_offset = 0;
    for (auto iter = desc.factors.begin(); iter + 1 != desc.factors.end(); iter++) {
      local_mem_twiddles_initial_offset +=
          static_cast<std::size_t>(*iter * std::accumulate(iter + 1, desc.factors.end(), 1, std::multiplies<int>()));
    }
    auto input_pointer = static_cast<const Scalar*>(desc.scratch.get());
    auto default_output_pointer = desc.scratch.get();
    auto twiddles_ptr = static_cast<const Scalar*>(desc.twiddles_forward.get());
    auto dev_factors_and_batches = static_cast<const int*>(desc.dev_factors.get());
    std::size_t fft_size = desc.params.lengths[0];
    // This copy should be avoided. ideally the first kernel should read from acutual input pointer
    //  and the from there on the consecutive kernels should use the scratch pointer.
    desc.queue
        .submit([&](sycl::handler& cgh) {
          auto in_acc_or_usm = detail::get_access<const Scalar>(in, cgh);
          cgh.parallel_for(sycl::nd_range<1>({SubgroupSize * desc.n_compute_units * 4}, {SubgroupSize}),
                           [=](sycl::nd_item<1> it) {
                             for (std::size_t i = it.get_global_id(0); i < 2 * fft_size; i += it.get_global_range(0)) {
                               default_output_pointer[i] = in_acc_or_usm[i];
                             }
                           });
        })
        .wait();
    for (std::size_t batch_num = 0; batch_num < desc.params.number_of_transforms; batch_num += desc.num_batches_in_l2) {
      Event = detail::dispatch_level_struct<0, Scalar, Domain, Dir, mem, detail::transpose::TRANSPOSED,
                                            detail::transpose::TRANSPOSED, false, true,
                                            SubgroupSize>::execute(input_pointer + batch_num * 2 * fft_size,
                                                                   default_output_pointer + batch_num * 2 * fft_size,
                                                                   twiddles_ptr, dev_factors_and_batches, 0,
                                                                   local_mem_twiddles_initial_offset,
                                                                   desc.num_batches_in_l2, scale_factor, desc.factors,
                                                                   desc.levels, desc.local_mem_per_factor,
                                                                   desc.launch_configurations, desc.queue);
    }
    desc.queue.wait();
    // Same as above goes for output pointer as well
    desc.queue
        .submit([&](sycl::handler& cgh) {
          auto out_acc_or_usm = detail::get_access<Scalar>(out, cgh);
          cgh.parallel_for(sycl::nd_range<1>({SubgroupSize * desc.n_compute_units * 4}, {SubgroupSize}),
                           [=](sycl::nd_item<1> it) {
                             for (std::size_t i = it.get_global_id(0); i < 2 * fft_size; i += it.get_global_range(0)) {
                               out_acc_or_usm[i] = default_output_pointer[i];
                             }
                           });
        })
        .wait();
    return Event;
  }
};

template <typename Scalar, domain Domain>
template <detail::transpose TransposeIn, typename Dummy>
struct committed_descriptor<Scalar, Domain>::num_scalars_in_local_mem_impl_struct::inner<detail::level::GLOBAL,
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
  static void execute(committed_descriptor& desc, sycl::kernel_bundle<sycl::bundle_state::input>& in_bundle) { ; }
};

}  // namespace portfft

#endif