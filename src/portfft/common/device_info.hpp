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

#ifndef PORTFFT_COMMON_DEVICE_INFO_HPP
#define PORTFFT_COMMON_DEVICE_INFO_HPP

#include <sycl/sycl.hpp>

#include "portfft/common/exceptions.hpp"
#include "portfft/common/runtime_tuning_profile.hpp"
#include "portfft/defines.hpp"

namespace portfft::detail {

/// @brief A device and often-used information for kernel dispatch.
class device_info {
 public:
  // Device
  sycl::device dev;
  // Compute unit count
  Idx n_compute_units;
  // Supported sub-group sizes
  std::vector<std::size_t> supported_sg_sizes;
  // The local memory size
  Idx local_memory_size;
  // Last level cache size
  IdxGlobal llc_size;

  device_info(sycl::queue& q)
      : dev(q.get_device()),
        n_compute_units(static_cast<Idx>(dev.get_info<sycl::info::device::max_compute_units>())),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        local_memory_size(static_cast<Idx>(dev.get_info<sycl::info::device::local_mem_size>())),
        llc_size(static_cast<IdxGlobal>(dev.get_info<sycl::info::device::global_mem_cache_size>())) {}

  /** Return true if the device supports a given subgroup size.
   * @param subgroup_size The subgroup size to check for compatibility.
   */
  bool supports_sg_size(int subgroup_size) const {
    return static_cast<bool>(std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), subgroup_size));
  }

  /** Return true if the device supports all kernels given.
   * @param ids The kernel ids to check for compatibilty.
   */
  bool is_compatible(std::vector<sycl::kernel_id>& ids) { return sycl::is_compatible(ids, dev); }

  rt_config_desc get_runtime_config() {
    std::string device_name = dev.get_info<sycl::info::device::name>();
    rt_config_desc config;
    auto contains_fn = [&](std::string&& s) { return device_name.find(s) != std::string::npos; };

    // Try and choose a profile based on the device's name.
    if (contains_fn("Intel")) {
      if (contains_fn("UHD")) {
        config = rt_config_descs[rt_config::intel_uhd];
      }
      if (contains_fn("Arc")) {
        config = rt_config_descs[rt_config::intel_arc];
      }
      if (contains_fn("Max")) {
        config = rt_config_descs[rt_config::intel_pvc];
      }
    } else if (contains_fn("AMD")) {
      config = rt_config_descs[rt_config::amd];
    } else if (contains_fn("NVIDIA")) {
      config = rt_config_descs[rt_config::nvidia];
    } else {
      config = rt_config_descs[rt_config::amd];
    }

    // Check if the preferred runtime config is available and compatible.
    if (!is_enabled(config.preferred_ct_profile) || !supports_sg_size(sg_size(config.preferred_ct_profile))) {
      config.preferred_ct_profile =
          find_ct_profile_with_predicate([&](ct_profile profile) { return supports_sg_size(sg_size(profile)); });
    }
    if (config.preferred_ct_profile == ct_profile::invalid) {
      throw invalid_configuration("Could not find a suitable compiled kernel configuration for device.");
    }

    return config;
  }
};

}  // namespace portfft::detail

#endif  // PORTFFT_COMMON_DEVICE_INFO_HPP
