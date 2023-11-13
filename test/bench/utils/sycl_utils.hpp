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

#ifndef PORTFFT_TEST_BENCH_UTILS_SYCL_UTILS_HPP
#define PORTFFT_TEST_BENCH_UTILS_SYCL_UTILS_HPP

#include <sycl/sycl.hpp>

#include <benchmark/benchmark.h>

void print_device(sycl::queue queue) {
  namespace info = sycl::info::device;
  using sycl::info::device_type;
  sycl::device dev = queue.get_device();
  sycl::platform platform = dev.get_info<info::platform>();

  std::string device_type_str;
  switch (dev.get_info<info::device_type>()) {
    case device_type::cpu:
      device_type_str = "CPU";
      break;
    case device_type::gpu:
      device_type_str = "GPU";
      break;
    case device_type::accelerator:
      device_type_str = "accelerator";
      break;
    case device_type::custom:
      device_type_str = "custom";
      break;
    case device_type::host:
      device_type_str = "host";
      break;
    default:
      device_type_str = "unknown";
      break;
  };

  bool supports_double = dev.get_info<info::double_fp_config>().empty();

  auto subgroup_sizes = dev.get_info<info::sub_group_sizes>();
  std::stringstream subgroup_sizes_str;
  subgroup_sizes_str << "[";
  for (std::size_t i = 0; i < subgroup_sizes.size(); ++i) {
    if (i > 0) {
      subgroup_sizes_str << ", ";
    }
    subgroup_sizes_str << subgroup_sizes[i];
  }
  subgroup_sizes_str << "]";

  benchmark::AddCustomContext("Device type", device_type_str);
  benchmark::AddCustomContext("Platform", platform.get_info<sycl::info::platform::name>());
  benchmark::AddCustomContext("Name", dev.get_info<info::name>());
  benchmark::AddCustomContext("Vendor", dev.get_info<info::vendor>());
  benchmark::AddCustomContext("Version", dev.get_info<info::version>());
  benchmark::AddCustomContext("Driver version", dev.get_info<info::driver_version>());
  benchmark::AddCustomContext("Double supported", (supports_double ? "no" : "yes"));
  benchmark::AddCustomContext("Subgroup sizes", subgroup_sizes_str.str());
}

#endif  // PORTFFT_TEST_BENCH_UTILS_SYCL_UTILS_HPP
