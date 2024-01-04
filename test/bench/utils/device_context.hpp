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

#ifndef PORTFFT_TEST_BENCH_UTILS_DEVICE_CONTEXT_HPP
#define PORTFFT_TEST_BENCH_UTILS_DEVICE_CONTEXT_HPP

#include <sycl/sycl.hpp>

#include <benchmark/benchmark.h>

#include "common/sycl_utils.hpp"

void add_device_context(sycl::queue queue) {
  namespace info = sycl::info::device;
  sycl::device dev = queue.get_device();
  sycl::platform platform = dev.get_info<info::platform>();

  std::string device_type_str = get_device_type(dev);
  std::string supports_double_str = dev.has(sycl::aspect::fp64) ? "yes" : "no";
  std::string supports_usm_str = dev.has(sycl::aspect::usm_device_allocations) ? "yes" : "no";
  std::string subgroup_sizes_str = get_device_subgroup_sizes(dev);
  std::string local_memory_size_str = std::to_string(dev.get_info<sycl::info::device::local_mem_size>()) + "B";

  benchmark::AddCustomContext("Device type", device_type_str);
  benchmark::AddCustomContext("Platform", platform.get_info<sycl::info::platform::name>());
  benchmark::AddCustomContext("Name", dev.get_info<info::name>());
  benchmark::AddCustomContext("Vendor", dev.get_info<info::vendor>());
  benchmark::AddCustomContext("Version", dev.get_info<info::version>());
  benchmark::AddCustomContext("Driver version", dev.get_info<info::driver_version>());
  benchmark::AddCustomContext("Double supported", supports_double_str);
  benchmark::AddCustomContext("USM supported", supports_usm_str);
  benchmark::AddCustomContext("Subgroup sizes", subgroup_sizes_str);
  benchmark::AddCustomContext("Local memory size", local_memory_size_str);
}

#endif  // PORTFFT_TEST_BENCH_UTILS_DEVICE_CONTEXT_HPP
