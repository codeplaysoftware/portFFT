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

#include <gtest/gtest.h>

#include "sycl_utils.hpp"

/**
 * Print information of the device selected by the default selector.
 * Use this as a test to print the information once when using ctest.
 */
void print_device() {
  namespace info = sycl::info::device;
  sycl::queue queue;
  sycl::device dev = queue.get_device();
  sycl::platform platform = dev.get_info<info::platform>();

  std::string device_type_str = get_device_type(dev);
  std::string supports_double_str = dev.has(sycl::aspect::fp64) ? "yes" : "no";
  std::string supports_usm_str = dev.has(sycl::aspect::usm_device_allocations) ? "yes" : "no";
  std::string subgroup_sizes_str = get_device_subgroup_sizes(dev);
  auto local_memory_size = dev.get_info<sycl::info::device::local_mem_size>();

  std::cout << "Running tests on:\n";
  std::cout << "  Device type: " << device_type_str << "\n";
  std::cout << "  Platform: " << platform.get_info<sycl::info::platform::name>() << "\n";
  std::cout << "  Name: " << dev.get_info<info::name>() << "\n";
  std::cout << "  Vendor: " << dev.get_info<info::vendor>() << "\n";
  std::cout << "  Version: " << dev.get_info<info::version>() << "\n";
  std::cout << "  Driver version: " << dev.get_info<info::driver_version>() << "\n";
  std::cout << "  Double supported: " << supports_double_str << "\n";
  std::cout << "  USM supported: " << supports_usm_str << "\n";
  std::cout << "  Subgroup sizes: " << subgroup_sizes_str << "\n";
  std::cout << "  Local memory size: " << local_memory_size << "B\n";
  std::cout << std::endl;
}

TEST(print_device, run) { print_device(); }
