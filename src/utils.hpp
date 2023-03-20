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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/
#ifndef SYCL_FFT_UTILS_HPP
#define SYCL_FFT_UTILS_HPP

#include <sycl/sycl.hpp>
constexpr int SYCL_FFT_DEFAULT_SUB_GROUP_SIZE = 32;

template <typename kernel>
int get_max_sub_group_size(
    sycl::device& dev,
    sycl::kernel_bundle<sycl::bundle_state::executable>& exec_bundle) {
  try {
    auto k = exec_bundle.get_kernel(sycl::get_kernel_id<kernel>());
    try {
      return k.template get_info<
          sycl::info::kernel_device_specific::max_sub_group_size>(dev);
    } catch (const std::exception& e) {
      return SYCL_FFT_DEFAULT_SUB_GROUP_SIZE;
    }
  } catch (const std::exception& e) {
    return SYCL_FFT_DEFAULT_SUB_GROUP_SIZE;
  }
}

#endif