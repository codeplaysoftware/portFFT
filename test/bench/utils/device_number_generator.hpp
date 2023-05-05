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

#ifndef SYCL_FFT_BENCH_DEVICE_NUMBER_GENERATOR_HPP
#define SYCL_FFT_BENCH_DEVICE_NUMBER_GENERATOR_HPP

#include <complex>
#include <sycl/sycl.hpp>

template <typename T>
class memFillKernel;

/**
 * @brief Kernel for populating device pointer with values
 *
 * @tparam T Type of the input pointer
 * @param input The Device pointer
 * @param queue sycl::queue associated with the device
 * @param num_elements Batch times the number of elements in each FFT
 */
template <typename T>
void memFill(T* input, sycl::queue& queue, std::size_t num_elements) {
  constexpr std::size_t group_dim = 32;
  auto global_range = static_cast<std::size_t>(ceil(static_cast<float>(num_elements) / group_dim) * group_dim);
  queue.submit([&](sycl::handler& h) {
    h.parallel_for<memFillKernel<T>>(sycl::nd_range<1>(sycl::range<1>(global_range), sycl::range<1>(group_dim)),
                                     [=](sycl::nd_item<1> itm) {
                                       auto idx = itm.get_global_id(0);
                                       if (idx < num_elements) {
                                         const auto id = static_cast<float>(itm.get_local_linear_id());
                                         const auto divisor = static_cast<float>(group_dim);
                                         if constexpr (std::is_floating_point_v<T>) {
                                           input[idx] = static_cast<T>(id / divisor);
                                         } else {
                                           input[idx] = T{id / divisor, id + 1 / divisor};
                                         }
                                       }
                                     });
  });
  queue.wait();
}

#endif  // SYCL_FFT_BENCH_DEVICE_NUMBER_GENERATOR_HPP
