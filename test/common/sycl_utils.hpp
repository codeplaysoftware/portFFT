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

#ifndef PORTFFT_TEST_COMMON_SYCL_UTILS_HPP
#define PORTFFT_TEST_COMMON_SYCL_UTILS_HPP

#include <memory>

#include <sycl/sycl.hpp>

/**
 * Utility function to create a shared pointer, with memory allocated on device
 * @tparam T Type of the memory being allocated
 * @param size Number of elements to allocate
 * @param queue Associated queue
 */
template <typename T>
inline std::shared_ptr<T> make_shared(std::size_t size, sycl::queue queue) {
  return std::shared_ptr<T>(sycl::malloc_device<T>(size, queue), [captured_queue = queue](T* ptr) {
    if (ptr != nullptr) {
      sycl::free(ptr, captured_queue);
    }
  });
}

#endif  // PORTFFT_TEST_COMMON_SYCL_UTILS_HPP
