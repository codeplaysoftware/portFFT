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

#ifndef SYCL_FFT_UNIT_TEST_UTILS_HPP
#define SYCL_FFT_UNIT_TEST_UTILS_HPP

#include "enums.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <random>
#include <sycl/sycl.hpp>

using namespace std::complex_literals;
using namespace sycl_fft;

#define CHECK_QUEUE(queue) \
  if (!queue.first) GTEST_SKIP() << queue.second;

template <typename type>
void compare_arrays(std::vector<std::complex<type>> reference_output, std::vector<std::complex<type>> device_output,
                    double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i].real(), device_output[i].real(), tol) << "i=" << i;
    EXPECT_NEAR(reference_output[i].imag(), device_output[i].imag(), tol) << "i=" << i;
  }
}

template <typename type>
void compare_arrays(std::vector<type> reference_output, std::vector<type> device_output, double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i], device_output[i], tol) << "i=" << i;
  }
}

template <typename deviceSelector>
std::pair<std::optional<sycl::queue>, std::string> get_queue(deviceSelector selector) {
  try {
    sycl::queue queue(selector);
    return std::make_pair(queue, "");
  } catch (sycl::exception& e) {
    return std::make_pair(std::nullopt, e.what());
  }
}

int fp64_selector(sycl::device dev) {
  if (dev.has(sycl::aspect::fp64))
    return 1;
  else
    return -1;
}
#endif
