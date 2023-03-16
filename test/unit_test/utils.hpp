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

#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <random>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace std::complex_literals;

#define CHECK_QUEUE(queue) \
  if (!queue.first) GTEST_SKIP() << queue.second;

template <typename type>
void compare_arrays(std::vector<type> array1, std::vector<type> array2,
                    double tol) {
  ASSERT_EQ(array1.size(), array2.size());
  for (size_t i = 0; i < array1.size(); i++) {
    EXPECT_NEAR(array1[i].real(), array2[i].real(), tol);
    EXPECT_NEAR(array1[i].imag(), array2[i].imag(), tol);
  }
}

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(std::vector<TypeIn>& in, std::vector<TypeOut>& out,
                           size_t length, size_t offset = 0) {
  long double TWOPI = 2.0l * std::atan(1.0l) * 4.0l;

  size_t N = length;
  for (size_t k = 0; k < N; k++) {
    std::complex<long double> out_temp = 0;
    for (size_t n = 0; n < N; n++) {
      auto multiplier = std::complex<long double>{std::cos(n * k * TWOPI / N),
                                                  -std::sin(n * k * TWOPI / N)};
      out_temp +=
          static_cast<std::complex<long double>>(in[offset + n]) * multiplier;
    }
    out[offset + k] = static_cast<TypeOut>(out_temp);
  }
}

template <typename deviceSelector>
std::pair<std::optional<sycl::queue>, std::string> get_queue(
    deviceSelector selector) {
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
