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

#include "common/subgroup.hpp"
#include "common/transfers.hpp"
#include "common/workitem.hpp"
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

/**
 * Calculates the amount of local memory needed for given problem.
 *
 * @tparam T type of the scalar used for computations
 * @param fft_size size of each transform
 * @param Subgroup_size size of subgroup used by the compute kernel
 * @return Number of elements of size T that need to fit into local memory
 */
template <typename T>
std::size_t num_scalars_in_local_mem(std::size_t fft_size, std::size_t subgroup_size) {
  if (sycl_fft::detail::fits_in_wi<T>(fft_size)) {
    return detail::pad_local(2 * fft_size * subgroup_size) * SYCLFFT_SGS_IN_WG;
  } else {
    if (fft_size <= sycl_fft::detail::MAX_FFT_SIZE_WI * subgroup_size) {
      int factor_sg = detail::factorize_sg(static_cast<int>(fft_size), static_cast<int>(subgroup_size));
      int fact_wi = static_cast<int>(fft_size) / factor_sg;
      if (sycl_fft::detail::fits_in_wi<T>(fact_wi)) {
        std::size_t n_ffts_per_sg = subgroup_size / static_cast<std::size_t>(factor_sg);
        return detail::pad_local(2 * fft_size * n_ffts_per_sg) * SYCLFFT_SGS_IN_WG;
      }
    } else {
      return detail::pad_local(2 * fft_size);
    }
  }
  return 0;
}

template <typename T, bool TransposeIn>
bool exceeds_local_mem_size(sycl::queue& queue, int fft_size) {
  std::size_t local_mem_available = queue.get_device().get_info<sycl::info::device::local_mem_size>();
  if constexpr (TransposeIn) {
    std::size_t local_memory_required =
        num_scalars_in_local_mem<T>(static_cast<std::size_t>(fft_size * SYCLFFT_SUBGROUP_SIZES * SYCLFFT_SGS_IN_WG / 2),
                                    SYCLFFT_SUBGROUP_SIZES) *
        sizeof(T);
    if (!detail::fits_in_wi<T>(fft_size / detail::factorize_sg(fft_size, (SYCLFFT_SUBGROUP_SIZES)))) {
      int N = detail::factorize(fft_size);
      int M = fft_size / N;
      local_memory_required += 2 * sizeof(T) * static_cast<std::size_t>(M + N);
    } else {
      local_memory_required += 2 * sizeof(T) * static_cast<std::size_t>(fft_size);
    }
    if (local_memory_required > local_mem_available) {
      return true;
    } else {
      return false;
    }
  }
  if (!detail::fits_in_wi<T>(fft_size / detail::factorize_sg(fft_size, (SYCLFFT_SUBGROUP_SIZES)))) {
    int N = detail::factorize(fft_size);
    int M = fft_size / N;
    std::size_t local_mem_required =
        sizeof(T) * (2 * sizeof(T) * static_cast<std::size_t>(M + N) +
                     num_scalars_in_local_mem<T>(static_cast<std::size_t>(fft_size), SYCLFFT_SUBGROUP_SIZES));
    if (local_mem_required > local_mem_available) {
      return true;
    }
  }
  return false;
}

template <typename T>
void compare_arrays(std::vector<std::complex<T>> reference_output, std::vector<std::complex<T>> device_output,
                    double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i].real(), device_output[i].real(), tol) << "i=" << i;
    EXPECT_NEAR(reference_output[i].imag(), device_output[i].imag(), tol) << "i=" << i;
  }
}

template <typename T>
void compare_arrays(std::vector<T> reference_output, std::vector<T> device_output, double tol) {
  ASSERT_EQ(reference_output.size(), device_output.size());
  for (size_t i = 0; i < reference_output.size(); i++) {
    EXPECT_NEAR(reference_output[i], device_output[i], tol) << "i=" << i;
  }
}

template <typename DeviceSelector>
std::pair<std::optional<sycl::queue>, std::string> get_queue(DeviceSelector selector) {
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
