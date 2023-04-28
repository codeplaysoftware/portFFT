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

#ifndef SYCLFFT_BENCH_BENCH_UTILS_HPP
#define SYCLFFT_BENCH_BENCH_UTILS_HPP

#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "enums.hpp"
#include "reference_dft.hpp"

/**
 * @brief Compares two arrays
 *
 * @tparam type Type of the two arrays
 * @param array1 pointer of type to the reference output
 * @param array2 pointer of type to the device output
 * @param dimensions Dimensions of the reference output
 * @param absTol absolute tolerance value during to pass the comparision
 * @param utilize_symm Whether or not device output exploit symmetric nature of transform
 * @return true if the arrays are equal within the given tolerance
 */
template <typename type>
bool compare_result(type* reference_output, type* device_output, const std::vector<int>& dimensions, double absTol,
                    bool utilize_symm = false) {
  int symm_col = dimensions.back();
  if (utilize_symm) {
    symm_col = symm_col / 2 + 1;
  }
  int dims_squashed = std::accumulate(dimensions.begin(), dimensions.end() - 1, 1, std::multiplies<int>());
  for (int i = 0; i < dims_squashed; i++) {
    for (int j = 0; j < symm_col; j++) {
      int reference_output_idx = i * dimensions.back() + j;
      int device_output_idx = i * symm_col + j;
      if (!(std::abs(reference_output[reference_output_idx] - device_output[device_output_idx]) < absTol)) {
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Compute the reference DFT and compare it with the provided output
 *
 * @tparam dir DFT direction
 * @tparam T Data type (domain and precision)
 * @param host_input Input used
 * @param host_output Computed output
 * @param batch Batch size
 * @param N DFT size
 * @param scaling_factor DFT scaling_factor
 */
template <sycl_fft::direction dir, typename T>
void verify_dft(T* host_input, T* host_output, std::size_t batch, std::size_t N, double scaling_factor = 1.0) {
  std::vector<T> host_result(N);
  for (std::size_t i = 0; i < batch; i++) {
    reference_dft<dir>(host_input + i * N, host_result.data(), {static_cast<int>(N)}, scaling_factor);
    bool correct = compare_result(host_output + i * N, host_result.data(), {static_cast<int>(N)}, 1e-5);
    if (!correct) {
      throw std::runtime_error("Verification Failed");
    }
  }
}

#endif  // SYCLFFT_BENCH_BENCH_UTILS_HPP
