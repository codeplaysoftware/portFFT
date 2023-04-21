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

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <cmath>
#include <cstdarg>
#include <functional>
#include <iostream>
#include <vector>

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

#endif  // SYCLFFT_BENCH_BENCH_UTILS_HPP
