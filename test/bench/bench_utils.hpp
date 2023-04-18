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
 * @param array1 pointer of type to the first array
 * @param array2 pointer of type to the second array
 * @param dimensions Dimensions of the reference output
 * @param absTol absolute tolerance value during to pass the comparision
 * @param utilize_symm Whether or not device output exploit symmetric nature of transform
 * @return true if the arrays are equal within the given tolerance
 */
template <typename type>
bool compare_arrays(type* reference_output, type* device_output, std::vector<int> dimensions, double absTol,
                    bool utilize_symm = false) {
  std::function<void(int, std::vector<int>)> nested_loop;
  std::vector<int> symm_dimensions = dimensions;
  if (utilize_symm) {
    symm_dimensions.at(symm_dimensions.size() - 1) = symm_dimensions.back() / 2 + 1;
  }
  bool correct = true;

  nested_loop = [&](int recursion_level, std::vector<int>&& iter_values) -> void {
    if (recursion_level != dimensions.size()) {
      for (int i = 0; i < dimensions.at(recursion_level - 1); i++) {
        iter_values[recursion_level - 1] = i;
        nested_loop(recursion_level + 1, iter_values);
      }
    } else {
      int nested_offset = 0;
      int symmetric_nested_offset = 0;
      for (int i = 0; i < iter_values.size(); i++) {
        symmetric_nested_offset +=
            iter_values.at(i) *
            std::accumulate(symm_dimensions.begin() + i + 1, symm_dimensions.end(), 1, std::multiplies<int>());
        nested_offset += iter_values.at(i) *
                         std::accumulate(dimensions.begin() + i + 1, dimensions.end(), 1, std::multiplies<int>());
      }
      int symm_value = dimensions.at(recursion_level - 1) / 2 + 1;
      for (int i = 0; i < dimensions.at(recursion_level - 1); i++) {
        if (utilize_symm) {
          if (i == symm_value) {
            break;
          }
        }
        correct =
            correct &&
            (std::abs(reference_output[nested_offset + i] - device_output[symmetric_nested_offset + i]) <= absTol);
      }
    }
  };

  std::vector<int> iter_values(dimensions.size() - 1);
  nested_loop(1, iter_values);
  return correct;
}

#endif  // SYCLFFT_BENCH_BENCH_UTILS_HPP
