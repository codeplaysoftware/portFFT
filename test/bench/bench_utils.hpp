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

/**
 * @brief Compares two arrays 
 * 
 * @tparam type Type of the two arrays
 * @param array1 pointer of type to the first array
 * @param array2 pointer of type to the second array
 * @param num_elements total number of elements to compare
 * @param absTol absolute tolerance value during to pass the comparision
 * @return true 
 * @return false 
 */
template <typename type>
bool compare_arrays(type* array1, type* array2, size_t num_elements, double absTol) {
  bool correct = true;
  for (size_t i = 0; i < num_elements; i++) {
    correct = correct && (std::abs(array1[i] - array2[i]) <= absTol);
  }
  return correct;
}

#endif //SYCLFFT_BENCH_BENCH_UTILS_HPP
