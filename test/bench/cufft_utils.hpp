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

#ifndef SYCL_FFT_BENCH_CUFFT_UTILS_HPP
#define SYCL_FFT_BENCH_CUFFT_UTILS_HPP

#include <cuda.h>
#include <curand.h>

/**
 * @brief populates device ptr with random values using the curand host api
 * 
 * @tparam T Type of device pointer, must be either float, double or int
 * @param dev_ptr Device Pointer
 * @param N Batch times the number of elements in each FFT
 */
template <typename T>
void populate_with_random(T* dev_ptr, std::size_t N) {
  curandGenerator_t generator;
  if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW) != CURAND_STATUS_SUCCESS) {
    std::runtime_error("Failed to Create Random Number Generator");
  }

  if (curandSetPseudoRandomGeneratorSeed(generator, 5678) != CURAND_STATUS_SUCCESS) {
    std::runtime_error("Failed to Set Random Seed");
  }

  if ([&]() {
        if constexpr (std::is_same_v<T, float>) {
          return curandGenerateNormal(generator, dev_ptr, 2 * N, 0, 2);
        }

        else {
          return curandGenerateNormalDouble(generator, dev_ptr, 2 * N, 0, 2);
        }
      }() != CURAND_STATUS_SUCCESS) {
    std::runtime_error("Failed to populate device pointer with random values");
  }
  cudaDeviceSynchronize();
  curandDestroyGenerator(generator);
}

#endif //SYCL_FFT_BENCH_CUFFT_UTILS_HPP
