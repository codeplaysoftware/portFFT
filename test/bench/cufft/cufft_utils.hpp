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

#ifdef SYCLFFT_VERIFY_BENCHMARK
#include <curand.h>
#endif  // SYCLFFT_VERIFY_BENCHMARK

#define CURAND_CHECK(expr)                                              \
  {                                                                     \
    auto status = expr;                                                 \
    if (status != CURAND_STATUS_SUCCESS) {                              \
      std::cerr << "failed with status: " << status << std::endl;       \
      throw std::runtime_error("cuRAND expression (" #expr ") failed"); \
    }                                                                   \
  }

#define CUFFT_CHECK(expr)                                              \
  {                                                                    \
    auto status = expr;                                                \
    if (status != CUFFT_SUCCESS) {                                     \
      std::cerr << "failed with status: " << status << std::endl;       \
      throw std::runtime_error("cuFFT expression (" #expr ") failed"); \
    }                                                                  \
  }

#define CUFFT_CHECK_NO_THROW(expr)                                \
  {                                                               \
    auto status = expr;                                           \
    if (status != CUFFT_SUCCESS) {                                \
      std::cerr << "failed with status: " << status << std::endl;       \
      state.SkipWithError("cuFFT expression (" #expr ") failed"); \
    }                                                             \
  }

#define CUDA_CHECK(expr)                                              \
  {                                                                   \
    auto status = expr;                                               \
    if (status != cudaSuccess) {                                      \
      std::cerr << "failed with status: " << status << std::endl;       \
      throw std::runtime_error("CUDA expression (" #expr ") failed"); \
    }                                                                 \
  }

#define CUDA_CHECK_NO_THROW(expr)                                 \
  {                                                               \
    auto status = expr;                                           \
    if (status != cudaSuccess) {                                  \
      std::cerr << "failed with status: " << status << std::endl; \
      state.SkipWithError("CUDA expression (" #expr ") failed");  \
    }                                                             \
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
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
  CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));

  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234));

  if constexpr (std::is_same_v<T, float>) {
    CURAND_CHECK(curandGenerateNormal(generator, dev_ptr, N, 0.f, 2.f));
  } else {
    CURAND_CHECK(curandGenerateNormalDouble(generator, dev_ptr, N, 0.0, 2.0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CURAND_CHECK(curandDestroyGenerator(generator));
}
#endif  // SYCLFFT_VERIFY_BENCHMARK

#endif  // SYCL_FFT_BENCH_CUFFT_UTILS_HPP
