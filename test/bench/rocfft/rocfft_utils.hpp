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

#ifndef SYCL_FFT_BENCH_ROCFFT_UTILS_HPP
#define SYCL_FFT_BENCH_ROCFFT_UTILS_HPP

#include <hip/hip_runtime_api.h>
#ifdef SYCLFFT_VERIFY_BENCHMARK
#include <rocrand.h>
#endif  // SYCLFFT_VERIFY_BENCHMARK

#define ROCFFT_CHECK(expr)                                                                                    \
  {                                                                                                           \
    auto status = expr;                                                                                       \
    if (status != rocfft_status_success) {                                                                    \
      throw std::runtime_error("rocFFT expression (" #expr ") failed with status " + std::to_string(status)); \
    }                                                                                                         \
  }

#define ROCFFT_CHECK_NO_THROW(expr)                                                                      \
  {                                                                                                      \
    auto status = expr;                                                                                  \
    if (status != rocfft_status_success) {                                                               \
      std::string msg = "rocFFT expression (" #expr ") failed with status " + std::to_string(status); \ 
      state.SkipWithError(msg.c_str()); \
    }                                                                                                    \
  }

#define HIP_CHECK(expr)                                                                                    \
  {                                                                                                        \
    auto status = expr;                                                                                    \
    if (status != hipSuccess) {                                                                            \
      throw std::runtime_error("HIP expression (" #expr ") failed with status " + std::to_string(status)); \
    }                                                                                                      \
  }

#define HIP_CHECK_NO_THROW(expr)                                                                   \
  {                                                                                                \
    auto status = expr;                                                                            \
    if (status != hipSuccess) {                                                                    \
      std::string msg = "HIP expression (" #expr ") failed with status " + std::to_string(status); \ 
      state.SkipWithError(msg.c_str());                                                            \
    }                                                                                              \
  }

#ifdef SYCLFFT_VERIFY_BENCHMARK
#define ROCRAND_CHECK(expr)                                                                                    \
  {                                                                                                            \
    auto status = expr;                                                                                        \
    if (status != ROCRAND_STATUS_SUCCESS) {                                                                    \
      throw std::runtime_error("rocRAND expression (" #expr ") failed with status " + std::to_string(status)); \
    }                                                                                                          \
  }

/**
 * @brief populates device ptr with random values using the rocrand host api
 *
 * @tparam T Type of device pointer, must be either float or double
 * @param dev_ptr Device Pointer to be filled
 * @param N Number of elements to generate and store to dev_ptr
 */
template <typename T>
void roc_populate_with_random(T* dev_ptr, std::size_t N) {
  rocrand_generator generator;
  ROCRAND_CHECK(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_XORWOW));

  ROCRAND_CHECK(rocrand_set_seed(generator, 5678));

  if constexpr (std::is_same_v<T, float>) {
    ROCRAND_CHECK(rocrand_generate_normal(generator, dev_ptr, N, 0.f, 2.f));
  } else {
    static_assert(std::is_same_v<T, double>);
    ROCRAND_CHECK(rocrand_generate_normal_double(generator, dev_ptr, N, 0., 2.));
  }

  HIP_CHECK(hipStreamSynchronize(nullptr));
  ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

#undef ROCRAND_CHECK
#endif  // SYCLFFT_VERIFY_BENCHMARK

#endif  // SYCL_FFT_BENCH_ROCFFT_UTILS_HPP
