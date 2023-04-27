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

#ifndef SYCL_FFT_BENCH_OPS_ESTIMATE_HPP
#define SYCL_FFT_BENCH_OPS_ESTIMATE_HPP

#include <cmath>

/**
 * Estimates the number of operations required to compute the FFT.
 * The estimate is based on radix-2 decimation in time Cooley-Tukey.
 * @param fft_size size of the FFT problem
 * @param batches number of batches computed. Defaults to 1.
 * @return estimated number of operations to compute FFT. Returns a double to
 * avoid rounding.
 */
inline double cooley_tukey_ops_estimate(int fft_size, int batches = 1) {
  return 5 * batches * fft_size * std::log2(static_cast<double>(fft_size));
}

/**
 * Calculates the number of Memory transactions in bytes.
 * Assumes load from global memory only once.
 * @tparam TypeIn Input Type
 * @tparam TypeOut Output Type
 * @param fft_size size of the FFT Problem
 * @param batches Number of batches computed.
 * @return size_t Number of memory transactions in bytes.
 */
template <typename TypeIn, typename TypeOut>
inline size_t global_mem_transactions(int batches, int num_in, int num_out) {
  return batches * (sizeof(TypeIn) * num_in + sizeof(TypeOut) * num_out);
}

#endif  // SYCL_FFT_BENCH_OPS_ESTIMATE_HPP
