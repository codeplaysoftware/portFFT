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
 *  Codeplay's portFFT
 *
 **************************************************************************/

#ifndef PORTFFT_BENCH_BENCH_UTILS_HPP
#define PORTFFT_BENCH_BENCH_UTILS_HPP

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef PORTFFT_VERIFY_BENCHMARKS
#include "common/reference_data_wrangler.hpp"
#endif  // PORTFFT_VERIFY_BENCHMARKS

/**
 * number of runs to do when doing an average of many host runs.
 */
static constexpr std::size_t runs_to_average = 10;

/**
 * Get the number of inputs to allocate for the average_host benchmark.
 * Try to use \p target_num_inputs distinct inputs so that each call to compute uses a different input and avoids
 * relying on cache. We allow to allocate up to 90% of the global memory for inputs and outputs. Multiple inputs are
 * needed to avoid affecting the timings.
 *
 * @param input_size_bytes Size of the FFT input in bytes
 * @param output_size_bytes Upper bound estimation of the size of the FFT output in bytes
 * @param global_mem_size Global memory size available on the device
 * @param target_num_inputs Target number of inputs
 */
std::size_t get_average_host_num_inputs(std::size_t input_size_bytes, std::size_t output_size_bytes,
                                        std::size_t global_mem_size, std::size_t target_num_inputs) {
  const std::size_t desired_allocation_size = input_size_bytes * target_num_inputs + output_size_bytes;
  const std::size_t allocation_size_threshold = static_cast<std::size_t>(0.9 * static_cast<double>(global_mem_size));
  const std::size_t num_inputs = desired_allocation_size <= allocation_size_threshold ? target_num_inputs : 1;
  if (num_inputs < target_num_inputs) {
    std::cerr << "Warning: Not enough global memory to allocate " << target_num_inputs
              << " input(s). The results may appear better than they would be in a real application due to the "
                 "device's cache."
              << std::endl;
  }
  return num_inputs;
}

// Handle an exception by passing the message onto `SkipWithError`.
// It is expected that this will be placed so the benchmark ends after this is called,
// allowing the test to exit gracefully with an error message before moving onto the next test.
inline void handle_exception(benchmark::State& state, std::exception& e) {
  std::string msg{"Exception thrown: "};
  msg += e.what();
  state.SkipWithError(msg.c_str());
}

#endif  // PORTFFT_BENCH_BENCH_UTILS_HPP
