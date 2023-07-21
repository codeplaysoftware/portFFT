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

#include "enums.hpp"

#ifdef PORTFFT_VERIFY_BENCHMARK
// The following file in generated during the build and located at
// ${BUILD_DIR}/ref_data_include/
#include <benchmark_reference.hpp>
#endif  // PORTFFT_VERIFY_BENCHMARK

/**
 * number of runs to do when doing an average of many host runs.
 */
static constexpr std::size_t runs_to_average = 10;

// Handle an exception by passing the message onto `SkipWithError`.
// It is expected that this will be placed so the benchmark ends after this is called,
// allowing the test to exit gracefully with an error message before moving onto the next test.
inline void handle_exception(benchmark::State& state, std::exception& e) {
  std::string msg{"Exception thrown: "};
  msg += e.what();
  state.SkipWithError(msg.c_str());
}

#endif  // PORTFFT_BENCH_BENCH_UTILS_HPP
