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
 *  A set of common reference DFT benchmarks.
 *
 **************************************************************************/
#ifndef PORTFFT_REFERENCE_DFT_SET_HPP
#define PORTFFT_REFERENCE_DFT_SET_HPP

/**
 * A common set of reference benchmarks. To use, two macros must be defined:
 * BENCH_COMPLEX_FLOAT(...)
 * BENCH_SINGLE_FLOAT(...)
 * and then the benchmarks can be instantiated:
 * INSTANTIATE_REFERENCE_BENCHMARK_SET(BENCH_COMPLEX_FLOAT, BENCH_SINGLE_FLOAT)
 * See pre-existing benchmark implementations for examples of what these
 * macros should do.
 **/

#include <benchmark/benchmark.h>

#include <vector>

// clang-format off
// Forward, float, out-of-place only:
// 1. small        complex 1D fits in workitem Cooley-Tukey        (batch=8*1024*1024 N=16)
// 2. medium-small complex 1D fits in subgroup Cooley-Tukey        (batch=512*1024    N=256)
// 3. medium-large complex 1D fits in local memory Cooley-Tukey    (batch=32*1024     N=4*1024)
// 4. large        complex 1D fits in global memory Cooley-Tukey   (batch=2*1024      N=64*1024)
// 5. large        complex 1D fits in global memory Bluestein      (batch=2*1024      N=64*1024+1)
// 6. small        real    1D fits in workitem Cooley-Tukey        (batch=8*1024*1024 N=32)
// 7. medium-small real    1D fits in subgroup Cooley-Tukey        (batch=512*1024    N=512)
// 8. medium-large real    1D fits in local memory Cooley-Tukey    (batch=32*1024     N=8*1024)
// 9. large        real    1D fits in global memory Cooley-Tukey   (batch=2*1024      N=128*1024)
//
// Configurations must match with the ones in test/bench/portfft/bench_float.cpp
// clang-format on

/**
 * Helper function to register a single benchmark
 *
 * @param prefix Benchmark prefix, expected to end with the description of the FFT problem. The lengths and batch are
 * appended to this description.
 * @param suffix Benchmark suffix (i.e. a short name)
 * @param args Function to benchmark followed by optional arguments like a SYCL queue
 * @param lengths FFT lengths
 * @param batch FFT batch size
 */
template <typename... Args>
void register_benchmark(std::string prefix, const std::string& suffix, Args&&... args,
                        const std::vector<std::size_t>& lengths, std::size_t batch) {
  prefix += ",n=[";
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    if (i > 0) {
      prefix += ", ";
    }
    prefix += std::to_string(lengths[i]);
  }
  prefix += "],batch=";
  prefix += std::to_string(batch);
  prefix += "/" + suffix;
  benchmark::RegisterBenchmark(prefix.c_str(), args..., lengths, batch)->UseManualTime();
}

/**
 * Register benchmarks for complex float configurations
 *
 * @param prefix Prefix to the benchmarks' name
 * @param args The first argument is the function to benchmark.
 *             Followed by optional arguments forwarded to the benchmark function (i.e. a SYCL queue).
 */
template <typename... Args>
void register_complex_float_benchmark_set(std::string prefix, Args&&... args) {
  prefix += "/d=cpx,prec=single";
  // clang-format off
  register_benchmark<Args...>(prefix, "small_1d",        args..., {16},            8 * 1024 * 1024);
  register_benchmark<Args...>(prefix, "medium_small_1d", args..., {256},           512 * 1024);
  register_benchmark<Args...>(prefix, "medium_large_1d", args..., {4 * 1024},      32 * 1024);
  register_benchmark<Args...>(prefix, "large_1d",        args..., {64 * 1024},     2 * 1024);
  register_benchmark<Args...>(prefix, "large_1d_prime",  args..., {64 * 1024 + 1}, 2 * 1024);
  // clang-format on
}

/**
 * Register benchmarks for real float configurations
 *
 * @param prefix Prefix to the benchmarks' name
 * @param args The first argument is the function to benchmark.
 *             Followed by optional arguments forwarded to the benchmark function (i.e. a SYCL queue).
 */
template <typename... Args>
void register_real_float_benchmark_set(std::string prefix, Args&&... args) {
  prefix += "/d=re,prec=single";
  // clang-format off
  register_benchmark<Args...>(prefix, "small_1d",        args..., {32},         8 * 1024 * 1024);
  register_benchmark<Args...>(prefix, "medium_small_1d", args..., {512},        512 * 1024);
  register_benchmark<Args...>(prefix, "medium_large_1d", args..., {8 * 1024},   32 * 1024);
  register_benchmark<Args...>(prefix, "large_1d",        args..., {128 * 1024}, 2 * 1024);
  // clang-format on
}

#endif  // PORTFFT_REFERENCE_DFT_SET_HPP
