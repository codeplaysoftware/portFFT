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

#include <benchmark/benchmark.h>
#include <descriptor.hpp>

#include "number_generators.hpp"
#include "ops_estimate.hpp"

#include <complex>
#include <iostream>

constexpr int N_transforms = 1024 * 512;

template <typename ftype>
static void BM_dft_real_time(benchmark::State& state) {
  using complex_type = std::complex<ftype>;
  size_t N = state.range(0);
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::vector<complex_type> a(N * N_transforms);
  populate_with_random(a);

  sycl::queue q;
  complex_type* a_dev = sycl::malloc_device<complex_type>(N * N_transforms, q);
  q.copy(a.data(), a_dev, N * N_transforms);

  sycl_fft::descriptor<ftype, sycl_fft::domain::COMPLEX> desc{{N}};
  desc.number_of_transforms = N_transforms;
  auto committed = desc.commit(q);

  q.wait();

  // warmup
  committed.compute_forward(a_dev).wait();

  for (auto _ : state) {
    // we need to manually measure time, so as to have it available here for the
    // calculation of flops
    auto start = std::chrono::high_resolution_clock::now();
    committed.compute_forward(a_dev).wait();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(a_dev, q);
}

template <typename ftype>
static void BM_dft_device_time(benchmark::State& state) {
  using complex_type = std::complex<ftype>;
  size_t N = state.range(0);
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::vector<complex_type> a(N * N_transforms);
  populate_with_random(a);

  sycl::queue q({sycl::property::queue::enable_profiling()});
  complex_type* a_dev = sycl::malloc_device<complex_type>(N * N_transforms, q);
  q.copy(a.data(), a_dev, N * N_transforms);

  sycl_fft::descriptor<ftype, sycl_fft::domain::COMPLEX> desc{{N}};
  desc.number_of_transforms = N_transforms;
  auto committed = desc.commit(q);

  q.wait();

  // warmup
  committed.compute_forward(a_dev).wait();

  for (auto _ : state) {
    sycl::event e = committed.compute_forward(a_dev);
    e.wait();
    int64_t start =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    int64_t end =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
    double elapsed_seconds = (end - start) / 1e9;
    state.counters["flops"] = ops / elapsed_seconds;
    state.SetIterationTime(elapsed_seconds);
  }
  sycl::free(a_dev, q);
}

// 29 / 13 are max sizes that fit into integrated GPU (128 32b registers)
BENCHMARK(BM_dft_real_time<float>)
    ->UseManualTime()
    ->Arg(8)
    ->Arg(16)
    ->Arg(17)
    ->Arg(27)
    ->Arg(29);
BENCHMARK(BM_dft_device_time<float>)
    ->UseManualTime()
    ->Arg(8)
    ->Arg(16)
    ->Arg(17)
    ->Arg(27)
    ->Arg(29);
BENCHMARK(BM_dft_real_time<double>)->UseManualTime()->Arg(8)->Arg(12)->Arg(13);
BENCHMARK(BM_dft_device_time<double>)
    ->UseManualTime()
    ->Arg(8)
    ->Arg(12)
    ->Arg(13);

BENCHMARK_MAIN();
