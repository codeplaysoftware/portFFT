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

#ifndef SYCL_FFT_BENCH_LAUNCH_BENCH_HPP
#define SYCL_FFT_BENCH_LAUNCH_BENCH_HPP

#include <benchmark/benchmark.h>
#include <descriptor.hpp>
#include <type_traits.hpp>

#include "number_generators.hpp"
#include "ops_estimate.hpp"

template <typename ftype, sycl_fft::domain domain>
void bench_dft_real_time(benchmark::State& state,
                         sycl_fft::descriptor<ftype, domain> desc) {
  using complex_type = std::complex<ftype>;
  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::vector<complex_type> a(N * N_transforms);
  populate_with_random(a);

  sycl::queue q;
  complex_type* a_dev = sycl::malloc_device<complex_type>(N * N_transforms, q);
  q.copy(a.data(), a_dev, N * N_transforms);

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

template <typename ftype, sycl_fft::domain domain>
void bench_dft_device_time(benchmark::State& state,
                           sycl_fft::descriptor<ftype, domain> desc) {
  using complex_type = std::complex<ftype>;
  std::size_t N = desc.get_total_length();
  std::size_t N_transforms = desc.number_of_transforms;
  double ops = cooley_tukey_ops_estimate(N, N_transforms);
  std::vector<complex_type> a(N * N_transforms);
  populate_with_random(a);

  sycl::queue q({sycl::property::queue::enable_profiling()});
  complex_type* a_dev = sycl::malloc_device<complex_type>(N * N_transforms, q);
  q.copy(a.data(), a_dev, N * N_transforms);

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

template <typename ftype, sycl_fft::domain domain>
sycl_fft::descriptor<ftype, domain> create_descriptor(benchmark::State& state) {
  std::size_t N = state.range(0);
  sycl_fft::descriptor<ftype, sycl_fft::domain::COMPLEX> desc{{N}};
  desc.number_of_transforms = state.range(1);
  return desc;
}

template <typename T>
void bench_dft_real_time(benchmark::State& state) {
  using ftype = typename sycl_fft::get_real<T>::type;
  constexpr sycl_fft::domain domain = sycl_fft::get_domain<T>::value;
  auto desc = create_descriptor<ftype, domain>(state);
  bench_dft_real_time<ftype, domain>(state, desc);
}

template <typename T>
void bench_dft_device_time(benchmark::State& state) {
  using ftype = typename sycl_fft::get_real<T>::type;
  constexpr sycl_fft::domain domain = sycl_fft::get_domain<T>::value;
  auto desc = create_descriptor<ftype, domain>(state);
  bench_dft_device_time<ftype, domain>(state, desc);
}

#endif
